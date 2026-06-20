#!/usr/bin/env python3
"""Turn a YouTube link into a readable transcript and a detailed summary.

Usage:
    python yt_summarize.py "https://www.youtube.com/watch?v=..." [options]

Pipeline:
    URL -> yt-dlp (English captions + metadata)
        -> parse VTT + de-duplicate rolling captions  -> raw phrase lines
        -> OpenAI cleanup pass (single call; chunked   -> <slug>.transcript.md
           only for very long transcripts)
        -> OpenAI summary pass                          -> <slug>.summary.md

Requires `yt-dlp` on PATH and the `openai` package, plus OPENAI_API_KEY in the
environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# A current large-context OpenAI chat model. Override with --model or OPENAI_MODEL
# if your account prefers/needs a different one (e.g. gpt-4o).
DEFAULT_MODEL = "gpt-5.4-mini"

# Clean the whole transcript in one call when it fits comfortably under the
# model's output cap (gpt-5.4-mini: 128k). ~45k words of cleaned text plus
# reasoning stays well under that. Longer transcripts fall back to chunking.
SINGLE_PASS_MAX_WORDS = 45000

# Words per cleanup chunk for the long-transcript fallback path. Cleanup output
# is roughly as long as its input, so chunks keep each call's output bounded.
CHUNK_WORDS = 2800

# Live model-pricing registry (LiteLLM's community-maintained price list). Fetched
# once and cached, so we don't hardcode rates that go stale. Override per run with
# PRICE_INPUT / PRICE_OUTPUT (dollars per 1M tokens); offline, fall back to the
# small table below.
PRICE_REGISTRY_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
PRICE_CACHE = Path(tempfile.gettempdir()) / "yt_summarize_model_prices.json"
PRICE_CACHE_TTL = 24 * 3600  # seconds

# Offline fallback, dollars per 1M tokens as (input, output). Output includes
# reasoning tokens. Only consulted if the registry is unreachable and no override.
PRICING_FALLBACK = {
    "gpt-5.4-mini": (0.75, 4.50),
}

# Accumulated token usage across all OpenAI calls in a run.
USAGE = {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}

CLEANUP_SYSTEM = (
    "You restore readability to raw, auto-generated speech-to-text. "
    "Add correct punctuation, capitalization, and paragraph breaks. "
    "Fix obvious transcription artifacts (stray casing, run-ons). "
    "Do NOT add, remove, reword, summarize, translate, or reorder the content - "
    "output the same words the speaker said, only made readable. "
    "Return only the cleaned text, with no preamble or commentary."
)

SUMMARY_SYSTEM = (
    "You are an expert note-taker who writes thorough, faithful summaries of "
    "talks and interviews for someone who wants to absorb everything that was "
    "said without watching the whole thing."
)


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def load_dotenv() -> None:
    """Load KEY=VALUE pairs from a .env file (script dir, then cwd) into the
    environment, without overriding variables already set. Minimal parser, no
    dependency."""
    for path in (Path(__file__).resolve().parent / ".env", Path.cwd() / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


# --------------------------------------------------------------------------- #
# yt-dlp: metadata + captions
# --------------------------------------------------------------------------- #

def _run_ytdlp(args: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["yt-dlp", *args],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        die("yt-dlp not found on PATH. Install it: pip install -r requirements.txt")


def fetch_metadata(url: str) -> dict:
    """Return id/title/channel/upload_date/duration for the video."""
    sep = "\x1f"  # unit separator, unlikely to appear in titles
    fmt = sep.join(
        f"%({f})s" for f in ("id", "title", "channel", "upload_date", "duration")
    )
    proc = _run_ytdlp(["--skip-download", "--no-warnings", "--print", fmt, url])
    if proc.returncode != 0:
        die(f"yt-dlp could not read the video:\n{proc.stderr.strip()}")
    line = proc.stdout.strip().splitlines()[0]
    vid, title, channel, upload_date, duration = (line.split(sep) + [""] * 5)[:5]
    return {
        "id": vid,
        "title": title or vid,
        "channel": channel,
        "upload_date": _fmt_date(upload_date),
        "duration": _fmt_duration(duration),
        "url": f"https://www.youtube.com/watch?v={vid}" if vid else url,
    }


def fetch_captions(url: str, tmp: Path) -> str:
    """Download English captions (manual preferred, else auto) and return the VTT text."""
    proc = _run_ytdlp([
        "--skip-download",
        "--no-warnings",
        "--write-subs",        # human-authored track wins if present
        "--write-auto-subs",   # fall back to auto-generated
        "--sub-langs", "en.*,en",
        "--sub-format", "vtt/best",
        "-o", str(tmp / "%(id)s.%(ext)s"),
        url,
    ])
    vtts = sorted(tmp.glob("*.vtt"))
    if not vtts:
        extra = f"\n{proc.stderr.strip()}" if proc.stderr.strip() else ""
        die(
            "no English captions available for this video. "
            "Only videos with English (manual or auto-generated) captions are "
            f"supported.{extra}"
        )
    # Prefer a manual track (filename without the auto-only language tags) when
    # several exist; otherwise just take the first.
    return vtts[0].read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# VTT parsing + rolling-caption de-duplication
# --------------------------------------------------------------------------- #

_TS_LINE = re.compile(r"-->")
_TAG = re.compile(r"<[^>]+>")            # <00:00:01.234>, <c>, </c>
_CUE_SETTINGS = re.compile(r"\s+(align|position|line|size):\S+")


def parse_vtt(vtt: str) -> list[str]:
    """Extract spoken phrase lines from a VTT, de-duplicating rolling captions.

    YouTube auto-captions repeat each line as the next caption scrolls up, so a
    naive read produces every phrase 2-3 times. Stripping inline tags and then
    collapsing consecutive duplicate lines yields one clean copy of each phrase.
    """
    lines: list[str] = []
    for raw in vtt.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in ("WEBVTT",) or _TS_LINE.search(line):
            continue
        if line.startswith(("Kind:", "Language:", "NOTE")):
            continue
        if line.isdigit():  # numeric cue index (some VTTs include them)
            continue
        text = _TAG.sub("", line)
        text = _CUE_SETTINGS.sub("", text).strip()
        if not text:
            continue
        if lines and text == lines[-1]:  # collapse rolling duplicates
            continue
        lines.append(text)
    return lines


def chunk_lines(lines: list[str], max_words: int = CHUNK_WORDS) -> list[str]:
    """Group phrase lines into text chunks under a word budget, breaking only
    between lines so chunk seams fall at natural phrase boundaries."""
    chunks: list[str] = []
    buf: list[str] = []
    count = 0
    for line in lines:
        w = len(line.split())
        if buf and count + w > max_words:
            chunks.append(" ".join(buf))
            buf, count = [], 0
        buf.append(line)
        count += w
    if buf:
        chunks.append(" ".join(buf))
    return chunks


# --------------------------------------------------------------------------- #
# OpenAI passes
# --------------------------------------------------------------------------- #

def make_client():
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        die("OPENAI_API_KEY is not set. Put it in a .env file or: export OPENAI_API_KEY=sk-...")
    try:
        from openai import OpenAI
    except ImportError:
        die("openai package not installed. Run: pip install -r requirements.txt")
    return OpenAI()


# Reasoning models (gpt-5 family, o-series) only accept the default temperature.
# We try a low temperature for fidelity and remember if the model rejects it.
_SUPPORTS_TEMPERATURE = True


def _chat(client, model: str, system: str, user: str, max_tokens: int) -> str:
    from openai import OpenAIError
    global _SUPPORTS_TEMPERATURE

    base = dict(
        model=model,
        max_completion_tokens=max_tokens,  # current param; gpt-5/o-series require it
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    def call(with_temp: bool):
        kw = dict(base)
        if with_temp:
            kw["temperature"] = 0.2
        return client.chat.completions.create(**kw)

    try:
        resp = call(_SUPPORTS_TEMPERATURE)
    except OpenAIError as e:
        if _SUPPORTS_TEMPERATURE and "temperature" in str(e).lower():
            _SUPPORTS_TEMPERATURE = False  # model only allows default; retry without
            try:
                resp = call(False)
            except OpenAIError as e2:
                die(f"OpenAI request failed ({type(e2).__name__}): {e2}")
        else:
            die(f"OpenAI request failed ({type(e).__name__}): {e}")
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        print(
            "  warning: output hit the token cap and may be truncated.",
            file=sys.stderr,
        )
    u = getattr(resp, "usage", None)
    if u:
        USAGE["prompt"] += getattr(u, "prompt_tokens", 0) or 0
        USAGE["completion"] += getattr(u, "completion_tokens", 0) or 0
        ptd = getattr(u, "prompt_tokens_details", None)
        if ptd:
            USAGE["cached"] += getattr(ptd, "cached_tokens", 0) or 0
        ctd = getattr(u, "completion_tokens_details", None)
        if ctd:
            USAGE["reasoning"] += getattr(ctd, "reasoning_tokens", 0) or 0
    return (choice.message.content or "").strip()


def ai_clean(client, model: str, lines: list[str]) -> str:
    """Restore readability. Single pass when the transcript fits the output cap;
    chunked fallback only for very long transcripts."""
    full = " ".join(lines)
    word_count = len(full.split())

    if word_count <= SINGLE_PASS_MAX_WORDS:
        print("  cleaning in a single pass...", file=sys.stderr)
        return _chat(
            client, model, CLEANUP_SYSTEM,
            f"Clean up this entire transcript:\n\n{full}",
            max_tokens=128000,
        )

    chunks = chunk_lines(lines)
    print(
        f"  transcript is very long ({word_count} words); cleaning in "
        f"{len(chunks)} chunks...",
        file=sys.stderr,
    )
    cleaned: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  cleaning chunk {i}/{len(chunks)}...", file=sys.stderr)
        cleaned.append(_chat(
            client, model, CLEANUP_SYSTEM,
            f"Clean up this transcript segment:\n\n{chunk}",
            max_tokens=16000,
        ))
    return "\n\n".join(cleaned)


def ai_summarize(client, model: str, transcript: str, meta: dict) -> str:
    prompt = (
        f"Below is the full transcript of a talk/interview titled "
        f"\"{meta['title']}\""
        + (f" by {meta['channel']}" if meta["channel"] else "")
        + ".\n\nWrite a faithful summary in Markdown.\n\n"
        "Requirements:\n"
        "- LENGTH: keep it tight and clearly shorter than the source — distill, "
        "don't transcribe. Compress aggressively; never pad, restate, or add "
        "filler.\n"
        "- COVERAGE: include every substantive point, argument, example, and "
        "conclusion, but drop filler, repetition, and digressions. A reader "
        "should learn essentially everything that was said.\n"
        "- STRUCTURE: start with an `## Overview` (2-3 sentences on the talk and "
        "its central thrust). Then use only as many `## <topic>` sections as the "
        "material genuinely warrants, following the talk's flow — a short talk may "
        "need just one or two. End with a `## Key takeaways` bullet list.\n"
        "- Do not invent anything not in the transcript. Output only the Markdown "
        "(start at the `## Overview` heading).\n\n"
        f"---\nTRANSCRIPT:\n{transcript}"
    )
    return _chat(client, model, SUMMARY_SYSTEM, prompt, max_tokens=32000)


# --------------------------------------------------------------------------- #
# Formatting / output
# --------------------------------------------------------------------------- #

def _fmt_date(yyyymmdd: str) -> str:
    if re.fullmatch(r"\d{8}", yyyymmdd or ""):
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
    return ""


def _fmt_duration(seconds: str) -> str:
    try:
        s = int(float(seconds))
    except (ValueError, TypeError):
        return ""
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:80].strip("-") or "transcript"


def header(meta: dict, kind: str, words: int | None = None) -> str:
    bits = [f"# {meta['title']} — {kind}", ""]
    if meta["channel"]:
        bits.append(f"- **Channel:** {meta['channel']}")
    if meta["upload_date"]:
        bits.append(f"- **Published:** {meta['upload_date']}")
    if meta["duration"]:
        bits.append(f"- **Duration:** {meta['duration']}")
    bits.append(f"- **Source:** {meta['url']}")
    if words is not None:
        bits.append(f"- **Words:** {words:,}")
    bits.append("")
    bits.append("---")
    bits.append("")
    return "\n".join(bits)


def _load_price_registry() -> dict:
    """Return the LiteLLM model-price registry, fetched once and cached locally."""
    try:
        if PRICE_CACHE.is_file() and time.time() - PRICE_CACHE.stat().st_mtime < PRICE_CACHE_TTL:
            return json.loads(PRICE_CACHE.read_text(encoding="utf-8"))
    except Exception:
        pass
    try:
        with urllib.request.urlopen(PRICE_REGISTRY_URL, timeout=10) as r:
            data = r.read().decode("utf-8")
        json.loads(data)  # validate before caching
        try:
            PRICE_CACHE.write_text(data, encoding="utf-8")
        except OSError:
            pass
        return json.loads(data)
    except Exception:
        # Network failed — a stale cache still beats nothing.
        try:
            if PRICE_CACHE.is_file():
                return json.loads(PRICE_CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def model_rates(model: str) -> tuple[float, float, str] | None:
    """Resolve (input, output) dollars-per-1M-tokens and the source used."""
    env_in, env_out = os.environ.get("PRICE_INPUT"), os.environ.get("PRICE_OUTPUT")
    if env_in and env_out:
        try:
            return float(env_in), float(env_out), "env override"
        except ValueError:
            pass
    entry = _load_price_registry().get(model)
    if entry and entry.get("input_cost_per_token") is not None:
        return (
            entry["input_cost_per_token"] * 1e6,
            (entry.get("output_cost_per_token") or 0.0) * 1e6,
            "live registry",
        )
    fb = PRICING_FALLBACK.get(model)
    if fb:
        return fb[0], fb[1], "offline fallback"
    return None


def print_cost(model: str) -> None:
    inp, out = USAGE["prompt"], USAGE["completion"]
    extras = []
    if USAGE["reasoning"]:
        extras.append(f"{USAGE['reasoning']:,} reasoning")
    if USAGE["cached"]:
        extras.append(f"{USAGE['cached']:,} cached")
    note = f" ({', '.join(extras)})" if extras else ""
    print(f"Tokens:     {inp:,} in + {out:,} out{note} = {inp + out:,} total")

    rates = model_rates(model)
    if rates:
        in_rate, out_rate, src = rates
        cost = inp / 1e6 * in_rate + out / 1e6 * out_rate
        print(
            f"Est. cost:  ${cost:.4f}  "
            f"({model} @ ${in_rate:g}/${out_rate:g} per 1M in/out, via {src})"
        )
    else:
        print(
            f"Est. cost:  n/a (no pricing for '{model}' — set PRICE_INPUT/PRICE_OUTPUT "
            "or add it to PRICING_FALLBACK)"
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="YouTube link -> readable transcript + detailed summary (Markdown)."
    )
    ap.add_argument("url", help="YouTube video URL")
    ap.add_argument("--outdir", default="output", help="output directory (default: output)")
    ap.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"OpenAI model (default: {DEFAULT_MODEL} or $OPENAI_MODEL)",
    )
    ap.add_argument("--keep-raw", action="store_true",
                    help="also write the un-cleaned mechanical transcript")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Fetching metadata...", file=sys.stderr)
    meta = fetch_metadata(args.url)

    print("Fetching captions...", file=sys.stderr)
    with tempfile.TemporaryDirectory() as td:
        vtt = fetch_captions(args.url, Path(td))
    lines = parse_vtt(vtt)
    if not lines:
        die("captions were downloaded but no spoken text could be extracted.")
    raw_text = " ".join(lines)
    print(f"  extracted ~{len(raw_text.split())} words.", file=sys.stderr)

    client = make_client()

    print(f"Cleaning transcript with {args.model}...", file=sys.stderr)
    transcript = ai_clean(client, args.model, lines)

    print("Summarizing...", file=sys.stderr)
    summary = ai_summarize(client, args.model, transcript, meta)

    t_words = len(transcript.split())
    s_words = len(summary.split())

    slug = slugify(meta["title"])
    t_path = outdir / f"{slug}.transcript.md"
    s_path = outdir / f"{slug}.summary.md"
    t_path.write_text(header(meta, "Transcript", t_words) + transcript + "\n", encoding="utf-8")
    s_path.write_text(header(meta, "Summary", s_words) + summary + "\n", encoding="utf-8")

    if args.keep_raw:
        r_path = outdir / f"{slug}.raw.txt"
        r_path.write_text(raw_text + "\n", encoding="utf-8")
        print(f"Wrote {r_path}", file=sys.stderr)

    compression = f"{s_words / t_words:.0%}" if t_words else "n/a"
    print(f"\nTranscript: {t_path}  ({t_words:,} words)")
    print(f"Summary:    {s_path}  ({s_words:,} words, {compression} of transcript)")
    print()
    print_cost(args.model)


if __name__ == "__main__":
    main()
