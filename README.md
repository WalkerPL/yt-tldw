# yt-tldw

*Too Long; Didn't Watch* — give it a YouTube link (talks, interviews, lectures)
and get back two Markdown files:

- `output/<slug>.transcript.md` — the English captions cleaned into readable,
  punctuated prose.
- `output/<slug>.summary.md` — a detailed, sectioned summary (Overview →
  topic-by-topic sections → Key takeaways) that covers everything said but is far
  shorter than the talk. Section headings and specific points carry **clickable
  timestamps** that open the video at that moment, so you can jump straight to the
  source.

The AI passes (transcript cleanup + summary) use the **OpenAI API**.

## Example output

From the 6-minute talk *"How to sound smart in your TEDx Talk"* — see the full
generated summary in [examples/how-to-sound-smart.summary.md](examples/how-to-sound-smart.summary.md):

```markdown
## Overview

Will Stephen's talk is a deliberately empty performance about how speakers can
sound intelligent without actually saying anything. By openly demonstrating the
tricks of confident delivery—gestures, pacing, visuals, fake data, and rhetorical
buildup—he shows how style can create the illusion of substance.

## [1:02](https://www.youtube.com/watch?v=8S0FDjFBj8o&t=62s) How performance creates the illusion of meaning

He begins "the opening" with deliberate hand gestures, glasses-adjusting, and a
show-of-hands question to create the feel of engagement [1:02](...&t=62s). He then
pretends to share a personal anecdote ... gesturing toward a scientist image he
admits he found by googling "Scientist" [1:30](...&t=90s).
...
## Key takeaways

- Confident delivery can create the illusion of intelligence even when the content is empty.
- Charts, numbers, and visuals can function as authority signals even when they add no real substance.
- ...
```

Headings and specific points carry clickable timestamps linking back to that
moment in the video.

## Requirements

- Python 3.9+
- An OpenAI API key

(`yt-dlp` is installed by the requirements file — no separate media tooling needed,
since only captions are downloaded.)

## Setup

```bash
pip install -r requirements.txt        # installs openai + yt-dlp
cp .env.example .env                   # then put your key in .env (gitignored)
# ...or instead: export OPENAI_API_KEY=sk-...
```

The key is read from a `.env` file in the project directory or from the
environment.

## Usage

```bash
python yt_summarize.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Options:

| Flag | Default | Purpose |
|------|---------|---------|
| `--outdir DIR` | `output` | where the `.md` files are written |
| `--model NAME` | `gpt-5.4-mini` (or `$OPENAI_MODEL`) | OpenAI chat model for both passes |
| `--keep-raw` | off | also write the un-cleaned mechanical transcript (`.raw.txt`) |

`OPENAI_API_KEY` is read from a `.env` file in the project directory (or the
environment). Pick a different model if `gpt-5.4-mini` isn't enabled on your
account, e.g.:

```bash
python yt_summarize.py "<url>" --model gpt-4o
# or
export OPENAI_MODEL=gpt-4o
```

On completion the script prints the output paths with word counts (and the
summary's size as a percentage of the transcript), plus a token and estimated
cost breakdown:

```
Transcript: output/....transcript.md  (17,402 words)
Summary:    output/....summary.md  (2,310 words, 13% of transcript)

Tokens:     24,118 in + 6,005 out = 30,123 total
Est. cost:  $0.0451  (gpt-5.4-mini @ $0.75/$4.5 per 1M in/out)
```

Cost is computed from token usage. Rates are resolved in this order:

1. `PRICE_INPUT` / `PRICE_OUTPUT` env vars (dollars per 1M tokens) — manual override.
2. The **live LiteLLM price registry** (community-maintained, day-0 model coverage),
   fetched once and cached for 24h in your temp dir — so rates aren't hardcoded and
   don't go stale.
3. A tiny offline `PRICING_FALLBACK` table in the script, used only if the registry
   is unreachable and no override is set.

The line shows which source was used (`via live registry` / `env override` /
`offline fallback`). If a model has no rate anywhere, it prints token counts
without a dollar figure. Word counts also appear in each file's `**Words:**`
header line, and the summary is generated to stay clearly shorter than the
transcript.

## How it works

1. **`yt-dlp`** fetches metadata and the English caption track — a human-authored
   track if one exists, otherwise the auto-generated one. No video is downloaded.
2. The VTT is parsed, inline timing tags stripped, and YouTube's **rolling-caption
   duplicates** collapsed into one clean copy of each phrase — keeping each
   phrase's start time.
3. The whole transcript is **punctuation-restored** by the model in a single pass
   — same words, made readable. Only transcripts longer than ~45k words (≈ 5–6
   hours) fall back to chunked cleanup, to stay under the model's output cap. This
   produces `transcript.md`.
4. A timestamped copy of the transcript (with `[h:mm:ss]` markers every ~30s) is
   summarized in a single pass into the sectioned summary. The model stamps each
   section heading and specific points with the nearest marker; those are then
   rewritten into **clickable links** to the video at that second. With
   gpt-5.4-mini's 400k context window, even a multi-hour talk (~35k tokens) is
   summarized whole, so no cross-section context is lost.

## Notes & troubleshooting

- **No English captions** → the script exits with a clear message. Videos without
  any English (manual or auto) captions aren't supported.
- **YouTube bot-throttling** (`Sign in to confirm you're not a bot`, HTTP 429): pass
  cookies from a logged-in browser by adding the flag to the `yt-dlp` call, or run
  `yt-dlp` once manually with `--cookies-from-browser chrome` to confirm access.
  The simplest fix is usually to retry later or from a different network.
- **Cost** scales with transcript length. A ~1-hour talk is a few cents; a 3-hour
  podcast, more. Use a cheaper `--model` for routine runs if needed.
- Long talks fit comfortably in a single summary pass (a 3-hour transcript is
  ~35k tokens), so the summary is not chunked.
