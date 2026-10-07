# Contributing

Thanks for helping! The short version:

```bash
git clone <your fork> && cd lessong
uv sync --extra dev            # Python 3.10-3.12, installs PyTorch/Demucs too
uv run pytest                  # ~45 s; no network, no API keys, no real music needed
uv run ruff check src tests
```

- The tests build a synthetic song (click track, chords, a vocal-like stem) and fake the ElevenLabs/OpenAI clients, so they run
  anywhere `ffmpeg` is installed. Please keep it that way: **never commit audio or lyrics** (the `.gitignore` already blocks
  `*.mp3`, `*.wav`, `.lessong/` and `.env`), and use made-up text in tests.
- Every tunable lives once in `src/lessong/config.py`; the CLI flags and their defaults are generated from it.
- A change to the audio logic should come with a measurement, not just a listening impression: see the checks in `tests/` and the
  `transitions.json` written next to each render.
- Open an issue before a large change; small fixes can go straight to a pull request.
