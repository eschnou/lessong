# lessong

[![CI](https://github.com/eschnou/lessong/actions/workflows/ci.yml/badge.svg)](https://github.com/eschnou/lessong/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Learn a language from the songs you love.**

`lessong` turns any song into a listening lesson. For each part of the song you first hear every lyric line spoken in the
song's language, then its translation, over a looping instrumental of the song itself. Then the music swells and the real song
section plays (the last section plays on to the song's natural end), so you hear exactly what you were just taught. Verse, chorus, verse, chorus, and so on.

```
 ┌─ lesson ─────────────────────────────────────┐┌─ the real song ──────────┐
 │ line 1 (EN) · line 1 (FR) · line 2 (EN) · …  ││ verse 1, as recorded     │  ···
 │ over a seamless instrumental loop, ducked    ││ (the music swells in on  │
 │ under the voice                              ││ the loop's own beat)     │
 └──────────────────────────────────────────────┘└──────────────────────────┘
```

It is inspired by *Plan Langue*, a segment on the Belgian radio station Classic 21 that taught English through pop songs this
way. `lessong` automates the whole production: vocal removal, lyric timing, translation, narration and a beat-aligned mix.

## Listen to an example

https://github.com/user-attachments/assets/ec4c8ee7-a4ce-4cd2-9680-5228ab0c3877

A French song, taught in English: a short spoken intro, then each part of the song line by line (French, then English) over a loop of
the song itself, then the real song section, and a sign-off. The song was made with Suno; see [the credits](examples/CREDITS.md).
Prefer the files? [Demo lesson (MP4)](https://github.com/eschnou/lessong/raw/main/examples/pas_tant_de_temps_lesson.mp4) · [original song (MP3)](https://github.com/eschnou/lessong/raw/main/examples/pas_tant_de_temps.mp3).

It was made with one command:

```bash
lessong build examples/pas_tant_de_temps.mp3 --title "Pas tant de temps" --from fr --to en --no-lookup \
  --intro "Here is a short demo on learning french with lessong" \
  --outro "Make your own song lessons with lessong!" -o lesson.mp3
```

## The video

`--video` makes an MP4 next to the audio (or `-o lesson.mp4` for just the video): a title card with the next section, the spoken
intro, then for each lesson line the song-language text, with its translation appearing as the voice says it; while the real song
plays, the section's lines are shown with the line being sung highlighted and its translation underneath; and the sign-off.
A progress bar and the section name stay on screen. Because the picture only changes when a line changes, a five-minute video
renders in well under a minute and is a few MB. Every moment comes from the same timeline as the audio, so the two cannot drift.
Latin, Cyrillic and Greek text works out of the box; for other scripts pass `--video-font /path/to/font.ttf`.

## What you need

- Python 3.10 – 3.12 (macOS on Apple silicon or Linux; Windows is untested)
- [`ffmpeg`](https://ffmpeg.org) on your `PATH`
- An [ElevenLabs](https://elevenlabs.io) API key (speech-to-text *and* text-to-speech permissions)
- An [OpenAI](https://platform.openai.com) API key (translation and song-structure analysis)
- A song file you are allowed to use (mp3, mp4, wav, …). `lessong` ships no music and no lyrics.

## Install

```bash
git clone https://github.com/eschnou/lessong && cd lessong
uv sync                                  # or: pip install -e .
cp .env.example .env                     # then paste your two API keys into .env
uv run lessong doctor                    # checks ffmpeg, the keys and the compute device
```

To use it as a global command: `uv tool install .` (or `pipx install .`). The first install downloads PyTorch for Demucs
(a couple of GB); the first run downloads the Demucs model.

## Make your first lesson

```bash
lessong build song.mp3 --title "Song Title" --artist "Artist" --to fr -o lesson.mp3
```

That runs the whole pipeline (about 4–5 minutes for a 4-minute song on an Apple-silicon laptop, most of it Demucs) and writes
`lesson.mp3`. Useful variations:

```bash
lessong build song.mp3 --title "…" --artist "…" --to es --voice-gender female -o lesson.mp3
lessong build song.mp3 --lyrics-file my_lyrics.txt -o lesson.mp3     # skip the online lookup, use your own text
lessong build song.mp3 --repeats skip -o short.mp3                   # don't repeat a chorus you already taught
lessong build song.mp3 --video -o lesson.mp3                         # also writes lesson.mp4: the lines on screen, in sync with the audio
lessong render .lessong/song --intro "Welcome to my show" -o lesson.mp3  # add a spoken intro (2 bars of music, then the first lesson)
```

### Edit, then re-render (the fast loop)

Every stage caches its result in `.lessong/<song>/`, and the plan is a plain JSON file you can edit.

```bash
lessong plan song.mp3 --title "…" --artist "…"     # stops after writing plan.json; shows sections and the TTS cost
$EDITOR .lessong/song/plan.json                    # fix a translation, move a line to another section, mute a line…
lessong render .lessong/song -o lesson.mp3         # ~15 s once the narration is cached; every mix flag is available
```

`render` re-uses the narration it already paid for, so you can tweak levels, timing and transitions as often as you like.
`lessong loops song.mp3` lists and exports candidate instrumental loops so you can audition them.

## How it works

| step | what happens | tool |
|---|---|---|
| separate | the song is split into vocals and instrumental | [Demucs](https://github.com/adefossez/demucs) |
| transcribe | word timestamps of the isolated vocals | ElevenLabs Scribe |
| lyrics | text from [lrclib.net](https://lrclib.net) or `--lyrics-file`, matched to the transcript; words the recognizer "hears" over silence are ignored | `difflib` |
| plan | translation, then the song is cut into sections (verse, chorus, …) | OpenAI |
| loop | find a voice-free stretch that repeats exactly and loops without a seam | librosa |
| render | narration, ducked bed, beat-aligned hand-overs, song excerpts, limiter | ElevenLabs, numpy, ffmpeg |
| video (optional) | the lines being said and sung, on screen, in sync | Pillow, ffmpeg |

A few details that make it sound right, because they are the hard parts:

- **Sections follow the song's structure.** An LLM reads the lyrics and the real pauses and returns verse / chorus / bridge …
  Every section's lesson teaches *all* of its lines and its excerpt plays *exactly* those lines.
- **Cuts happen where the singer breathes.** Pauses are measured on the isolated vocal track (and on gaps between transcribed
  words), so an excerpt never stops mid-word and the fade-out never covers the last sung line.
- **The loop is found by analysis, not guessed.** Candidates are scored on being voice-free, a typical loudness, how exactly one
  cycle repeats the next, and how natural the seam is. The loop is cyclic, so its end continues into its start.
- **The music meets the song on the beat.** After the last voice the loop swells to the song's loudness, and the song enters on a
  beat whose position in the bar matches the loop's. Drum onsets line up to within a few milliseconds, and the same trick runs in
  reverse when the song hands back to the next lesson. Each render writes `transitions.json` so you can check the numbers.

## Options

`lessong build --help` lists everything (all defaults live in [`src/lessong/config.py`](src/lessong/config.py)). The ones you will
reach for:

| flag | what it does |
|---|---|
| `--from`, `--to` | song language and translation language (ISO codes, default `en` → `fr`) |
| `--voice-gender male\|female` | narrator voices (defaults exist for English and French; other languages use your voice library) |
| `--src-voice`, `--dst-voice` | a specific ElevenLabs voice (name or id) |
| `--sections llm\|gap` | cut by song structure (default) or by pauses and size only |
| `--max-lines`, `--min-lines` | section size |
| `--repeats teach\|skip` | teach a returning chorus again (default) or leave repeated sections out |
| `--loop-bars`, `--loop-start` | loop length (default 4; if the song has no voice-free stretch that long, it is cut from the instrumental stem where the song sings, or falls back to a shorter loop) or force its start time |
| `--bed-db`, `--duck-db` | how far the music sits under the voice, and how much extra it dips while someone speaks |
| `--swell-bars`, `--transition-bars`, `--settle-bars` | length of the swell, the crossfade and the settle after the song |
| `--plain-entry` | turn the beat-aligned transitions off |
| `--video`, `-o lesson.mp4` | also write a video next to the audio, or just the video (the file extension decides) |
| `--video-size`, `--video-font` | video size (default 1280x720) and a TrueType font for scripts the bundled DejaVu Sans lacks (e.g. Japanese) |
| `--intro "text"` | a spoken intro (e.g. your show's name) read over the loop before the first lesson, in the translation voice (`--intro-lang source` for the song's voice) |
| `--outro "text"` | a sign-off read after the song has played to its very end: the voice alone, no music (`--outro-lang`, `--outro-gap` for the silence before it) |
| `--intro-bars`, `--intro-swell-db` | music left after the intro before the first lesson line (default 2 bars, rising 10 dB then settling) |
| `--lead-in`, `--gap-lang`, `--gap-line` | timing of the narration |
| `--tts-model`, `--tts-speed` | ElevenLabs model and speaking speed |
| `--llm-model` | OpenAI model for translation and structure |

`--replan` redoes the lyrics and sections but keeps your edited translations (`--retranslate` redoes those too); `--force`
redoes everything including Demucs.

## What it costs

A typical song is 2,000–4,000 characters of ElevenLabs speech (each distinct line is synthesized once and cached), a couple of
Scribe minutes, and two OpenAI calls. `lessong plan` prints the character count before you spend anything on narration.

## Limitations

- Assumes steady 4/4 music with a detectable beat (`--beats-per-bar` for others). Waltzes, free tempo and heavy rubato will not
  get clean loops or aligned hand-overs; use `--plain-entry`.
- A loop needs a voice-free stretch in the song. If there is none it falls back to shorter loops and says so; check the log.
- The text you are taught comes from a community lyric database, which can differ from the recording in a few words. A speech
  recognizer is not a reliable referee for sung words either, so give `--lyrics-file` for songs you care about, or edit
  `plan.json`.
- Quality depends on Demucs separation, which is excellent but not perfect on dense mixes.

## A note on rights

You are responsible for having the right to use the audio you feed in; `lessong` bundles no music or lyrics, and fetches lyric text
from lrclib.net only for your own use. Generated lessons contain the original recording and are meant for personal study. Check
the terms of the ElevenLabs and OpenAI services you use.

## Development

```bash
uv sync --extra dev
uv run pytest            # ~45 s, fully offline: a synthetic song and fake API clients
uv run ruff check src tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md). The pipeline is split by stage (`separate`, `lyrics`, `plan`, `loops`, `transition`, `mix`)
with `pipeline.py` as the orchestrator and `config.py` as the single source of truth for every parameter.

## License

[MIT](LICENSE). Thanks to the authors of Demucs, librosa and lrclib, and to Classic 21's *Plan Langue* for the idea.
