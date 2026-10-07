# Changelog

## Unreleased

- Voices for languages without a named default (anything but English/French) now prefer native voices (tagged with that language)
  over voices that merely speak it, so e.g. Dutch gets a Flemish voice instead of an English-accented one.

- `--video` / `-o lesson.mp4`: a lyric-synced lesson video (title card, spoken lines with their translation appearing as it is said,
  karaoke view of the real song, intro/outro text, progress bar). Rendered from the mixer's timeline with Pillow + ffmpeg;
  `--video-size`, `--video-fps`, `--video-font`. Bundles DejaVu Sans (see `src/lessong/assets/DejaVu-LICENSE.txt`).
  `timeline.json` is written next to the other work files.

- The last excerpt now plays the song to its natural end (no fade-out); `--outro "text"` adds a voice-only sign-off after it
  (`--outro-lang`, `--outro-gap`).
- Cut points go in the middle of a real silence in the vocal track (the transcript's word end can fall inside a held note).
- Loop search: when no voice-free stretch is long enough, the requested length is cut from the instrumental stem where the song
  sings (if it repeats well) before falling back to a shorter loop.

- `--intro "text"`: a spoken intro read over the loop before the first lesson (translation voice by default). After it the music plays
  for the rest of the bar plus `--intro-bars` full bars (rising by `--intro-swell-db`, then settling), and the first lesson line lands
  on a bar line of the loop.
- More accurate beat period (loop lengths were up to 2% off), and a vocal threshold that works whatever the stem's noise floor.

## 0.1.0

First public release.

- Pipeline: Demucs vocal removal, Scribe transcription, lyric lookup (lrclib.net) or your own lyrics file, OpenAI translation and
  song-structure analysis, ElevenLabs narration, beat-aligned mix.
- Sections follow the song structure (verse, chorus, ...) and are only cut where the singer actually pauses.
- Seamless instrumental loop search (voice-free, repeats cycle to cycle, natural seam), with automatic fallback to shorter loops.
- Beat-aligned hand-overs between the looped bed and the real song, in both directions.
- Editable `plan.json`; cached stages; `--voice-gender`, `--repeats`, and ~50 mix/timing flags.
- `lessong doctor`, `lessong loops`.
