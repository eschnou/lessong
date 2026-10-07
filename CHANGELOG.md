# Changelog

## 0.1.0

First public release.

- Pipeline: Demucs vocal removal, Scribe transcription, lyric lookup (lrclib.net) or your own lyrics file, OpenAI translation and
  song-structure analysis, ElevenLabs narration, beat-aligned mix.
- Sections follow the song structure (verse, chorus, ...) and are only cut where the singer actually pauses.
- Seamless instrumental loop search (voice-free, repeats cycle to cycle, natural seam), with automatic fallback to shorter loops.
- Beat-aligned hand-overs between the looped bed and the real song, in both directions.
- Editable `plan.json`; cached stages; `--voice-gender`, `--repeats`, and ~50 mix/timing flags.
- `lessong doctor`, `lessong loops`.
