"""All tunable parameters live here, once. CLI flags and defaults are generated from this dataclass."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields


def opt(default, help, group, stage="render", type=None, flags=None, choices=None):
    return field(default=default, metadata=dict(help=help, group=group, stage=stage, type=type, flags=flags, choices=choices))


@dataclass
class Settings:
    # ---- languages -------------------------------------------------------
    source_lang: str = opt("en", "Language of the song (ISO 639-1 code).", "Languages", "both", str, ["--from"])
    target_lang: str = opt("fr", "Language to translate into.", "Languages", "both", str, ["--to"])

    # ---- lyrics & translation (plan stage) ------------------------------
    lyrics_file: str = opt(None, "Plain-text lyrics file (one line per lyric line) instead of looking them up on lrclib.net.", "Lyrics & translation", "plan", str)
    no_lookup: bool = opt(False, "Don't look lyrics up online; use the transcript of the isolated vocals.", "Lyrics & translation", "plan")
    llm_model: str = opt("gpt-6.1-sol", "OpenAI model used for translation and song-structure analysis.", "Lyrics & translation", "plan", str)
    device: str = opt(None, "Demucs device: mps, cuda or cpu (default: auto).", "Lyrics & translation", "plan", str)

    # ---- section splitting (plan stage) ---------------------------------
    section_mode: str = opt("llm", "How to cut sections: 'llm' = by song structure (verse, chorus, verse...) using the LLM, "
                            "'gap' = by pauses and size only.", "Sections", "plan", str, ["--sections"], ["llm", "gap"])
    section_gap: float = opt(6.0, "[gap mode] Seconds of silence between lyric lines that always starts a new section.", "Sections", "plan")
    min_lines: int = opt(4, "[gap mode] Minimum narrated lines per section (shorter ones are merged).", "Sections", "plan")
    max_lines: int = opt(8, "Maximum lines per section (longer ones are split at the biggest real pause, in both modes).", "Sections", "plan")
    max_song_seconds: float = opt(35.0, "[gap mode] Split a section when its song excerpt would play longer than this (seconds).", "Sections", "plan")
    repeats: str = opt("teach", "Sections that repeat earlier ones (a returning chorus): 'teach' = teach and play them again like any other; "
                       "'skip' = leave them out of the track entirely.", "Sections", "plan", str, None, ["teach", "skip"])

    # ---- the real-song excerpt after each learning block ----------------
    song_preroll: float = opt(2.0, "Seconds of music played before the first lyric of a section.", "Song excerpt")
    song_tail: float = opt(1.5, "Seconds of music played after the last lyric of a section.", "Song excerpt")
    song_fade_in: float = opt(0.3, "Fade-in of the song excerpt (seconds).", "Song excerpt")
    song_fade_out: float = opt(1.5, "Fade-out of the song excerpt (seconds).", "Song excerpt")
    song_gain_db: float = opt(-2.0, "Gain applied to the song excerpt (dB).", "Song excerpt")

    # ---- spoken intro and outro ------------------------------------------------
    intro: str = opt(None, "Text read at the very start, over the loop, before the first lesson (e.g. the name of your show).", "Intro & outro", "render", str)
    intro_lang: str = opt("target", "Which voice reads the intro: the translation language ('target') or the song's language ('source').", "Intro & outro", "render", str, None, ["target", "source"])
    intro_bars: int = opt(2, "Bars of music left after the intro, before the first lesson line (the loop swells and settles over them).", "Intro & outro")
    intro_swell_db: float = opt(10.0, "How far the music rises above its normal bed level during those bars (dB).", "Intro & outro")
    outro: str = opt(None, "Text read after the song has played to its end: just the voice, no music (e.g. a sign-off).", "Intro & outro", "render", str)
    outro_lang: str = opt("target", "Which voice reads the outro: the translation language ('target') or the song's language ('source').", "Intro & outro", "render", str, None, ["target", "source"])
    outro_gap: float = opt(1.0, "Silence between the end of the song and the outro (seconds).", "Intro & outro")

    # ---- bed -> song hand-over ---------------------------------------------
    swell_bars: float = opt(1.0, "After the last voice, the bed swells up to the song's loudness over this many bars.", "Transition")
    transition_bars: float = opt(1.0, "Length of the beat-aligned crossfade from the bed into the song, in bars.", "Transition")
    entry_lookback: float = opt(8.0, "How far before the first sung word (seconds) the song may start, to find a beat-aligned entry.", "Transition")
    settle_bars: float = opt(1.0, "After the song hands over to the loop (at the song's loudness), the music settles down to bed level over this many bars before the first voice.", "Transition")
    plain_entry: bool = opt(False, "Disable the beat-aligned transitions (both ways): plain fades with a short silence in between.", "Transition")

    # ---- instrumental loop ------------------------------------------------
    loop_bars: int = opt(4, "Loop length in bars.", "Loop")
    loop_start: float = opt(None, "Force the loop to start at this time (seconds) instead of searching.", "Loop", "render", float)
    loop_crossfade: float = opt(0.4, "Seam crossfade length (seconds).", "Loop")
    loop_vfree_db: float = opt(None, "Vocal-stem level (dB) below which a stretch counts as voice-free (default: auto).", "Loop", "render", float)
    loop_loudness_tol: float = opt(4.0, "Loop must be within this many dB of the song's typical instrumental loudness.", "Loop")
    beats_per_bar: int = opt(4, "Beats per bar.", "Loop")

    # ---- mix ---------------------------------------------------------------
    bed_db: float = opt(-17.0, "Level of the music bed under the voices, relative to the voices (dB).", "Mix")
    duck_db: float = opt(-7.0, "Extra attenuation of the bed while someone speaks (dB).", "Mix")
    duck_hold: float = opt(0.25, "How long the bed stays ducked after speech stops (seconds).", "Mix")
    voice_level_db: float = opt(-21.0, "RMS level each narrated line is normalised to (dBFS).", "Mix")
    learn_gain_db: float = opt(4.9, "Gain of the whole learning block, to match the song's loudness (dB).", "Mix")
    ceiling_db: float = opt(-1.0, "Peak ceiling of the final limiter (dBFS).", "Mix")
    bitrate: str = opt("192k", "Output bitrate (mp3/m4a).", "Mix", "render", str)

    # ---- narration timing --------------------------------------------------
    lead_in: float = opt(2.0, "Seconds of bed alone before the first narrated line.", "Narration timing")
    gap_lang: float = opt(0.45, "Pause between the song-language line and its translation (seconds).", "Narration timing")
    gap_line: float = opt(0.9, "Pause after a translation, before the next line (seconds).", "Narration timing")
    learn_tail: float = opt(0.5, "Seconds of bed alone after the last narrated line, before it starts to swell into the song.", "Narration timing")
    gap_to_song: float = opt(0.4, "Silence between the learning block and the song excerpt (seconds).", "Narration timing")
    gap_section: float = opt(0.6, "Silence between a song excerpt and the next learning block (seconds).", "Narration timing")
    trim_db: float = opt(35.0, "Trim leading/trailing silence quieter than this below peak (dB).", "Narration timing")

    # ---- text-to-speech ----------------------------------------------------
    tts_model: str = opt("eleven_multilingual_v2", "ElevenLabs TTS model.", "Text-to-speech", "render", str)
    voice_gender: str = opt("male", "Gender of the narrator voices chosen automatically (both languages).", "Text-to-speech", "render", str, None, ["male", "female"])
    src_voice: str = opt(None, "Voice for the song language: ElevenLabs voice id or a name (default: auto, by --voice-gender).", "Text-to-speech", "render", str)
    dst_voice: str = opt(None, "Voice for the translation: ElevenLabs voice id or a name (default: auto, by --voice-gender).", "Text-to-speech", "render", str)
    tts_speed: float = opt(None, "Speaking speed, 0.7-1.2 (default: voice default).", "Text-to-speech", "render", float)
    tts_stability: float = opt(None, "Voice stability, 0-1 (default: voice default).", "Text-to-speech", "render", float)


def add_arguments(parser: argparse.ArgumentParser, stages: set[str]) -> None:
    groups: dict[str, argparse._ArgumentGroup] = {}
    for f in fields(Settings):
        m = f.metadata
        if m["stage"] not in stages:
            continue
        g = groups.setdefault(m["group"], parser.add_argument_group(m["group"]))
        flags = m["flags"] or ["--" + f.name.replace("_", "-")]
        help_ = m["help"].replace("%", "%%")
        if isinstance(f.default, bool):
            g.add_argument(*flags, dest=f.name, action="store_true", default=f.default, help=help_)
        else:
            typ = m["type"] or type(f.default)
            g.add_argument(*flags, dest=f.name, type=typ, default=f.default, choices=m["choices"],
                           metavar=None if m["choices"] else typ.__name__.upper(),
                           help=f"{help_} [default: {f.default if f.default is not None else 'auto'}]")


def from_args(ns: argparse.Namespace) -> Settings:
    return Settings(**{f.name: getattr(ns, f.name) for f in fields(Settings) if hasattr(ns, f.name)})
