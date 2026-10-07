"""Narration, ducked music bed, song excerpts, final assembly."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from .audio import SR, db_to_gain, encode, fade, log, rms
from .config import Settings
from .elevenlabs import ElevenLabs
from .loops import Loop
from .loops import render as render_loop
from .lyrics import REAL_PAUSE
from .transition import find_entry


def trim(y: np.ndarray, top_db: float, frame: int = 512) -> np.ndarray:
    """Cut leading/trailing silence (frames quieter than top_db below the loudest frame)."""
    n = len(y) // frame * frame
    if n == 0:
        return y
    e = np.sqrt((y[:n].reshape(-1, frame) ** 2).mean(1))
    idx = np.nonzero(e > e.max() * 10 ** (-top_db / 20))[0]
    return y[idx[0] * frame:min(len(y), (idx[-1] + 1) * frame)]


class Voices:
    """TTS with an on-disk cache so tweaking the mix never costs characters twice."""
    def __init__(self, client: ElevenLabs, s: Settings, cache: Path, src_voice: str, dst_voice: str):
        self.c, self.s, self.cache, self.v = client, s, cache, {s.source_lang: src_voice, s.target_lang: dst_voice}
        self.chars = 0
        cache.mkdir(parents=True, exist_ok=True)

    def speak(self, text: str, lang: str) -> np.ndarray:
        s, voice = self.s, self.v[lang]
        h = hashlib.md5(f"{s.tts_model}|{voice}|{s.tts_speed}|{s.tts_stability}|{text}".encode()).hexdigest()[:16]
        p = self.cache / f"{h}.mp3"
        if not p.exists():
            p.write_bytes(self.c.tts(text, voice, s.tts_model, s.tts_speed, s.tts_stability))
            self.chars += len(text)
        y, sr = sf.read(p, dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y
        if sr != SR:
            raise RuntimeError(f"unexpected TTS sample rate {sr}")
        y = trim(y, s.trim_db)
        return y / rms(y) * db_to_gain(s.voice_level_db)


def smoothstep(n: int) -> np.ndarray:
    v = np.linspace(0.0, 1.0, max(2, n))
    return v * v * (3 - 2 * v)


def learn_block(lines: list[dict], voices: Voices, loop: np.ndarray, s: Settings, trans: dict | None = None,
                intro: dict | None = None, prelude: dict | None = None, events: list | None = None) -> tuple[np.ndarray, int]:
    """[lead-in] (src line, pause, translation, pause)* over a looped, ducked instrumental bed. Stereo.
    intro: the previous song excerpt hands over to this block (phase = where in the loop it enters, xfade, target_rms = the song's
           level, bar). The bed fades in at the song's loudness (overlapping the end of the song), then settles to bed level.
    trans: the bed swells to the song's loudness and fades out, on the loop's beat, so the next song excerpt can be laid on top
           at the returned sample T0. Returns (audio, T0); T0 == len(audio) without a transition.
    prelude: a spoken intro (audio, bar = samples per loop bar, bars, swell_db) placed after the lead-in. After it the music plays
           alone for the rest of the bar plus `bars` full bars (rising by swell_db in the first, settling in the last), so the first
           lesson line lands on a bar line of the loop. Timing is reported back in prelude["timing"] (seconds).
    events: if given, receives one dict per spoken clip ({kind: intro|src|dst, line, text, t0, t1}), times in seconds from the start of
           this block (used to draw the lesson video; it does not change the audio)."""
    z = lambda sec: np.zeros(int(sec * SR))
    p_in = 0
    if intro:
        Xi = int(intro["xfade"] * SR)
        st = max(1, int(round(s.settle_bars * intro["bar"])))
        p_in = int(round(intro["phase"] * SR)) % len(loop)
        loop = np.roll(loop, -p_in, axis=0)              # the loop is cyclic: start it where the song left off
        parts = [np.zeros(Xi + st + int(0.3 * SR))]      # the first voice waits until the music has settled
    else:
        parts = [z(s.lead_in)]
    gap = None
    if prelude:
        parts.append(prelude["audio"])
        spoken_end = sum(len(x) for x in parts)
        bar = prelude["bar"]
        teach_at = int(round(math.ceil(spoken_end / bar) * bar + prelude["bars"] * bar))     # a bar line of the loop
        parts.append(np.zeros(teach_at - spoken_end))
        gap = (spoken_end, teach_at)
        prelude["timing"] = dict(speech_start=(spoken_end - len(prelude["audio"])) / SR, speech_end=spoken_end / SR, teach_start=teach_at / SR)
        if events is not None:
            events.append(dict(kind="intro", text=prelude.get("text", ""), t0=prelude["timing"]["speech_start"], t1=prelude["timing"]["speech_end"]))
    pos = sum(len(x) for x in parts)                       # running position (samples) of the next clip, for the events
    for l in lines:
        a = voices.speak(l["text"].rstrip(".,;:"), s.source_lang)
        gap1 = z(s.gap_lang)
        b = voices.speak(l["translation"], s.target_lang)
        gap2 = z(s.gap_line)
        if events is not None:
            events += [dict(kind="src", line=l["i"], t0=pos / SR, t1=(pos + len(a)) / SR),
                       dict(kind="dst", line=l["i"], t0=(pos + len(a) + len(gap1)) / SR, t1=(pos + len(a) + len(gap1) + len(b)) / SR)]
        parts += [a, gap1, b, gap2]
        pos += len(a) + len(gap1) + len(b) + len(gap2)
    voice = np.concatenate(parts)
    L = len(loop)
    if trans:
        bar, X = trans["bar"], int(trans["xfade"] * SR)
        sw = max(1, int(round(s.swell_bars * bar)))
        qs = (trans["phase"] * SR - p_in) % bar          # the song enters this far into the loop (real beat position)
        need = len(voice) + s.learn_tail * SR + sw       # earliest moment the hand-over may happen
        T0 = int(round(qs + math.ceil((need - qs) / bar) * bar))   # loop phase at T0 == the song's entry beat
        S0 = T0 - sw
        n = T0 + X
    else:
        T0 = n = len(voice) + int(s.learn_tail * SR)
    bed = np.tile(loop / rms(loop), (n // L + 1, 1))[:n]
    env = np.pad(np.abs(voice), (0, n - len(voice)))
    vr = db_to_gain(s.voice_level_db)
    speaking = uniform_filter1d(maximum_filter1d(env, int(0.05 * SR)), int(0.05 * SR)) > 0.17 * vr
    act = uniform_filter1d(maximum_filter1d(speaking.astype(float), max(1, int(s.duck_hold * SR))), int(0.2 * SR))
    lg = db_to_gain(s.learn_gain_db)
    base0 = vr * db_to_gain(s.bed_db) * lg
    gain = vr * db_to_gain(s.bed_db) * 10 ** (s.duck_db * act / 20) * lg
    if gap:
        g0, g1 = gap
        n_g = g1 - g0
        up, down = min(int(prelude["bar"]), n_g // 2), min(int(prelude["bar"]), n_g // 2)
        shape = np.ones(n_g)
        shape[:up] = smoothstep(up)
        shape[n_g - down:] = 1 - smoothstep(down)
        gain[g0:g1] *= 10 ** (prelude["swell_db"] * shape / 20)               # the music rises, then settles before the first line
    if intro:
        full2 = intro["target_rms"] / rms(bed[:Xi])
        gain[:Xi] = full2 * smoothstep(Xi)                                    # fades in as the song fades out: constant level
        gain[Xi:Xi + st] = full2 * (base0 / full2) ** smoothstep(st)          # then settles under the narration
    if trans:
        base = gain[S0 - 1]
        full = trans["target_rms"] / rms(bed[T0:T0 + X])                      # loop level that equals the song's level
        gain[S0:T0] = base * (full / base) ** smoothstep(sw)                  # grows steadily (linear in dB)
        gain[T0:] = full * (1 - smoothstep(X))                                # then hands over to the song
        bed = bed * gain[:, None]
        if not intro:
            bed[:SR] *= np.linspace(0, 1, SR)[:, None]
    else:
        bed = fade(bed * gain[:, None], 0.0 if intro else 1.0, 1.0)
    bed[:len(voice)] += voice[:, None] * lg
    return bed, T0


def song_windows(plan: dict, s: Settings, duration: float) -> list[dict]:
    """Excerpt per section. It starts a little before the first sung line and ends a little after the last one, always
    cutting in a real vocal pause: a section boundary is the longest silence between the two sections, never a point in
    the middle of a sung phrase. Fades are shortened so they only ever happen after the last sung word / before the first."""
    L = plan["lines"]
    out = []
    for k, sec in enumerate(plan["sections"]):
        first, last = L[sec["lines"][0]], L[sec["lines"][-1]]
        sing0 = first.get("sing_start", first["start"])
        sing1 = last.get("sing_end", last["end"])
        a = max(0.0, sing0 - s.song_preroll)
        b = min(duration, sing1 + s.song_tail)
        if k > 0:    # boundary with the previous section = best pause after its last line
            prev_last = L[plan["sections"][k - 1]["lines"][-1]]
            a = max(a, prev_last.get("vgap_mid", (prev_last["end"] + first["start"]) / 2))
        last_section = k == len(plan["sections"]) - 1
        if last_section:
            b = duration                                   # the song plays out to its natural end (outro, held chord, vocalise...)
        if k < len(plan["sections"]) - 1:
            b = min(b, last.get("vgap_mid", (last["end"] + L[plan["sections"][k + 1]["lines"][0]]["start"]) / 2))
        b = max(b, sing1 if last.get("vgap", 1) >= REAL_PAUSE else b)
        out.append(dict(start=a, end=b,
                        fade_in=min(s.song_fade_in, max(0.05, sing0 - a)),
                        fade_out=0.05 if last_section else min(s.song_fade_out, max(0.1, b - sing1)),
                        cut_in_singing=bool(k < len(plan["sections"]) - 1 and last.get("vgap", 1) < REAL_PAUSE)))
    return out


class Track:
    """Growing stereo buffer: pieces are laid at sample offsets, so the end of one can overlap the start of the next."""
    def __init__(self):
        self.buf = np.zeros((SR * 30, 2))
        self.end = 0

    def add(self, x: np.ndarray, at: int) -> None:
        if at + len(x) > len(self.buf):
            self.buf = np.concatenate([self.buf, np.zeros((max(len(self.buf), at + len(x) - len(self.buf)), 2))])
        self.buf[at:at + len(x)] += x
        self.end = max(self.end, at + len(x))

    def audio(self) -> np.ndarray:
        return self.buf[:self.end]


def assemble(plan: dict, s: Settings, source_wav: Path, loop: Loop, nov: np.ndarray, feat: dict, voices: Voices, out: Path, tmp_wav: Path) -> tuple[float, dict]:
    """Mix everything and encode `out`. Returns (duration in seconds, timeline of what happens when, for the lesson video)."""
    src, _ = sf.read(source_wav)
    L, secs = plan["lines"], plan["sections"]
    wins = song_windows(plan, s, len(src) / SR)
    loop_audio = render_loop(nov, loop.t, loop.length, s.loop_crossfade)
    bar = len(loop_audio) / loop.bars                      # samples per bar, exactly as the loop is cut
    Xmax = s.transition_bars * bar / SR                    # longest crossfade, seconds
    gsong = db_to_gain(s.song_gain_db)
    has_nar = [any(L[i]["narrate"] for i in sec["lines"]) for sec in secs]

    def search(lo, hi):
        for xf in (Xmax, Xmax / 2, Xmax / 4, Xmax / 8):     # shorter crossfade when the pause is short
            e = find_entry(feat, loop.t, s.beats_per_bar, lo, hi, xf)
            if e:
                return dict(e, xfade=xf)
        return None

    # 1. where the song comes in (after the lesson) and where it hands back to the loop (before the next lesson)
    entries, exits = [None] * len(secs), [None] * len(secs)
    if not s.plain_entry and feat is not None:
        for k, sec in enumerate(secs):
            if not has_nar[k]:
                continue
            first, last = L[sec["lines"][0]], L[sec["lines"][-1]]
            sing0 = first.get("sing_start", first["start"])
            prev = L[secs[k - 1]["lines"][-1]] if k else None
            entries[k] = search(max((prev.get("sing_end", prev["end"]) + 0.05) if prev else 0.0, sing0 - s.entry_lookback), sing0 + 0.1)
            if k + 1 < len(secs) and has_nar[k + 1]:
                nxt = L[secs[k + 1]["lines"][0]]
                exits[k] = search(last.get("sing_end", last["end"]) + 0.02, nxt.get("sing_start", nxt["start"]) + 0.1)

    # 2. lay everything on a timeline
    tr, report, cursor, intro = Track(), [], 0, None
    timeline = dict(events=[], sections=[])
    prelude, intro_timing, last_song_end = None, None, None
    if s.intro and any(has_nar):
        prelude = dict(audio=voices.speak(s.intro, s.target_lang if s.intro_lang == "target" else s.source_lang),
                       bar=bar, bars=s.intro_bars, swell_db=s.intro_swell_db, text=s.intro)
    for k, (sec, w) in enumerate(zip(secs, wins)):
        ent, ext = entries[k], exits[k]
        nar = [L[i] for i in sec["lines"] if L[i]["narrate"]]
        s0, s1 = (ent["t"] if ent else w["start"]), (ext["t"] + ext["xfade"] if ext else w["end"])
        song = src[int(round(s0 * SR)):int(round(s1 * SR))].copy() * gsong
        warn = "  (!) boundary falls inside continuous singing; try --replan or a different --max-lines" if w["cut_in_singing"] else ""
        xe = int(ent["xfade"] * SR) if ent else 0
        trans = dict(bar=bar, xfade=ent["xfade"], target_rms=rms(song[:xe]), phase=ent["phase"]) if ent else None
        block_events: list = []
        if nar:
            block, T0 = learn_block(nar, voices, loop_audio, s, trans, intro, prelude, block_events)
            if prelude:
                intro_timing, prelude = prelude.get("timing"), None          # only the very first lesson gets the intro
                log(f"[mix] intro: spoken {intro_timing['speech_start']:.1f}-{intro_timing['speech_end']:.1f}s, then music; first lesson line at {intro_timing['teach_start']:.1f}s")
            tr.add(block, cursor)
            timeline["events"] += [dict(e, t0=e["t0"] + cursor / SR, t1=e["t1"] + cursor / SR, section=k + 1) for e in block_events]
        else:
            T0 = 0
        if ent:
            song[:xe] *= smoothstep(xe)[:, None]
            song_at = cursor + T0
        else:
            song = fade(song, w["fade_in"], 0.0)
            song_at = cursor + (len(block) + int(s.gap_to_song * SR) if nar else 0)
        if ext:
            xo = int(ext["xfade"] * SR)
            intro = dict(bar=bar, xfade=ext["xfade"], phase=ext["phase"], target_rms=rms(song[-xo:]))
            song[-xo:] *= (1 - smoothstep(xo))[:, None]
            nxt_cursor = song_at + len(song) - xo                      # the next lesson starts under the end of the song
        else:
            intro = None
            song = fade(song, 0.0, w["fade_out"] if not ext else 0.0)
            nxt_cursor = song_at + len(song) + int(s.gap_section * SR)
            last_song_end = song_at + len(song)
        tr.add(song, song_at)
        timeline["events"].append(dict(kind="song", section=k + 1, t0=song_at / SR, t1=(song_at + len(song)) / SR, src0=s0))
        timeline["sections"].append(dict(section=k + 1, label=sec.get("label"), lines=list(sec["lines"]), t0=cursor / SR, t1=(song_at + len(song)) / SR))
        log(f"[mix] section {k + 1}/{len(secs)}: {len(nar)} lines; song in "
            + (f"at {s0:.2f}s on the loop's beat {ent['q'] + 1} ({ent['xfade']:.2f}s crossfade)" if ent else f"at {s0:.1f}s (plain fade)")
            + "; out "
            + (f"at {ext['t']:.2f}s into the loop's beat {ext['q'] + 1} ({ext['xfade']:.2f}s crossfade)" if ext else f"at {s1:.1f}s (plain fade)") + warn)
        report.append(dict(section=k + 1, song_out=s1, entry=s0 if ent else None, at=(cursor + T0) / SR if ent else None,
                           **({"xfade": ent["xfade"], "q": ent["q"], "phase": ent["phase"], "onset_corr": ent["onset_corr"]} if ent else {}),
                           exit=dict(t=ext["t"], xfade=ext["xfade"], q=ext["q"], phase=ext["phase"], at=(song_at + len(song)) / SR - ext["xfade"]) if ext else None))
        cursor = nxt_cursor
    outro_timing = None
    if s.outro and last_song_end is not None:             # after the song: just the voice, no music
        v = voices.speak(s.outro, s.target_lang if s.outro_lang == "target" else s.source_lang) * db_to_gain(s.learn_gain_db)
        at = last_song_end + int(s.outro_gap * SR)
        tr.add(np.stack([v, v], 1), at)
        outro_timing = dict(song_end=last_song_end / SR, speech_start=at / SR, speech_end=(at + len(v)) / SR)
        timeline["events"].append(dict(kind="outro", text=s.outro, t0=at / SR, t1=(at + len(v)) / SR))
        log(f"[mix] outro: the song ends at {outro_timing['song_end']:.1f}s, then the voice ({outro_timing['speech_start']:.1f}-{outro_timing['speech_end']:.1f}s), no music")
    audio = tr.audio()
    sf.write(tmp_wav, audio, SR)
    if outro_timing:
        report.append({"outro": outro_timing})
    if s.intro and intro_timing:
        report.insert(0, {"intro": intro_timing})
    (tmp_wav.parent / "transitions.json").write_text(__import__("json").dumps(report, indent=1))
    encode(tmp_wav, out, s.bitrate, s.ceiling_db)
    return len(audio) / SR, timeline
