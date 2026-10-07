"""Find a seamless, voice-free instrumental loop anywhere in the track.

Candidates are every detected beat x the requested loop length. Hard filters: no vocals in the loop or the cycle after it,
and a loudness typical of the song (rules out quiet intros/noise). Ranked by how well one cycle repeats the next
(mean-centred log-mel similarity, with the true period refined on the onset envelope) and by how natural the seam is."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf

from .audio import SR, log
from .config import Settings

ASR, HOP, EHOP = 22050, 512, 64


@dataclass
class Loop:
    t: float
    length: float
    bars: int
    sim: float
    seam: float
    note: str = ""


def beat_period(bt: np.ndarray) -> float:
    """Seconds per beat. The tracker snaps beats to a ~23 ms grid, so the median interval can be 1-2% off (a 136 ms error over a
    4-bar loop). Instead, count how many beats each interval spans (1, or 2 where the tracker skipped one: a purely local
    decision, so errors cannot accumulate) and divide the total span by the total count: accurate to a fraction of a ms."""
    d = np.diff(bt)
    units = np.maximum(1.0, np.round(d / np.median(d)))
    return float((bt[-1] - bt[0]) / units.sum())


def analyse(inst_wav, vocals_wav, beats_per_bar: int) -> dict:
    import librosa  # heavy import, only needed when (re)analysing
    nov, _ = sf.read(inst_wav)
    voc, _ = sf.read(vocals_wav)
    y = librosa.resample(nov.mean(1), orig_sr=SR, target_sr=ASR)
    yv = librosa.resample(voc.mean(1), orig_sr=SR, target_sr=ASR)
    S = np.log(librosa.feature.melspectrogram(y=y, sr=ASR, n_mels=64, hop_length=HOP) + 1e-6)
    oenv = librosa.onset.onset_strength(y=y, sr=ASR, hop_length=HOP)
    env = librosa.onset.onset_strength(y=y, sr=ASR, hop_length=EHOP)
    _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=ASR, hop_length=HOP, start_bpm=110)
    bt = librosa.frames_to_time(beats, sr=ASR, hop_length=HOP)
    db = lambda a: 20 * np.log10(librosa.feature.rms(y=a, frame_length=2048, hop_length=HOP)[0] + 1e-9)
    return dict(S=S, env=env, bt=bt, vdb=db(yv), idb=db(y), fr=ASR / HOP, efps=ASR / EHOP,
                bar=beat_period(bt) * beats_per_bar, nov=nov)


def refine_period(env, t, L, efps, search=0.04):
    a, n, w = int(t * efps), int(L * efps), int(min(L, 4.0) * efps)
    best = (-9.0, 0)
    for d in range(-int(search * efps), int(search * efps) + 1):
        A, B = env[a:a + w], env[a + n + d:a + n + d + w]
        if len(B) < w or A.std() < 1e-6 or B.std() < 1e-6:
            continue
        c = float(np.corrcoef(A, B)[0, 1])
        if c > best[0]:
            best = (c, d)
    return L + best[1] / efps, best[0]


def cycle_sim(S, a, b, n) -> float:
    A, B = S[:, a:a + n], S[:, b:b + n]
    m = min(A.shape[1], B.shape[1])
    if m < 4:
        return -1.0
    A, B = A[:, :m], B[:, :m]
    return float(np.mean((A * B).sum(0) / (np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0) + 1e-9)))


def render(nov: np.ndarray, t: float, L: float, xf_s: float) -> np.ndarray:
    """Cyclic loop: its first xf seconds crossfade from the music's real continuation after the loop end (no dip at the seam)."""
    a, n, xf = int(t * SR), int(round(L * SR)), int(xf_s * SR)
    C = nov[a:a + n].copy()
    w = (np.sin(np.linspace(0, np.pi / 2, xf)) ** 2)[:, None]
    C[:xf] = w * nov[a:a + xf] + (1 - w) * nov[a + n:a + n + xf]
    return C


def seam_ratio(nov, loop, t, L) -> float:
    """Spectral jump at the loop seam / jump at the same bar position in the original music (1.0 = as natural as the song)."""
    import librosa

    def flux(y):
        S = np.log(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64, hop_length=256) + 1e-6)
        return np.abs(np.diff(S, axis=1)).mean(0)
    d = flux(np.tile(loop.mean(1), 3))
    i = int(len(loop) / 256)
    seam = np.mean([d[k * i - 3:k * i + 3].max() for k in (1, 2)])
    a = max(0, int((t - 0.2) * SR))
    ds = flux(nov[a:a + int((3 * L + 0.4) * SR)].mean(1))
    ref = np.mean([ds[int((0.2 + k * L) * SR / 256) - 3:int((0.2 + k * L) * SR / 256) + 3].max() for k in (0, 1, 2)])
    return float(seam / (ref + 1e-9))


def search(an: dict, bars: int, vfree_db: float, loud_tol: float, strict: bool = True) -> list[dict]:
    S, bt, fr, vdb, idb = an["S"], an["bt"], an["fr"], an["vdb"], an["idb"]
    free = vdb < vfree_db
    if not free.any():
        return []
    Sc = S - S[:, free[:S.shape[1]]].mean(1, keepdims=True)
    med = float(np.median(idb[free]))
    out = []
    for t in bt:
        L, pc = refine_period(an["env"], t, bars * an["bar"], an["efps"])
        a, b, c = int(t * fr), int((t + L) * fr), int((t + 2 * L) * fr)
        if a - int(.25 * fr) < 0 or c + int(.5 * fr) >= Sc.shape[1]:
            continue
        # strict: loop + the next cycle are voice-free; relaxed: only the loop itself and its crossfade continuation must be
        # (the next cycle is then only compared on the instrumental stem, which has no voice in it anyway)
        end = c if strict else b
        if vdb[a - int(.25 * fr):end + int(.5 * fr)].max() > vfree_db:
            continue
        if abs(float(idb[a:end].mean()) - med) > loud_tol:
            continue
        n = b - a
        sim = cycle_sim(Sc, a, b, n)
        e = int((t + 3 * L) * fr)
        if e < Sc.shape[1] and vdb[c:e].max() <= vfree_db:
            sim = min(sim, cycle_sim(Sc, b, c, n))    # two consecutive cycles must both match
        out.append(dict(t=float(t), length=float(L), pc=float(pc), sim=sim, score=sim + 0.3 * pc))
    return sorted(out, key=lambda d: -d["score"])


def candidates(an: dict, bars: int, s: Settings, top: int = 40, allow_vocal: bool = False) -> tuple[list[Loop], str]:
    """Ranked loops for a given length. Relaxes the voice-free threshold step by step if the song has no clean stretch."""
    from .lyrics import vocal_threshold
    base = vocal_threshold(an["vdb"])
    steps = [s.loop_vfree_db] if s.loop_vfree_db is not None else [base, base + 6, base + 12]    # relax upwards if nothing qualifies
    if allow_vocal:
        steps = [1e9]            # anywhere in the song: the instrumental stem has the voice taken out already
    for k, (vf, strict) in enumerate((vf, st) for vf in steps for st in (True, False)):
        tol = s.loop_loudness_tol * (1 if k < 4 else 2)
        cs = search(an, bars, vf, tol, strict)
        if cs:
            if allow_vocal:
                note = "no voice-free stretch long enough: cut from the instrumental stem where the song has singing (check for faint vocal residue)"
            else:
                note = ("" if vf == steps[0] else f"voice-free threshold relaxed to {vf:.0f} dB: a little vocal bleed may remain") \
                    + ("" if strict else ("; " if vf != steps[0] else "") + "only the loop itself is voice-free (the repeat check uses the instrumental stem)")
            res = []
            for d in cs[:top]:
                loop = render(an["nov"], d["t"], d["length"], s.loop_crossfade)
                sm = seam_ratio(an["nov"], loop, d["t"], d["length"])
                res.append(Loop(d["t"], d["length"], bars, d["sim"], sm, note))
            res.sort(key=lambda l: -(l.sim - 0.15 * max(0.0, l.seam - 1)))
            return res, note
    return [], "no voice-free stretch long enough for this loop length"


def pick(an: dict, s: Settings) -> Loop:
    if s.loop_start is not None:
        L, _ = refine_period(an["env"], s.loop_start, s.loop_bars * an["bar"], an["efps"])
        loop = render(an["nov"], s.loop_start, L, s.loop_crossfade)
        return Loop(s.loop_start, L, s.loop_bars, float("nan"), seam_ratio(an["nov"], loop, s.loop_start, L), "manual start")
    cs, note = candidates(an, s.loop_bars, s)
    if not cs:       # prefer the requested length taken from the instrumental stem over a shorter loop, if it repeats well
        alt, alt_note = candidates(an, s.loop_bars, s, allow_vocal=True)
        if alt and alt[0].sim >= 0.45:
            cs, note = alt, alt_note
    for b in sorted({s.loop_bars // 2, s.loop_bars // 4, 1} - {0}, reverse=True):     # e.g. 4 -> 2 -> 1 bar
        if cs or b >= s.loop_bars:
            continue
        cs, note = candidates(an, b, s)
        if cs:
            log(f"[loop] no voice-free stretch long enough for a {s.loop_bars}-bar loop in this song: using {b} bar(s) instead")
    if not cs:
        raise SystemExit(f"error: could not find a {s.loop_bars}-bar loop ({note}). Try --loop-bars 2 or 1, or set --loop-start.")
    if note:
        log(f"[loop] warning: {note}")
    return cs[0]
