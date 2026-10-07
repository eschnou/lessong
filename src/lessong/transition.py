"""Beat-aligned hand-over from the looped bed to the real song.

The loop is cut from the song's own instrumental, so tempo and sound already match. What has to match is the *position in the
bar*: we start the song excerpt on a beat that sits an exact number of bars away from the loop's start (so the drums land on
the loop's drums), pick, among those, the bar whose music resembles the loop best, and nudge it by a few milliseconds so the
onsets line up. numpy only: the features come from the loop analysis."""
from __future__ import annotations

import numpy as np


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or a.std() < 1e-9 or b.std() < 1e-9:
        return -1.0
    return float(np.corrcoef(a, b)[0, 1])


def find_entry(feat: dict, loop_t: float, beats_per_bar: int, lo: float, hi: float, X: float, max_lag: float = 0.05) -> dict | None:
    """Best start time t for the song excerpt, in [lo, hi - X], on any beat. The loop is cyclic, so it can be started at any
    rotation: a beat q beats after the bar position of `loop_t` is matched by playing the loop from phase q. We score every
    beat by how well the next X seconds resemble the loop's X seconds at that phase (timbre/harmony and drum onsets), refine
    by a few ms, and prefer entries that land on the loop's own bar line. Returns {t, q, score, onset_corr, sim, lag} or None."""
    bt, env, S, fr, efps = feat["bt"], feat["env"], feat["S"], float(feat["fr"]), float(feat["efps"])
    if len(bt) < 2 * beats_per_bar:
        return None
    j0 = int(np.argmin(np.abs(bt - loop_t)))
    W = max(X, 1.5)                                       # estimate timing/similarity on a longer stretch than the crossfade itself
    ne, ns = int(W * efps), int(W * fr)
    Sc = S - S.mean(1, keepdims=True)
    refs = {}
    for q in range(beats_per_bar):                       # the loop's reference segment at phase q (q beats after its start)
        if j0 + q >= len(bt):
            continue
        r_env = env[int(bt[j0 + q] * efps):int(bt[j0 + q] * efps) + ne]
        r_S = Sc[:, int(bt[j0 + q] * fr):int(bt[j0 + q] * fr) + ns]
        if len(r_env) == ne and r_S.shape[1] == ns:
            refs[q] = (r_env, r_S)
    best = None
    for j, t in enumerate(bt):
        q = (j - j0) % beats_per_bar
        if q not in refs or t < lo or t + X > hi or int((t + W) * fr) + 1 >= S.shape[1]:
            continue
        ref_env, ref_S = refs[q]
        a = int(t * efps)
        cs = [(_corr(ref_env, env[a + d:a + d + ne]), d) for d in range(-int(max_lag * efps), int(max_lag * efps) + 1) if a + d >= 0]
        c, d = max(cs, key=lambda x: x[0])
        t2 = t + d / efps
        seg = Sc[:, int(t2 * fr):int(t2 * fr) + ns]
        if seg.shape[1] < ns:
            continue
        sim = float(np.mean((ref_S * seg).sum(0) / (np.linalg.norm(ref_S, axis=0) * np.linalg.norm(seg, axis=0) + 1e-9)))
        score = sim + 0.5 * c + (0.05 if q == 0 else 0.0) - 0.01 * max(0.0, hi - X - t2)
        if best is None or score > best["score"]:
            best = dict(t=float(t2), q=int(q), phase=float(bt[j0 + q] - loop_t), score=float(score), onset_corr=float(c), sim=sim, lag=float(d / efps))
    return best
