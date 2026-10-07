"""Lyrics: fetch text (lrclib.net / file / transcript) and time every line against the isolated vocals."""
from __future__ import annotations

import difflib
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    import numpy as np

from .audio import log

UA = {"User-Agent": "lessong/0.1 (language-learning tracks)"}


def norm(s: str) -> str:
    return re.sub(r"[^\w']", "", s.lower().replace("’", "'"))


def clean_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("(", "").replace(")", "")).strip()


def parse_lyrics(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        s = re.sub(r"^\s*\[\d+:\d+(?:[.:]\d+)?\]\s*", "", raw).strip()
        if not s or re.match(r"^\[[a-z]+:.*\]$", s, re.I) or re.match(r"^\[(chorus|verse|bridge|intro|outro|hook)[^\]]*\]$", s, re.I):
            continue
        out.append(s)
    return out


def fetch_candidates(title: str, artist: str | None, cache: Path | None = None) -> list[list[str]]:
    if cache and cache.exists():
        res = json.loads(cache.read_text())
        return _candidates(res)
    params = {"track_name": title}
    if artist:
        params["artist_name"] = artist
    res = None
    for i in range(4):
        try:
            r = requests.get("https://lrclib.net/api/search", params=params, headers=UA, timeout=20)
            if r.status_code == 200:
                res = r.json()
                break
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    if res is None:
        log("[lyrics] lrclib.net unreachable (will fall back to the vocal transcript; rerun with --replan to retry)")
        return []
    if cache:
        cache.write_text(json.dumps(res))
    return _candidates(res)


def _candidates(res: list[dict]) -> list[list[str]]:
    seen, cands = set(), []
    for x in res:
        if x.get("instrumental"):
            continue
        lines = parse_lyrics(x.get("syncedLyrics") or x.get("plainLyrics") or "")
        key = tuple(norm(l) for l in lines)
        if lines and key not in seen:
            seen.add(key)
            cands.append(lines)
    return cands[:8]


def align(lines: list[str], words: list[dict], min_block: int = 2, max_jump: float = 2.5) -> tuple[list[dict], int, int]:
    """Match lyric words to transcribed words (in order); each line gets start/end from its matched words.
    If a line's matched words are scattered in time (a stray match), only the largest compact cluster is used."""
    lw = [(i, norm(w)) for i, t in enumerate(lines) for w in t.split() if norm(w)]
    sw = [x for x in words if x.get("type") == "word" and norm(x["text"])]
    sm = difflib.SequenceMatcher(None, [w for _, w in lw], [norm(x["text"]) for x in sw], autojunk=False)
    hits: dict[int, list[tuple[float, float]]] = {}
    matched = 0
    for a, b, n in sm.get_matching_blocks():
        if n < min_block:
            continue
        for k in range(n):
            matched += 1
            hits.setdefault(lw[a + k][0], []).append((sw[b + k]["start"], sw[b + k]["end"]))
    out = []
    for i, t in enumerate(lines):
        h = sorted(hits.get(i, []))
        clusters, cur = [], []
        for x in h:
            if cur and x[0] - cur[-1][1] > max_jump:
                clusters.append(cur)
                cur = []
            cur.append(x)
        if cur:
            clusters.append(cur)
        best = max(clusters, key=len) if clusters else []
        out.append({"i": i, "text": clean_line(t), "start": best[0][0] if best else None, "end": max(e for _, e in best) if best else None,
                    "timed": bool(best)})
    return out, matched, len(lw)


def fill_gaps(lines: list[dict]) -> None:
    """Lines the matcher missed get a time slot between their neighbours."""
    n = len(lines)
    for i, l in enumerate(lines):
        if l["start"] is not None:
            continue
        prev = next((lines[j]["end"] for j in range(i - 1, -1, -1) if lines[j]["end"] is not None), None)
        nxt = next((lines[j]["start"] for j in range(i + 1, n) if lines[j]["start"] is not None), None)
        if prev is not None and nxt is not None and nxt - prev > 0.3:
            l["start"], l["end"] = prev + 0.05, nxt - 0.05
        elif prev is not None:
            l["start"], l["end"] = prev + 0.05, prev + 1.5
        elif nxt is not None:
            l["start"], l["end"] = max(0.0, nxt - 1.5), nxt - 0.05
    for l in lines:
        if l["start"] is not None and (l["end"] is None or l["end"] <= l["start"]):
            l["end"] = l["start"] + 0.8


def from_transcript(words: list[dict], gap: float = 0.6, max_words: int = 10) -> list[dict]:
    ws = [w for w in words if w.get("type") == "word"]
    lines, cur = [], []
    for w in ws:
        if cur and (w["start"] - cur[-1]["end"] > gap or len(cur) >= max_words or re.search(r"[.!?]$", cur[-1]["text"])):
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return [{"i": i, "text": clean_line(" ".join(w["text"] for w in l)), "start": l[0]["start"], "end": l[-1]["end"], "timed": True}
            for i, l in enumerate(lines)]


def only_sung(words: list[dict], act) -> list[dict]:
    """Drop transcript words that sit on silence in the vocal stem (the recognizer sometimes invents words over instrumental parts)."""
    keep = []
    for w in words:
        if w.get("type") != "word":
            keep.append(w)
            continue
        a, b = int(w["start"] / HOP), max(int(w["start"] / HOP) + 1, int(w["end"] / HOP))
        if act[a:b].mean() >= 0.4:
            keep.append(w)
    return keep


def build_lines(title: str | None, artist: str | None, lyrics_text: str | None, no_lookup: bool, words: list[dict], cache: Path | None = None,
                act=None) -> tuple[list[dict], str]:
    if act is not None:
        n0 = sum(w.get("type") == "word" for w in words)
        words = only_sung(words, act)
        dropped = n0 - sum(w.get("type") == "word" for w in words)
        if dropped:
            log(f"[lyrics] ignored {dropped} transcribed word(s) that fall on silence in the vocal track")
    cands: list[tuple[str, list[str]]] = []
    if lyrics_text:
        cands.append(("lyrics file", parse_lyrics(lyrics_text)))
    elif not no_lookup and title:
        for c in fetch_candidates(title, artist, cache):
            cands.append(("lrclib.net", c))
    elif not no_lookup:
        log("[lyrics] no song title known, skipping the online lookup: pass --title (and --artist) or --lyrics-file")
    best = None
    for src, lines in cands:
        al, m, tot = align(lines, words)
        score = m - 0.5 * (tot - m)
        if best is None or score > best[0]:
            best = (score, src, al, m, tot)
    if best and best[3] >= 0.4 * best[4]:
        _, src, al, m, tot = best
        fill_gaps(al)
        al = [l for l in al if l["start"] is not None]
        for k, l in enumerate(al):
            l["i"] = k
        log(f"[lyrics] {src}: {m}/{tot} words aligned, {sum(l['timed'] for l in al)}/{len(al)} lines timed")
        return al, src
    log("[lyrics] no usable lyrics found" + (" (poor match)" if best else "") + "; falling back to the vocal transcript. "
        "Edit .lessong/<song>/lyrics.json or pass --lyrics-file for better results.")
    return from_transcript(words), "transcript"


# ---------------------------------------------------------------------------------------------------------------
# Real singing activity (from the isolated vocal stem): where each line is actually sung, held notes included,
# and where the real pauses are. Word timestamps alone cut sung phrases short and place cuts inside continuous singing.
VOCAL_HOLD = 0.3     # a pause shorter than this does not end a phrase
HOP = 0.02


def vocal_threshold(level_db) -> float:
    """dB level below which the vocal stem counts as 'no singing'. Stems can contain digital silence (-180 dB) or a steady bleed
    floor, so the threshold is anchored on the singing level (18 dB under it) but never closer to the floor than 40% of the way up."""
    import numpy as np
    x = np.clip(np.asarray(level_db, dtype=float), -80.0, None)
    sing = float(np.median(x[x > np.percentile(x, 90) - 15]))      # the frames that are clearly singing
    floor = float(np.percentile(x, 10))                            # the quiet frames (silence or bleed)
    return max(sing - 18.0, floor + 0.4 * (sing - floor))


def vocal_activity(vocals_wav) -> np.ndarray:
    import numpy as np
    import soundfile as sf
    from scipy.ndimage import binary_closing
    v, sr = sf.read(vocals_wav)
    v = v.mean(1) if v.ndim > 1 else v
    h = int(HOP * sr)
    n = len(v) // h * h
    e = 20 * np.log10(np.sqrt((v[:n].reshape(-1, h) ** 2).mean(1)) + 1e-9)
    act = e > vocal_threshold(e)
    return binary_closing(act, structure=np.ones(6, bool))  # bridge 0.1 s dropouts inside a note


def _runs(inactive, i0: int, i1: int):
    """(start, end) frame ranges of silence inside [i0, i1)."""
    out, s = [], None
    for i in range(max(0, i0), min(len(inactive), i1)):
        if inactive[i] and s is None:
            s = i
        elif not inactive[i] and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, min(len(inactive), i1)))
    return out


REAL_PAUSE = 0.2   # a boundary needs at least this much pause (stem silence OR gap between the last and next word) to be cut cleanly


def annotate_vocals(lines: list[dict], act) -> None:
    """Adds to each line: sing_start / sing_end (real phrase edges), vgap (usable pause after the line, s), vgap_mid (the best
    place to cut after this line). The pause is the longer of (a) real silence in the isolated vocals and (b) the gap between the
    last word of the line and the next word in the transcript: dense songs have reverb and backing vocals that keep the energy
    detector 'busy', but the recognizer still sees where a phrase ends."""
    quiet = ~act
    f = lambda t: int(t / HOP)
    n = len(lines)
    for i, l in enumerate(lines):
        prev, nxt = (lines[i - 1] if i else None), (lines[i + 1] if i + 1 < n else None)
        hi = (nxt["start"] + 0.1) if nxt else l["end"] + 30
        runs = _runs(quiet, f(l["end"] - 0.2), f(hi))
        long_runs = [r for r in runs if (r[1] - r[0]) * HOP >= VOCAL_HOLD]
        word_gap = max(0.0, (nxt["start"] - l["end"]) - 0.1) if nxt else 30.0
        best = max(runs, key=lambda r: r[1] - r[0]) if runs else None
        run_len = (best[1] - best[0]) * HOP if best else 0.0
        if long_runs:        # clear silence after the line: the phrase ends where it starts
            sing_end = min(long_runs[0][0] * HOP, nxt["start"] if nxt else 1e9)
        else:                # no clear silence: the phrase ends just after the last word (plus a little decay)
            sing_end = l["end"] + min(0.15, word_gap / 2 + 0.02)
        l["sing_end"] = round(min(max(l["end"], sing_end), nxt["start"] if nxt else 1e9), 2)
        if best and run_len >= 0.15:
            # a real silence exists: cut in the middle of it (the transcript's word end can fall inside a held note)
            l["vgap"], l["vgap_mid"] = round(max(run_len, word_gap), 2), round((best[0] + best[1]) / 2 * HOP, 2)
        else:
            l["vgap"] = round(word_gap, 2)
            l["vgap_mid"] = round((l["end"] + nxt["start"]) / 2 if nxt else l["end"] + 1.0, 2)
        lo = (prev["end"] - 0.2) if prev else l["start"] - 30
        back = [r for r in _runs(quiet, f(max(0.0, lo)), f(l["start"] + 0.1)) if (r[1] - r[0]) * HOP >= VOCAL_HOLD]
        l["sing_start"] = min(l["start"], max(back[-1][1] * HOP if back else l["start"] - 0.15, prev["end"] if prev else 0.0))
        l["sing_end"], l["sing_start"] = round(l["sing_end"], 2), round(l["sing_start"], 2)
