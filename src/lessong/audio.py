"""ffmpeg helpers and small signal utilities."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 44100


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def need_tools() -> None:
    for t in ("ffmpeg", "ffprobe"):
        if not shutil.which(t):
            raise SystemExit(f"error: '{t}' not found on PATH (install ffmpeg, e.g. `brew install ffmpeg`)")


def ffmpeg(*args: str) -> None:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()}")


def decode(src: Path, dst: Path) -> None:
    """Any audio/video file -> 44.1 kHz stereo wav."""
    ffmpeg("-i", str(src), "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst))


def to_mono_mp3(src: Path, dst: Path, sr: int = 22050) -> None:
    """Small mono file for speech-to-text upload."""
    ffmpeg("-i", str(src), "-ac", "1", "-ar", str(sr), "-b:a", "64k", str(dst))


def tags(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=title,artist:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout).get("format", {})
    except json.JSONDecodeError:
        return {}
    t = {k.lower(): v for k, v in d.get("tags", {}).items()}
    t["duration"] = float(d.get("duration", 0) or 0)
    return t


def encode(wav: Path, out: Path, bitrate: str, ceiling_db: float) -> None:
    """Final encode with a peak limiter."""
    lossy = out.suffix.lower() in (".mp3", ".m4a", ".aac", ".ogg", ".opus")
    limit = min(1.0, max(0.0625, 10 ** ((ceiling_db - (1.0 if lossy else 0.0)) / 20)))   # lossy codecs overshoot ~1 dB
    args = ["-i", str(wav), "-af", f"alimiter=limit={limit:.4f}:level=false"]
    if out.suffix.lower() in (".mp3", ".m4a", ".aac", ".ogg", ".opus"):
        args += ["-b:a", bitrate]
    ffmpeg(*args, str(out))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def db_to_gain(db: float) -> float:
    return 10 ** (db / 20)


def fade(x: np.ndarray, fade_in: float = 0.0, fade_out: float = 0.0) -> np.ndarray:
    """Linear fades on mono or (n, ch) arrays (in place)."""
    def ramp(n, up):
        r = np.linspace(0, 1, n) if up else np.linspace(1, 0, n)
        return r[:, None] if x.ndim == 2 else r
    n_in, n_out = min(len(x), int(fade_in * SR)), min(len(x), int(fade_out * SR))
    if n_in > 1:
        x[:n_in] *= ramp(n_in, True)
    if n_out > 1:
        x[-n_out:] *= ramp(n_out, False)
    return x
