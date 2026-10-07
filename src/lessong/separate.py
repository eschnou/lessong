"""Vocal removal with Demucs (htdemucs, 2 stems)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .audio import log


def auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def separate(source_wav: Path, stems_dir: Path, device: str | None, force: bool = False) -> tuple[Path, Path]:
    vocals, inst = stems_dir / "vocals.wav", stems_dir / "no_vocals.wav"
    if vocals.exists() and inst.exists() and not force:
        log("[separate] cached")
        return vocals, inst
    dev = device or auto_device()
    log(f"[separate] Demucs on {dev} (about 1 min for a 4 min song on Apple silicon)...")
    tmp = stems_dir.parent / "_demucs"
    shutil.rmtree(tmp, ignore_errors=True)
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-d", dev, "-o", str(tmp), str(source_wav)]
    r = subprocess.run(cmd)
    if r.returncode and dev != "cpu":
        log(f"[separate] failed on {dev}, retrying on cpu")
        r = subprocess.run(cmd[:-4] + ["cpu", "-o", str(tmp), str(source_wav)])
    if r.returncode:
        raise RuntimeError("Demucs failed")
    out = tmp / "htdemucs" / source_wav.stem
    stems_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(out / "vocals.wav", vocals)
    shutil.move(out / "no_vocals.wav", inst)
    shutil.rmtree(tmp, ignore_errors=True)
    return vocals, inst
