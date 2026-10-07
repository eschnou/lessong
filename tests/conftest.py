"""Shared fixtures: a synthetic 'song' (so nothing here depends on real music, the network or Demucs) and fake API clients."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 44100
BPM = 120
BEAT = 60 / BPM          # 0.5 s
BAR = 4 * BEAT           # 2 s
DURATION = 64.0

# (start, end) of the sung lines in the synthetic song. The first 24 s are an instrumental intro (12 bars, voice-free).
LINE_TIMES = [(24.0, 27.5), (28.0, 31.5), (32.0, 35.5), (36.0, 39.5), (44.0, 47.5), (48.0, 51.5), (52.0, 55.5), (56.0, 59.5)]
LYRICS = ["alpha beta gamma delta epsilon", "zeta eta theta iota kappa", "lambda mu nu xi omicron", "pi rho sigma tau upsilon",
          "phi chi psi omega alpha", "beta gamma delta epsilon zeta", "eta theta iota kappa lambda", "mu nu xi omicron pi"]

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def _instrumental(rng: np.random.Generator) -> np.ndarray:
    n = int(DURATION * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    chords = [(220.0, 277.2, 329.6), (246.9, 311.1, 370.0), (196.0, 246.9, 293.7), (261.6, 329.6, 392.0)]   # one chord per bar, 4-bar cycle
    for bar in range(int(DURATION / BAR)):
        a, b = int(bar * BAR * SR), int(min(n, (bar + 1) * BAR * SR))
        for f in chords[bar % 4]:
            y[a:b] += 0.06 * np.sin(2 * np.pi * f * t[a:b])
    for beat in range(int(DURATION / BEAT)):
        a = int(beat * BEAT * SR)
        m = min(n - a, int(0.12 * SR))
        env = np.exp(-np.arange(m) / (0.03 * SR))
        k = min(n - a, int(0.015 * SR))
        y[a:a + k] += 0.5 * rng.standard_normal(k) * np.exp(-np.arange(k) / (0.004 * SR))    # a tick on every beat (like a hi-hat)
        if beat % 2 == 0:       # kick on beats 1 and 3
            y[a:a + m] += 0.7 * np.sin(2 * np.pi * 60 * np.arange(m) / SR) * env
        else:                   # snare-ish noise on beats 2 and 4
            y[a:a + m] += 0.35 * rng.standard_normal(m) * env
    return y


def _vocals(rng: np.random.Generator) -> np.ndarray:
    n = int(DURATION * SR)
    t = np.arange(n) / SR
    v = 10 ** (-60 / 20) * rng.standard_normal(n)                       # a quiet noise floor, like a real stem
    for a, b in LINE_TIMES:
        i, j = int(a * SR), int(b * SR)
        env = np.minimum(1, np.minimum(np.arange(j - i), np.arange(j - i)[::-1]) / (0.05 * SR))
        v[i:j] += 0.25 * env * sum(np.sin(2 * np.pi * 300 * k * t[i:j]) / k for k in (1, 2, 3))
    return v


@pytest.fixture(scope="session")
def synth(tmp_path_factory) -> dict:
    """A 64 s, 120 BPM synthetic song written as source.wav + stems/{vocals,no_vocals}.wav, plus transcript words and lyric lines."""
    d = tmp_path_factory.mktemp("synth")
    rng = np.random.default_rng(7)
    inst, voc = _instrumental(rng), _vocals(rng)
    (d / "stems").mkdir()
    st = lambda x: np.stack([x, x * 0.98], 1)
    sf.write(d / "stems" / "no_vocals.wav", st(inst), SR, subtype="PCM_16")
    sf.write(d / "stems" / "vocals.wav", st(voc), SR, subtype="PCM_16")
    sf.write(d / "source.wav", st(0.8 * (inst + voc)), SR, subtype="PCM_16")
    words = []
    for (a, b), text in zip(LINE_TIMES, LYRICS):
        ws = text.split()
        step = (b - a - 0.2) / len(ws)
        words += [{"type": "word", "text": w, "start": a + k * step, "end": a + k * step + step * 0.8} for k, w in enumerate(ws)]
    return {"dir": d, "words": words, "lyrics": "\n".join(LYRICS), "duration": DURATION}


@pytest.fixture
def workspace(synth, tmp_path) -> Path:
    """A fresh work directory that already contains the decoded song and its stems (so Demucs is never needed)."""
    w = tmp_path / "work"
    shutil.copytree(synth["dir"], w)
    return w


class FakeEleven:
    """Stands in for the ElevenLabs client: Scribe returns the synthetic transcript, TTS returns a short sine-wave mp3."""
    def __init__(self, words=None):
        self.words, self.tts_calls, self.scribe_calls = words or [], [], 0

    def scribe(self, audio, language=None):
        self.scribe_calls += 1
        return {"words": self.words, "language_code": language}

    def tts(self, text, voice, model, speed, stability):
        self.tts_calls.append((text, voice))
        dur = 0.4 + 0.06 * len(text)
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency={440 if voice == 'src' else 330}:duration={dur}",
                            "-ar", "44100", "-b:a", "128k", "-f", "mp3", "-"], capture_output=True, check=True)
        return r.stdout

    def resolve_voice(self, spec, lang, gender="male"):
        return spec or ("src" if lang in ("en", "xx") else "dst")


@pytest.fixture
def fake_eleven(synth):
    return FakeEleven(synth["words"])
