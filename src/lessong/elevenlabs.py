"""Minimal ElevenLabs client: Scribe (speech-to-text), text-to-speech, voice lookup."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from .audio import log

API = "https://api.elevenlabs.io/v1"

# Sensible defaults for the most common pairing; other languages are resolved from the voice library.
DEFAULT_VOICES = {
    ("en", "male"): "onwK4e9ZLuTAKqWW03F9",    # Daniel - steady British broadcaster
    ("en", "female"): "Xb7hH8MSUJpSbSDYk0k2",  # Alice - clear, engaging educator (British)
    ("fr", "male"): "1a3lMdKLUcfcMtvN772u",    # Antoine - Parisian, explanatory
    ("fr", "female"): "PSVUmed8NvS8aUA3d5oO",  # Anna - French audiobook narrator
}


class ElevenLabs:
    def __init__(self, key: str | None):
        if not key:
            raise SystemExit("error: ELEVENLABS_API_KEY is not set (put it in .env or the environment)")
        self.h = {"xi-api-key": key}
        self._voices: list[dict] | None = None

    def _req(self, method: str, path: str, tries: int = 4, **kw) -> requests.Response:
        for i in range(tries):
            r = requests.request(method, API + path, headers=self.h, timeout=kw.pop("timeout", 180), **kw)
            if r.status_code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(2 * (i + 1))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"ElevenLabs {path} -> {r.status_code}: {r.text[:300]}")
            return r
        raise RuntimeError("unreachable")

    def scribe(self, audio: Path, language: str | None = None) -> dict:
        data = {"model_id": "scribe_v1", "timestamps_granularity": "word", "tag_audio_events": "false"}
        if language:
            data["language_code"] = language
        with open(audio, "rb") as f:
            return self._req("POST", "/speech-to-text", data=data, files={"file": f}, timeout=600).json()

    def tts(self, text: str, voice: str, model: str, speed: float | None, stability: float | None) -> bytes:
        body: dict = {"text": text, "model_id": model}
        vs = {k: v for k, v in (("speed", speed), ("stability", stability)) if v is not None}
        if vs:
            body["voice_settings"] = vs
        return self._req("POST", f"/text-to-speech/{voice}?output_format=mp3_44100_128", json=body, timeout=90).content

    def voices(self) -> list[dict]:
        if self._voices is None:
            self._voices = self._req("GET", "/voices").json()["voices"]
        return self._voices

    def resolve_voice(self, spec: str | None, lang: str, gender: str = "male") -> str:
        """voice id / name substring / None (auto, by language and gender) -> voice id."""
        if spec:
            for v in self.voices():
                if spec == v["voice_id"]:
                    return spec
            for v in self.voices():
                if spec.lower() in v["name"].lower():
                    log(f"[tts] voice '{spec}' -> {v['name']}")
                    return v["voice_id"]
            if len(spec) >= 18 and spec.isalnum():
                return spec  # probably an id from the shared library
            raise SystemExit(f"error: no voice matching '{spec}' in your voice library")
        if (lang, gender) in DEFAULT_VOICES:
            return DEFAULT_VOICES[(lang, gender)]
        pool = [v for v in self.voices() if lang in {x.get("language") for x in (v.get("verified_languages") or [])} | {(v.get("labels") or {}).get("language")}]
        pool.sort(key=lambda v: (v.get("labels") or {}).get("language") != lang)       # native voices first, then ones that merely speak it
        same = [v for v in pool if (v.get("labels") or {}).get("gender") == gender]
        if same or pool:
            v = (same or pool)[0]
            log(f"[tts] no default {gender} voice for '{lang}', using {v['name']}"
                + ("" if same else f" (no {gender} voice found for this language)") + " (override with --src-voice/--dst-voice)")
            return v["voice_id"]
        log(f"[tts] no native voice found for '{lang}'; using the English default (accent will be off)")
        return DEFAULT_VOICES[("en", gender)]
