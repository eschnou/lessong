from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from . import __version__, config, video
from . import pipeline as pl
from .audio import log
from .elevenlabs import ElevenLabs


def _common(p: argparse.ArgumentParser, needs_out: bool = False) -> None:
    p.add_argument("input", help="song file (mp3/mp4/wav/...) or an existing work directory")
    p.add_argument("--workdir", type=Path, help="where intermediate files live [default: .lessong/<song>]")
    p.add_argument("--env-file", help="path to a .env with ELEVENLABS_API_KEY and OPENAI_API_KEY")
    p.add_argument("--force", action="store_true", help="redo everything, including vocal separation")
    p.add_argument("--replan", action="store_true", help="redo lyrics, translation and sections (keeps the vocal stems)")
    p.add_argument("--retranslate", action="store_true", help="with --replan: translate again instead of keeping the translations in plan.json")
    p.add_argument("--title", help="song title (default: from file tags or file name)")
    p.add_argument("--artist", help="artist (default: from file tags)")
    if needs_out:
        p.add_argument("-o", "--output", type=Path, help="output file; the extension picks the format (.mp4 = a lesson video) [default: <song>.lessong.mp3]")


def doctor() -> int:
    """First-run check: everything the pipeline needs, with a fix hint for whatever is missing."""
    import shutil
    import sys

    import requests

    ok = True

    def line(good: bool, what: str, hint: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'ok' if good else '!!'}] {what}" + (f"  ->  {hint}" if hint and not good else ""))

    print(f"lessong {__version__} on Python {sys.version.split()[0]}")
    for tool in ("ffmpeg", "ffprobe"):
        line(bool(shutil.which(tool)), tool, "install ffmpeg (macOS: brew install ffmpeg, Debian/Ubuntu: apt install ffmpeg)")
    try:
        from .separate import auto_device
        line(True, f"Demucs compute device: {auto_device()}")
    except Exception as e:  # noqa: BLE001
        line(False, "PyTorch / Demucs", f"reinstall the package ({e})")
    for env, url, hdr in (("ELEVENLABS_API_KEY", "https://api.elevenlabs.io/v1/models", "xi-api-key"),
                          ("OPENAI_API_KEY", "https://api.openai.com/v1/models", "Authorization")):
        key = os.environ.get(env)
        if not key:
            line(False, env, "put it in a .env file or export it")
            continue
        try:
            r = requests.get(url, headers={hdr: key if hdr == "xi-api-key" else f"Bearer {key}"}, timeout=15)
            line(r.status_code == 200, f"{env} is accepted by the API" if r.status_code == 200 else f"{env} rejected (HTTP {r.status_code})",
                 "check the key and its permissions")
        except requests.RequestException as e:
            line(False, f"{env}: could not reach the API", str(e)[:80])
    print("ready." if ok else "fix the items marked !! above.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lessong", description="Turn a song you love into a language lesson.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Typical use:\n  lessong build song.mp3 --title 'Song Title' --artist 'Artist Name' --to fr -o out.mp3\n"
                                        "Tweak the result without redoing the slow steps:\n  lessong render song.mp3 --bed-db -20 --loop-bars 4 -o out.mp3")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="full pipeline: separate, lyrics, translate, plan, render")
    _common(b, True)
    config.add_arguments(b, {"plan", "render", "both"})
    pp = sub.add_parser("plan", help="stop after writing plan.json (edit it, then run render)")
    _common(pp)
    config.add_arguments(pp, {"plan", "both"})
    r = sub.add_parser("render", help="render from an existing plan.json (all mixing/TTS options available)")
    _common(r, True)
    config.add_arguments(r, {"render", "both"})
    sub.add_parser("doctor", help="check that ffmpeg, the API keys and the compute device are ready")
    lp = sub.add_parser("loops", help="list and export candidate instrumental loops for a song")
    _common(lp)
    config.add_arguments(lp, {"render"})

    a = ap.parse_args(argv)
    load_dotenv(getattr(a, "env_file", None) or find_dotenv(usecwd=True))
    if a.cmd == "doctor":
        return doctor()
    s = config.from_args(a)
    if a.cmd in ("build", "render") and (s.video or (a.output and a.output.suffix.lower() == ".mp4")):
        video.parse_size(s.video_size)           # fail fast on a bad --video-size, before any slow or paid step
    try:
        ws = pl.Workspace(Path(a.input), a.workdir)
        if a.cmd in ("build", "plan", "loops") or not ws.vocals.exists():
            pl.prepare(ws, s, a.force)
        if a.cmd == "loops":
            pl.list_loops(ws, s)
            return 0
        eleven = ElevenLabs(os.environ.get("ELEVENLABS_API_KEY"))
        if a.cmd == "render":
            if not ws.plan.exists():
                raise SystemExit(f"error: {ws.plan} not found; run `lessong plan` or `build` first")
            import json
            plan = json.loads(ws.plan.read_text())
        else:
            plan = pl.make_plan(ws, s, eleven, a.title, a.artist, a.replan or a.force, a.force, a.retranslate)
        if a.cmd in ("build", "render"):
            out = a.output or Path(f"{pl.slug(plan['meta'].get('title') or ws.dir.name)}.lessong.mp3")
            pl.do_render(ws, s, eleven, plan, out)
    except (RuntimeError, OSError) as e:
        log(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
