"""Stage orchestration. Every stage caches its output in the work directory, so reruns are cheap
and plan.json can be hand-edited (translations, section boundaries, narrate flags) before rendering."""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio, lyrics
from . import plan as plan_mod
from . import video as video_mod
from .audio import log
from .config import Settings
from .elevenlabs import ElevenLabs
from .loops import Loop, analyse, candidates, pick
from .loops import render as render_loop
from .mix import Voices, assemble
from .separate import separate


def slug(s: str) -> str:
    return re.sub(r"[^\w]+", "-", s.lower()).strip("-") or "song"


# Scribe reports ISO 639-3 codes (fra, nld, ...); everything else here uses ISO 639-1.
ISO3_TO_1 = dict(eng="en", fra="fr", spa="es", deu="de", ita="it", por="pt", nld="nl", swe="sv", pol="pl", rus="ru", jpn="ja", zho="zh",
                 kor="ko", ara="ar", tur="tr", hin="hi", dan="da", nor="no", nob="no", fin="fi", ell="el", ces="cs", hun="hu", ron="ro",
                 ukr="uk", heb="he", ind="id", tha="th", vie="vi", cat="ca", hrv="hr", bul="bg", slk="sk", slv="sl", srp="sr", msa="ms",
                 ben="bn", tam="ta", urd="ur", fas="fa", lit="lt", lav="lv", est="et", isl="is", gle="ga", cym="cy", eus="eu", glg="gl")
SHARED_FILES = ("source.wav", "stems", "scribe_vocals.json", "lrclib.json", "meta.json")      # language-independent work, reused across languages


def iso1(code: str | None) -> str | None:
    code = (code or "").lower()
    return code if len(code) == 2 else ISO3_TO_1.get(code)


def default_target(source: str) -> str:
    """English for any song that is not English; French for an English song (the original show's pairing)."""
    return "en" if source != "en" else "fr"


def plan_pair(ws: Workspace) -> tuple[str, str] | None:
    if not ws.plan.exists():
        return None
    p = json.loads(ws.plan.read_text())
    return p.get("source_lang"), p.get("target_lang")


def conflicts(pair: tuple[str, str], explicit_from: str | None, explicit_to: str | None) -> bool:
    """Does what the user asked for disagree with the languages of an existing plan? (Languages not asked for never conflict.)"""
    return bool((explicit_from and explicit_from != pair[0]) or (explicit_to and explicit_to != pair[1]))


class Workspace:
    def __init__(self, input_path: Path, workdir: Path | None):
        if input_path.is_dir():
            self.dir, self.input = input_path, None
        else:
            self.input = input_path
            self.dir = workdir or Path(".lessong") / slug(input_path.stem)
        if workdir and input_path.is_dir():
            self.dir = workdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.source = self.dir / "source.wav"
        self.stems = self.dir / "stems"
        self.scribe = self.dir / "scribe_vocals.json"
        self.lyrics = self.dir / "lyrics.json"
        self.plan = self.dir / "plan.json"
        self.tts = self.dir / "tts"

    @property
    def vocals(self): return self.stems / "vocals.wav"
    @property
    def inst(self): return self.stems / "no_vocals.wav"


def choose_workspace(input_path: Path, workdir: Path | None, explicit_from: str | None, explicit_to: str | None) -> Workspace:
    """The work folder for this run. A default folder that already holds a plan for other languages is left alone: the run gets
    its own folder (<song>-<from>-<to>) seeded with the separated stems, so asking for a new language never overwrites an old one
    and Demucs does not run twice. An explicit --workdir (or a folder given as input) is used as is."""
    ws = Workspace(input_path, workdir)
    pair = plan_pair(ws)
    if pair is None or not conflicts(pair, explicit_from, explicit_to) or workdir is not None or input_path.is_dir():
        return ws
    alt_dir = ws.dir.parent / f"{ws.dir.name}-{explicit_from or pair[0]}-{explicit_to or pair[1]}"
    created = not alt_dir.exists()
    alt = Workspace(input_path, alt_dir)
    if created or not alt.source.exists():
        for name in SHARED_FILES:
            src = ws.dir / name
            if src.is_dir():
                shutil.copytree(src, alt.dir / name, dirs_exist_ok=True)
            elif src.exists():
                shutil.copy2(src, alt.dir / name)
    log(f"[plan] {ws.dir} holds a {pair[0]} -> {pair[1]} plan; using {alt.dir} for {explicit_from or pair[0]} -> {explicit_to or pair[1]} "
        f"(the separated stems are copied, nothing existing is touched)")
    return alt


def resolve_languages(ws: Workspace, s: Settings, explicit_from: str | None, explicit_to: str | None, replanning: bool) -> None:
    """Fix s.source_lang / s.target_lang: an existing plan decides (unless it conflicts with what was asked), otherwise what was asked,
    otherwise None (= detect the song's language, then derive the target)."""
    pair = plan_pair(ws)
    if pair and not replanning:
        if conflicts(pair, explicit_from, explicit_to):
            raise SystemExit(
                f"error: {ws.plan} is a {pair[0]} -> {pair[1]} plan, but you asked for {explicit_from or pair[0]} -> {explicit_to or pair[1]}.\n"
                f"  --replan   rebuilds this plan for the new languages (overwrites it)\n"
                f"  --workdir  DIR   uses a separate folder (without --workdir, a default folder is split by language automatically)")
        s.source_lang, s.target_lang = pair
        return
    s.source_lang, s.target_lang = explicit_from, explicit_to
    if explicit_from and not explicit_to:
        s.target_lang = default_target(explicit_from)


def prepare(ws: Workspace, s: Settings, force: bool) -> None:
    """decode + vocal separation"""
    audio.need_tools()
    if not ws.source.exists() or force:
        if ws.input is None:
            raise SystemExit(f"error: {ws.dir} has no source audio; pass the song file")
        log(f"[decode] {ws.input}")
        audio.decode(ws.input, ws.source)
    separate(ws.source, ws.stems, s.device, force)


def make_plan(ws: Workspace, s: Settings, eleven: ElevenLabs, title: str | None, artist: str | None, replan: bool, force: bool = False, retranslate: bool = False) -> dict:
    if ws.plan.exists() and not replan:
        log(f"[plan] using existing {ws.plan} (edit it freely; --replan regenerates it)")
        return json.loads(ws.plan.read_text())
    meta_file = ws.dir / "meta.json"
    saved = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    tg = audio.tags(ws.input) if ws.input else {}
    title = title or saved.get("title") or tg.get("title") or (ws.input.stem.replace("_", " ") if ws.input else None)
    artist = artist or saved.get("artist") or tg.get("artist")
    meta_file.write_text(json.dumps({"title": title, "artist": artist}))
    if ws.scribe.exists() and not force:
        data = json.loads(ws.scribe.read_text())
    else:
        log("[transcribe] Scribe on the isolated vocals" + ("" if s.source_lang else " (detecting the language)"))
        mp3 = ws.dir / "vocals_mono.mp3"
        audio.to_mono_mp3(ws.vocals, mp3)
        data = eleven.scribe(mp3, s.source_lang)           # no language given: Scribe detects it
        ws.scribe.write_text(json.dumps(data))
        mp3.unlink(missing_ok=True)
    words = data["words"]
    if s.source_lang is None:
        s.source_lang = iso1(data.get("language_code"))
        if not s.source_lang:
            raise SystemExit(f"error: could not tell which language is sung (detected '{data.get('language_code')}'); pass --from CODE")
        log(f"[transcribe] the song is in {plan_mod.lang_name(s.source_lang)} ({s.source_lang}); use --from to override")
    if s.target_lang is None:
        s.target_lang = default_target(s.source_lang)
        log(f"[plan] translating into {plan_mod.lang_name(s.target_lang)} ({s.target_lang}); use --to for another language")
    if s.source_lang == s.target_lang:
        raise SystemExit(f"error: the song and the translation are both '{s.source_lang}'; choose another --to")
    known: dict[str, str] = {}
    if ws.plan.exists() and not retranslate and not force:      # keep (possibly hand-edited) translations across a --replan
        old = json.loads(ws.plan.read_text())
        if (old.get("source_lang"), old.get("target_lang")) == (s.source_lang, s.target_lang):
            known = {lyrics.norm(l["text"]): l["translation"] for l in old["lines"] if l.get("translation")}
    text = Path(s.lyrics_file).read_text() if s.lyrics_file else None
    act = lyrics.vocal_activity(ws.vocals)
    lines, src = lyrics.build_lines(title, artist, text, s.no_lookup, words, ws.dir / "lrclib.json" if not force and not replan else None, act)
    ws.lyrics.write_text(json.dumps(lines, indent=1, ensure_ascii=False))
    lyrics.annotate_vocals(lines, act)
    plan = plan_mod.make_plan(lines, s, {"title": title, "artist": artist, "lyrics_source": src}, known=known)
    ws.plan.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    log(f"[plan] wrote {ws.plan}\n{plan_mod.summarize(plan)}")
    return plan


def _loop_key(s: Settings) -> dict:
    return {k: getattr(s, k) for k in ("loop_bars", "loop_start", "loop_crossfade", "loop_vfree_db", "loop_loudness_tol", "beats_per_bar")}


def do_render(ws: Workspace, s: Settings, eleven: ElevenLabs, plan: dict, out: Path) -> None:
    if "sing_end" not in plan["lines"][0]:      # plan written by an older version
        log("[plan] adding vocal timing to plan.json (use --replan to also re-section on real pauses)")
        lyrics.annotate_vocals(plan["lines"], lyrics.vocal_activity(ws.vocals))
        ws.plan.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    lf, ff = ws.dir / "loop.json", ws.dir / "loop_features.npz"
    cached = json.loads(lf.read_text()) if lf.exists() else {}
    if cached.get("key") == _loop_key(s) and ff.exists():
        loop = Loop(**cached["loop"])
        nov, _ = sf.read(ws.inst)
        log(f"[loop] cached: {loop.bars} bars @ {loop.t:.2f}s")
    else:
        log("[loop] analysing the instrumental")
        an = analyse(ws.inst, ws.vocals, s.beats_per_bar)
        loop, nov = pick(an, s), an["nov"]
        lf.write_text(json.dumps({"key": _loop_key(s), "loop": asdict(loop)}, indent=1))
        np.savez(ff, S=an["S"].astype("float32"), env=an["env"].astype("float32"), bt=an["bt"], fr=an["fr"], efps=an["efps"])
        log(f"[loop] {loop.bars} bars @ {loop.t:.2f}s, {loop.length:.2f}s long, repeat-sim {loop.sim:.2f}, seam {loop.seam:.2f}"
            + (f" ({loop.note})" if loop.note else ""))
        if loop.sim < 0.3:
            log("[loop] warning: this loop does not repeat very exactly; listen to it, and try `lessong loops` / --loop-start / --loop-bars")
    feat = dict(np.load(ff))
    src_v = eleven.resolve_voice(s.src_voice, plan["source_lang"], s.voice_gender)
    dst_v = eleven.resolve_voice(s.dst_voice, plan["target_lang"], s.voice_gender)
    s.source_lang, s.target_lang = plan["source_lang"], plan["target_lang"]
    voices = Voices(eleven, s, ws.tts, src_v, dst_v)
    want_video = s.video or out.suffix.lower() == ".mp4"
    audio_out = ws.dir / "lesson_audio.mp3" if out.suffix.lower() == ".mp4" else out      # -o lesson.mp4: the audio is only a by-product
    video_out = out if out.suffix.lower() == ".mp4" else out.with_suffix(".mp4")
    dur, timeline = assemble(plan, s, ws.source, loop, nov, feat, voices, audio_out, ws.dir / "mix.wav")
    (ws.dir / "timeline.json").write_text(json.dumps(timeline, indent=1))
    if want_video:
        video_mod.make_video(timeline, plan, audio_out, video_out, s, plan.get("meta", {}).get("title"))
    log(f"[done] {video_out if want_video and out.suffix.lower() == '.mp4' else out}  ({dur / 60:.1f} min; {voices.chars} new TTS characters)"
        + (f"; video: {video_out}" if want_video and out.suffix.lower() != ".mp4" else ""))


def list_loops(ws: Workspace, s: Settings, bars_list=(1, 2, 4)) -> None:
    an = analyse(ws.inst, ws.vocals, s.beats_per_bar)
    d = ws.dir / "loops"
    d.mkdir(exist_ok=True)
    print("best loops per length (repeat-sim: 1 = identical cycles; seam: 1 = as natural as the song's own bar lines)")
    for b in bars_list:
        cs, note = candidates(an, b, s)
        if not cs:
            print(f"  {b} bar: none ({note})")
        for l in cs[:2]:
            audio_ = render_loop(an["nov"], l.t, l.length, s.loop_crossfade)
            p = d / f"loop_{b}bar_{l.t:.0f}s.wav"
            sf.write(p, np.tile(audio_, (max(2, int(24 / l.length)), 1)), audio.SR)
            print(f"  {b} bar @ {l.t:7.2f}s  len {l.length:.2f}s  repeat-sim {l.sim:.2f}  seam {l.seam:.2f}  -> {p}   (use --loop-bars {b} --loop-start {l.t:.2f})")
