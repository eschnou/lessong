"""Plan: translate lyrics, decide which lines to narrate, group lines into learning sections."""
from __future__ import annotations

import json
import os

from .audio import log
from .config import Settings
from .lyrics import REAL_PAUSE, norm

LANG_NAMES = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian", "pt": "Portuguese",
              "nl": "Dutch", "sv": "Swedish", "pl": "Polish", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
              "ru": "Russian", "ar": "Arabic", "tr": "Turkish", "hi": "Hindi"}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code.lower(), code)


def translate(lines: list[dict], src: str, dst: str, model: str) -> list[str]:
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("error: OPENAI_API_KEY is not set (needed for translation)")
    client = OpenAI()
    numbered = "\n".join(f"{l['i']}: {l['text']}" for l in lines)
    system = (f"You are a professional song-lyric translator helping language learners. Translate each lyric line from "
              f"{lang_name(src)} to {lang_name(dst)}. Rules: (1) Translate the meaning faithfully, in natural, idiomatic phrasing a "
              f"native speaker would actually say; keep the register (colloquial stays colloquial). (2) Make every line grammatical "
              f"on its own in {lang_name(dst)}: add the articles, possessives and pronouns the target language requires, even if the "
              f"original omits them (e.g. 'Brother waiting at the door' -> 'Mon frère attend à la porte'; "
              f"'Mother wears her red coat' -> 'Maman porte son manteau rouge'), and use the context of the whole song to choose them. (3) Exactly "
              f"one output line per input line, never merging or splitting; a fragment stays a fragment. (4) Keep each line about as "
              f"short as the original so it can be spoken in the time of the sung line. (5) Keep proper nouns. No notes or explanations. "
              f'Reply with JSON: {{"lines": [{{"i": <int>, "translation": <string>}}, ...]}}.')
    for attempt in range(3):
        extra = {"temperature": 0} if model.startswith("gpt-4") else {}   # reasoning models reject temperature
        r = client.chat.completions.create(model=model, response_format={"type": "json_object"}, **extra,
                                           messages=[{"role": "system", "content": system}, {"role": "user", "content": numbered}])
        try:
            got = {int(x["i"]): str(x["translation"]).strip() for x in json.loads(r.choices[0].message.content)["lines"]}
        except (ValueError, KeyError, TypeError):
            got = {}
        if all(l["i"] in got and got[l["i"]] for l in lines):
            return [got[l["i"]] for l in lines]
        log(f"[translate] incomplete answer (attempt {attempt + 1}/3), retrying")
    raise RuntimeError("translation failed: model did not return one translation per line")


def mark_narration(lines: list[dict]) -> None:
    """Every line of a section is taught: the lesson must cover exactly what the excerpt then plays. (The flag stays in plan.json
    so a line can still be muted by hand.)"""
    for l in lines:
        l["narrate"] = True


def make_sections(lines: list[dict], s: Settings) -> list[list[int]]:
    """Group consecutive lines. Break at long silences, split long groups at the biggest pause, merge tiny ones."""
    n = len(lines)
    w = [1 if l["narrate"] else 0 for l in lines]
    # silence after line i: the real vocal pause when known (word timestamps alone hide held notes), else the timestamp gap
    gap = lambda i: lines[i].get("vgap", lines[i + 1]["start"] - lines[i]["end"])
    spans, cur = [], [0]
    for i in range(1, n):
        if gap(i - 1) >= s.section_gap:
            spans.append(cur)
            cur = []
        cur.append(i)
    spans.append(cur)

    def weight(sp): return sum(w[i] for i in sp)

    def seconds(sp): return lines[sp[-1]]["end"] - lines[sp[0]]["start"]

    def split(sp):
        too_many = weight(sp) > s.max_lines
        if not too_many and seconds(sp) <= s.max_song_seconds:
            return [sp]
        need = s.min_lines if too_many else max(1, s.min_lines // 2)   # a pure length split may be a bit shorter
        ok = [k for k in range(1, len(sp)) if weight(sp[:k]) >= need and weight(sp[k:]) >= need]
        if not ok:
            return [sp]
        k = max(ok, key=lambda k: gap(sp[k - 1]))
        return split(sp[:k]) + split(sp[k:])

    spans = [p for sp in spans for p in split(sp)]
    kept: set[int] = set()   # spans we may not merge without breaking the song-length cap
    while len(spans) > 1:
        small = [k for k, sp in enumerate(spans) if weight(sp) < s.min_lines and id(sp) not in kept]
        if not small:
            break
        k = small[0]
        left = gap(spans[k - 1][-1]) if k > 0 else 1e9
        right = gap(spans[k][-1]) if k < len(spans) - 1 else 1e9
        j = k - 1 if left <= right else k + 1
        merged = spans[min(j, k)] + spans[max(j, k)]
        if weight(spans[k]) > 0 and seconds(merged) > s.max_song_seconds:
            kept.add(id(spans[k]))
            continue
        spans[min(j, k)] = merged
        spans.pop(max(j, k))
    return spans


def llm_structure(lines: list[dict], s: Settings) -> list[dict] | None:
    """Ask the LLM to segment the lyrics into structural sections. Returns [{label, first, last}] or None."""
    from openai import OpenAI
    n = len(lines)
    rows = "\n".join(f"{l['i']}: {l['text']}   [t={l['start']:.0f}s, silence after: {l.get('vgap', 0):.1f}s]" for l in lines)
    system = (
        "You are a music analyst. Segment these song lyrics into structural sections for a language lesson: verse, pre-chorus, "
        "chorus, bridge, outro, etc. Rules: (1) Sections are contiguous, in order, and together cover every line exactly once; a "
        "boundary falls between lines, never inside one. (2) A verse and the chorus that follows are separate sections, even when "
        "sung without a break. (3) Every recurrence of the chorus (or any repeated block) is its own section with the same label. "
        "Repeated text is the best clue to the chorus. (4) Use the timing: a long silence after a line is usually a boundary. "
        f"(5) If a section would be longer than {s.max_lines} lines, split it at a natural phrase boundary. Avoid one-line sections "
        "unless the line is truly isolated. "
        'Reply with JSON: {"sections": [{"label": "verse 1", "first": <first line index>, "last": <last line index>}, ...]}.')
    client = OpenAI()
    extra = {"temperature": 0} if s.llm_model.startswith("gpt-4") else {}
    for attempt in range(3):
        r = client.chat.completions.create(model=s.llm_model, response_format={"type": "json_object"}, **extra,
                                           messages=[{"role": "system", "content": system}, {"role": "user", "content": rows}])
        try:
            segs = sorted(({"label": str(x.get("label") or "section"), "first": int(x["first"]), "last": int(x["last"])}
                           for x in json.loads(r.choices[0].message.content)["sections"]), key=lambda x: x["first"])
        except (ValueError, KeyError, TypeError):
            segs = []
        if segs and segs[0]["first"] == 0 and segs[-1]["last"] == n - 1 and all(
                x["first"] <= x["last"] and (k == 0 or x["first"] == segs[k - 1]["last"] + 1) for k, x in enumerate(segs)):
            return segs
        log(f"[sections] the LLM's segmentation did not cover the lyrics exactly once (attempt {attempt + 1}/3)")
    return None


def _split_long(lines: list[dict], sp: list[int], max_w: int, need: int = 2) -> list[list[int]]:
    w = lambda part: sum(1 for i in part if lines[i]["narrate"])
    if w(sp) <= max_w:
        return [sp]
    ok = [k for k in range(1, len(sp)) if w(sp[:k]) >= need and w(sp[k:]) >= need
          and lines[sp[k - 1]].get("vgap", 1.0) >= REAL_PAUSE]       # never split inside continuous singing
    if not ok:
        return [sp]
    # prefer a long pause, but also a balanced cut (each line in the smaller half is worth 0.15 s of pause)
    k = max(ok, key=lambda k: lines[sp[k - 1]].get("vgap", 0) + 0.15 * min(w(sp[:k]), w(sp[k:])))
    return _split_long(lines, sp[:k], max_w, need) + _split_long(lines, sp[k:], max_w, need)


def _is_adlib(lines: list[dict], idx: list[int]) -> bool:
    """A tiny isolated fragment (under a second, at most two words) is an ad-lib, not something to teach."""
    ls = [lines[i] for i in idx]
    return (ls[-1]["end"] - ls[0]["start"]) < 1.0 and sum(len(l["text"].split()) for l in ls) <= 2


def structured_sections(lines: list[dict], s: Settings) -> list[dict]:
    """Sections by song structure (verse / chorus / verse ...). Falls back to pause-based sections if the LLM fails."""
    log(f"[sections] asking {s.llm_model} for the song structure")
    try:
        segs = llm_structure(lines, s)
    except Exception as e:   # network / API errors must not lose the whole plan
        log(f"[sections] LLM call failed ({e})")
        segs = None
    if segs is None:
        log("[sections] falling back to pause-based sections (--sections gap)")
        return [{"label": None, "lines": sp} for sp in make_sections(lines, s)]
    out: list[dict] = []
    for g in segs:
        parts = _split_long(lines, list(range(g["first"], g["last"] + 1)), s.max_lines)
        for k, part in enumerate(parts):
            out.append({"label": g["label"] + (f" ({k + 1}/{len(parts)})" if len(parts) > 1 else ""), "lines": part})
    # Two sections sung without a real pause between them cannot be played separately (the song would be cut mid-phrase):
    # they become one lesson and one uninterrupted excerpt.
    out = [o for o in out if not _is_adlib(lines, o["lines"])]
    k = 0
    while k < len(out) - 1:
        if lines[out[k]["lines"][-1]].get("vgap", 1.0) < REAL_PAUSE:
            merged = out[k]["lines"] + out[k + 1]["lines"]
            label = f"{out[k]['label']} + {out[k + 1]['label']}"
            parts = _split_long(lines, merged, int(1.5 * s.max_lines))     # too long for one lesson: cut at the best REAL pause inside
            if len(parts) == 1:
                log(f"[sections] '{out[k]['label']}' runs straight into '{out[k + 1]['label']}' without a pause: joined")
                out[k:k + 2] = [{"label": label, "lines": merged}]
            else:
                log(f"[sections] '{out[k]['label']}' + '{out[k + 1]['label']}' run together: cut at the best pause inside instead ({len(parts)} parts)")
                out[k:k + 2] = [{"label": f"{label} ({i + 1}/{len(parts)})", "lines": part} for i, part in enumerate(parts)]
                k += len(parts) - 1
        else:
            k += 1
    return out


def drop_repeats(lines: list[dict], sections: list[dict]) -> list[dict]:
    """--repeats skip: a section whose every line already appeared in an earlier section is removed from the track."""
    seen: set[str] = set()
    kept = []
    for sec in sections:
        keys = {norm(lines[i]["text"]) for i in sec["lines"]}
        if keys <= seen:
            log(f"[sections] '{sec.get('label') or sec['lines'][0]}' repeats earlier material: left out (--repeats teach to keep it)")
        else:
            kept.append(sec)
        seen |= keys
    return kept


def make_plan(lines: list[dict], s: Settings, meta: dict, translate_fn=None, known: dict | None = None) -> dict:
    mark_narration(lines)
    known = known or {}
    if all(norm(l["text"]) in known for l in lines):
        log("[translate] reusing the translations already in plan.json (--retranslate to redo them)")
        tr = [known[norm(l["text"])] for l in lines]
    else:
        log(f"[translate] {len(lines)} lines {s.source_lang} -> {s.target_lang} with {s.llm_model}")
        tr = (translate_fn or translate)(lines, s.source_lang, s.target_lang, s.llm_model)
    uniq: dict[str, str] = {}
    for l, t in zip(lines, tr):
        l["translation"] = uniq.setdefault(norm(l["text"]), t)   # identical lines get identical translations
    sections = structured_sections(lines, s) if s.section_mode == "llm" else [{"label": None, "lines": sp} for sp in make_sections(lines, s)]
    if s.repeats == "skip":
        sections = drop_repeats(lines, sections)
    return {"meta": meta, "source_lang": s.source_lang, "target_lang": s.target_lang, "lines": lines, "sections": sections}


def summarize(plan: dict) -> str:
    L = plan["lines"]
    out, chars = [], 0
    for k, sec in enumerate(plan["sections"], 1):
        ls = [L[i] for i in sec["lines"]]
        nar = [l for l in ls if l["narrate"]]
        chars += sum(len(l["text"]) + len(l["translation"]) for l in nar)
        cut = sec["lines"][-1] < len(L) - 1 and L[sec["lines"][-1]].get("vgap", 9) < 0.3
        label = sec.get("label") or ls[0]["text"][:30]
        out.append(f"  section {k}: song {ls[0]['start']:6.1f}-{ls[-1]['end']:6.1f}s  {len(ls):2d} lines, {len(nar):2d} narrated  | {label}" + ("  (!) ends inside continuous singing" if cut else ""))
    out.append(f"  ElevenLabs TTS cost: about {chars} characters")
    return "\n".join(out)
