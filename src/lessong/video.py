"""The lesson as a video: what is being said and sung, on screen, in sync with the audio.

One still frame is rendered (with Pillow) per visual *state* (which line is being spoken, whether its translation is showing, which
line of the song is being sung, ...) and ffmpeg's concat demuxer assembles them with the audio, so a five-minute video costs a few
hundred small PNGs and a few seconds, not thousands of frames. Everything is driven by the `timeline` that `mix.assemble` returns."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .audio import log, tags
from .config import Settings
from .plan import lang_name

ASSETS = Path(__file__).parent / "assets"
BG, WHITE = (20, 27, 45), (255, 255, 255)
DIM, FUTURE, TRACK = (125, 139, 176), (88, 100, 132), (36, 48, 73)
ACCENT, ACCENT_LIGHT, ORANGE = (77, 124, 255), (169, 196, 255), (255, 159, 67)


class Style:
    """Sizes are given for a 1280x720 canvas and scaled to the real one."""

    def __init__(self, size: tuple[int, int], font_path: str | None = None):
        self.w, self.h = size
        self.k = self.w / 1280
        self.font_path = font_path
        self._fonts: dict = {}

    def px(self, v: float) -> int:
        return int(round(v * self.k))

    def font(self, size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (self.px(size), bold)
        if key not in self._fonts:
            path = self.font_path or str(ASSETS / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))
            self._fonts[key] = ImageFont.truetype(path, max(8, key[0]))
        return self._fonts[key]


def parse_size(spec: str) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in spec.lower().split("x"))
        if w < 320 or h < 180 or w % 2 or h % 2:
            raise ValueError
        return w, h
    except ValueError:
        raise SystemExit(f"error: --video-size must look like 1280x720 (even numbers, at least 320x180), got '{spec}'") from None


def wrap(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    """Greedy word wrap; a token wider than the line (e.g. Chinese/Japanese, which has no spaces) is split by character."""
    lines, cur = [], ""
    for word in text.split():
        if font.getlength(word) > max_w:
            for ch in word:
                if cur and font.getlength(cur + ch) > max_w:
                    lines.append(cur)
                    cur = ch
                else:
                    cur += ch
            continue
        trial = f"{cur} {word}".strip()
        if cur and font.getlength(trial) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


def fit(st: Style, text: str, bold: bool, max_w: float, max_h: float, size: float, min_size: float = 20, spacing: float = 1.3):
    """Largest font size (<= size) whose wrapped text fits in max_w x max_h. Returns (font, lines, size)."""
    s = size
    while True:
        f = st.font(s, bold)
        lines = wrap(text, f, max_w)
        if len(lines) * st.px(s) * spacing <= max_h or s <= min_size:
            return f, lines, s
        s -= 2


def centered_block(d: ImageDraw.ImageDraw, st: Style, text: str, *, bold: bool, fill, cy: float, max_w: float, max_h: float, size: float) -> None:
    f, lines, s = fit(st, text, bold, max_w, max_h, size)
    lh = st.px(s) * 1.3
    y = st.px(cy) - lh * len(lines) / 2 + lh / 2
    for ln in lines:
        d.text((st.w / 2, y), ln, font=f, fill=fill, anchor="mm")
        y += lh


def cap(s: str | None) -> str:
    return (s[:1].upper() + s[1:]) if s else ""


def mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


# ----------------------------------------------------------------------------------------------------------------------------
class Scenes:
    def __init__(self, st: Style, plan: dict, title: str):
        self.st, self.plan, self.title = st, plan, title or ""
        self.L = plan["lines"]
        self.src_name = lang_name(plan["source_lang"]).upper()
        self.dst_name = lang_name(plan["target_lang"]).upper()

    def _chrome(self, img: Image.Image, mode: str | None, label: str | None) -> None:
        st, d = self.st, ImageDraw.Draw(img)
        d.text((st.px(64), st.px(50)), "lessong", font=st.font(30, True), fill=ACCENT_LIGHT, anchor="lm")
        if mode:
            text, col = ("LESSON", ACCENT) if mode == "lesson" else ("SONG", ORANGE)
            f, pad = st.font(22, True), st.px(16)
            x1 = st.w - st.px(64)
            x0 = x1 - f.getlength(text) - 2 * pad
            d.rounded_rectangle([x0, st.px(32), x1, st.px(70)], radius=st.px(19), fill=col)
            d.text(((x0 + x1) / 2, st.px(51)), text, font=f, fill=WHITE, anchor="mm")
            if label:
                d.text((x0 - st.px(18), st.px(50)), cap(label), font=st.font(26), fill=DIM, anchor="rm")

    def render(self, state: tuple, label: str | None) -> Image.Image:
        st = self.st
        img = Image.new("RGB", (st.w, st.h), BG)
        d = ImageDraw.Draw(img)
        kind = state[0]
        mw = st.w - 2 * st.px(110)
        if kind == "lesson":
            _, line_i, show_dst, pos, total = state
            self._chrome(img, "lesson", label)
            l = self.L[line_i]
            d.text((st.w / 2, st.px(165)), self.src_name, font=st.font(22, True), fill=DIM, anchor="mm")
            centered_block(d, st, l["text"], bold=True, fill=WHITE, cy=262, max_w=mw, max_h=st.px(170), size=60)
            if show_dst:
                d.text((st.w / 2, st.px(395)), self.dst_name, font=st.font(22, True), fill=DIM, anchor="mm")
                centered_block(d, st, l["translation"], bold=False, fill=ACCENT_LIGHT, cy=490, max_w=mw, max_h=st.px(150), size=50)
            d.text((st.w / 2, st.h - st.px(138)), f"{pos} / {total}", font=st.font(24), fill=FUTURE, anchor="mm")
        elif kind == "song":
            _, lines, cur = state
            self._chrome(img, "song", label)
            self._song_lines(d, lines, cur)
        elif kind == "text":
            self._chrome(img, None, None)
            centered_block(d, st, state[1], bold=True, fill=WHITE, cy=360, max_w=st.w - 2 * st.px(140), max_h=st.px(360), size=66)
        else:                                                   # title card
            self._chrome(img, None, None)
            d.text((st.w / 2, st.px(260)), "lessong", font=st.font(124, True), fill=WHITE, anchor="mm")
            if self.title:
                centered_block(d, st, self.title, bold=True, fill=ACCENT_LIGHT, cy=400, max_w=mw, max_h=st.px(120), size=46)
            if state[1]:
                d.text((st.w / 2, st.px(500)), f"Next: {cap(state[1])}", font=st.font(30), fill=DIM, anchor="mm")
        return img

    def _song_lines(self, d: ImageDraw.ImageDraw, lines: tuple[int, ...], cur: int | None) -> None:
        st = self.st
        top, bottom = st.px(120), st.h - st.px(190)
        n = max(1, len(lines))
        size = min(40, (bottom - top) / n / st.k / 1.45)
        max_w = st.w - 2 * st.px(150)
        while size > 20 and max(st.font(size, True).getlength(self.L[i]["text"]) for i in lines) > max_w:
            size -= 2
        f_cur, f = st.font(size, True), st.font(size)
        lh = (bottom - top) / n
        for k, i in enumerate(lines):
            y = top + lh * (k + 0.5)
            is_cur = cur is not None and k == cur
            past = cur is not None and k < cur
            color = WHITE if is_cur else (DIM if past else FUTURE)
            d.text((st.px(150), y), self.L[i]["text"], font=f_cur if is_cur else f, fill=color, anchor="lm")
            if is_cur:
                d.rounded_rectangle([st.px(122), y - lh * 0.36, st.px(130), y + lh * 0.36], radius=st.px(4), fill=ORANGE)
        if cur is not None:
            centered_block(d, st, self.L[lines[cur]]["translation"], bold=False, fill=ACCENT_LIGHT, cy=(st.h - st.px(122)) / st.k,
                           max_w=st.w - 2 * st.px(110), max_h=st.px(80), size=30)

    def progress(self, img: Image.Image, t: float, total: float) -> Image.Image:
        st, out = self.st, img.copy()
        d = ImageDraw.Draw(out)
        x0, x1, y = st.px(64), st.w - st.px(64), st.h - st.px(52)
        d.rounded_rectangle([x0, y, x1, y + st.px(8)], radius=st.px(4), fill=TRACK)
        fill_to = x0 + (x1 - x0) * min(1.0, max(0.0, t / total))
        if fill_to > x0 + st.px(8):
            d.rounded_rectangle([x0, y, fill_to, y + st.px(8)], radius=st.px(4), fill=ACCENT)
        d.text((x0, y - st.px(20)), self.title, font=st.font(22), fill=FUTURE, anchor="lm")
        d.text((x1, y - st.px(20)), f"{mmss(t)} / {mmss(total)}", font=st.font(22), fill=FUTURE, anchor="rm")
        return out


# ----------------------------------------------------------------------------------------------------------------------------
def build_intervals(timeline: dict, plan: dict, total: float, tick: float = 1.0) -> list[tuple[float, float, tuple, str | None]]:
    """Cut [0, total) into intervals with a constant visual state: [(t0, t1, state, section label)]."""
    L, events, sections = plan["lines"], timeline["events"], timeline["sections"]
    sec_label = {s["section"]: s.get("label") or f"part {s['section']}" for s in sections}
    sec_lines = {s["section"]: tuple(s["lines"]) for s in sections}
    srcs = sorted((e for e in events if e["kind"] == "src"), key=lambda e: e["t0"])
    dsts = {(e["section"], e["line"]): e for e in events if e["kind"] == "dst"}
    songs = [e for e in events if e["kind"] == "song"]
    texts = [e for e in events if e["kind"] in ("intro", "outro")]

    def sung_windows(e):
        out = []
        for k, i in enumerate(sec_lines[e["section"]]):
            a = e["t0"] + (L[i].get("sing_start", L[i]["start"]) - e["src0"])
            b = e["t0"] + (L[i].get("sing_end", L[i]["end"]) - e["src0"])
            out.append((a, b, k))
        return out

    sung = {e["section"]: sung_windows(e) for e in songs}
    # when each section's content really starts (its first spoken line, or its song excerpt if it has no lesson)
    first_at = {}
    for e in srcs + songs:
        first_at[e["section"]] = min(first_at.get(e["section"], 1e18), e["t0"])
    cuts = {0.0, total}
    for e in events:
        cuts |= {e["t0"], e["t1"]}
    for w in sung.values():
        for a, b, _ in w:
            cuts |= {a, b}
    cuts |= set(first_at.values())
    cuts |= {i * tick for i in range(1, int(total / tick) + 1)}
    pts = sorted(c for c in cuts if 0 <= c <= total)
    pts = [p for i, p in enumerate(pts) if i == 0 or p - pts[i - 1] > 0.04]

    def state_at(t: float):
        for e in texts:
            if e["t0"] <= t < e["t1"] + 0.6:
                return ("text", e["text"]), None
        for e in songs:
            if e["t0"] <= t < e["t1"]:
                cur = None
                for a, _, k in sung[e["section"]]:
                    if a <= t:
                        cur = k
                return ("song", sec_lines[e["section"]], cur), sec_label[e["section"]]
        started = [e for e in srcs if e["t0"] <= t]
        if started:
            e = started[-1]
            sec = e["section"]
            song_start = next((x["t0"] for x in songs if x["section"] == sec), 1e18)
            if t < song_start:
                line_ids = [x["line"] for x in srcs if x["section"] == sec]
                show = (sec, e["line"]) in dsts and dsts[(sec, e["line"])]["t0"] <= t
                return ("lesson", e["line"], show, line_ids.index(e["line"]) + 1, len(line_ids)), sec_label[sec]
        upcoming = sorted((at, sec) for sec, at in first_at.items() if at > t)
        return ("title", sec_label[upcoming[0][1]] if upcoming else None), None

    return [(a, b, *state_at((a + b) / 2)) for a, b in zip(pts, pts[1:])]


def make_video(timeline: dict, plan: dict, audio: Path, out: Path, s: Settings, title: str | None = None) -> int:
    """Render the lesson video `out` for the finished `audio`. Returns the number of distinct frames drawn."""
    size = parse_size(s.video_size)
    total = tags(audio)["duration"]
    st = Style(size, s.video_font)
    scenes = Scenes(st, plan, title or plan.get("meta", {}).get("title") or "")
    intervals = build_intervals(timeline, plan, total)
    base: dict = {}
    with tempfile.TemporaryDirectory(prefix="lessong-video-") as tmp:
        tmp_dir = Path(tmp)
        listing = []
        for n, (a, b, state, label) in enumerate(intervals):
            key = (state, label)
            if key not in base:
                base[key] = scenes.render(state, label)
            f = tmp_dir / f"f{n:05d}.png"
            scenes.progress(base[key], (a + b) / 2, total).save(f, compress_level=1)
            listing += [f"file '{f}'", f"duration {b - a:.4f}"]
        listing.append(f"file '{tmp_dir / f'f{len(intervals) - 1:05d}.png'}'")           # the concat demuxer ignores the last duration
        (tmp_dir / "frames.txt").write_text("\n".join(listing))
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp_dir / "frames.txt"),
               "-i", str(audio), "-map", "0:v", "-map", "1:a", "-vf", f"fps={s.video_fps},format=yuv420p", "-c:v", "libx264",
               "-preset", "veryfast", "-tune", "stillimage", "-crf", "24", "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.3f}",
               "-movflags", "+faststart", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"ffmpeg failed while making the video: {r.stderr.strip()[-400:]}")
    log(f"[video] {out} ({len(intervals)} frames from {len(base)} visual states, {size[0]}x{size[1]})")
    return len(base)
