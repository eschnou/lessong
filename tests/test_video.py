import subprocess

import pytest
from conftest import needs_ffmpeg
from PIL import Image

from lessong.config import Settings
from lessong.video import Scenes, Style, build_intervals, fit, make_video, parse_size, wrap

TOTAL = 24.0


def plan():
    lines = [{"i": i, "text": t, "translation": f"{t} (tr)", "sing_start": 100.5 + 2 * i, "sing_end": 102 + 2 * i, "start": 100.5 + 2 * i, "end": 102 + 2 * i}
             for i, t in enumerate(["alpha beta gamma", "delta epsilon zeta"])]
    return {"lines": lines, "source_lang": "xx", "target_lang": "en", "meta": {"title": "Test Song"}, "sections": [{"label": "verse 1", "lines": [0, 1]}]}


def timeline():
    ev = [dict(kind="intro", text="Welcome to the test", t0=1.0, t1=3.0),
          dict(kind="src", section=1, line=0, t0=5.0, t1=6.0), dict(kind="dst", section=1, line=0, t0=6.4, t1=7.4),
          dict(kind="src", section=1, line=1, t0=8.0, t1=9.0), dict(kind="dst", section=1, line=1, t0=9.4, t1=10.4),
          dict(kind="song", section=1, t0=12.0, t1=20.0, src0=100.0),
          dict(kind="outro", text="See you soon", t0=22.0, t1=23.5)]
    return dict(events=ev, sections=[dict(section=1, label="verse 1", lines=[0, 1], t0=0.0, t1=20.0)])


def state_at(t):
    for a, b, state, label in build_intervals(timeline(), plan(), TOTAL):
        if a <= t < b:
            return state, label
    raise AssertionError(t)


def test_intervals_tile_the_whole_video_without_gaps():
    iv = build_intervals(timeline(), plan(), TOTAL)
    assert iv[0][0] == 0.0 and iv[-1][1] == TOTAL
    assert all(abs(a2 - b1) < 1e-9 for (_, b1, _, _), (a2, _, _, _) in zip(iv, iv[1:]))
    assert all(b > a for a, b, _, _ in iv)


def test_what_is_on_screen_follows_the_audio():
    assert state_at(0.5)[0] == ("title", "verse 1")                     # before anything: the title card, announcing the first section
    assert state_at(2.0)[0] == ("text", "Welcome to the test")          # the spoken intro
    assert state_at(4.0)[0] == ("title", "verse 1")                     # music only, still before the first line
    assert state_at(5.5)[0] == ("lesson", 0, False, 1, 2)               # first line being said, translation not yet
    assert state_at(7.0)[0] == ("lesson", 0, True, 1, 2)                # its translation is being said
    assert state_at(8.5)[0] == ("lesson", 1, False, 2, 2)               # next line: translation hidden again
    assert state_at(11.0)[0] == ("lesson", 1, True, 2, 2)               # the last line stays up until the song starts
    assert state_at(12.2)[0] == ("song", (0, 1), None)                  # the song is in, nobody is singing yet
    assert state_at(13.0)[0] == ("song", (0, 1), 0)                     # first line sung (source 100.5 -> output 12.5)
    assert state_at(15.0)[0] == ("song", (0, 1), 1)
    assert state_at(18.0)[0] == ("song", (0, 1), 1)                     # after the last line the highlight stays
    assert state_at(21.0)[0] == ("title", None)                         # between the song and the outro
    assert state_at(22.5)[0] == ("text", "See you soon")
    assert state_at(5.5)[1] == "verse 1" and state_at(15.0)[1] == "verse 1"


def test_wrap_and_fit_keep_text_inside_the_box():
    st = Style((1280, 720))
    f = st.font(60, True)
    lines = wrap("one two three four five six seven eight nine ten eleven twelve", f, 600)
    assert len(lines) > 1 and all(f.getlength(x) <= 600 for x in lines)
    cjk = wrap("これは非常に長い日本語の文章でスペースがありません" * 3, f, 600)           # no spaces to break on
    assert len(cjk) > 1 and all(f.getlength(x) <= 600 + f.size for x in cjk)
    big, lines_big, size_big = fit(st, "short", True, 1000, 200, 60)
    small, lines_small, size_small = fit(st, "a much longer sentence " * 12, True, 1000, 200, 60)
    assert size_big == 60 and size_small < 60 and len(lines_small) * size_small * 1.3 <= 200 + size_small


@pytest.mark.parametrize("bad", ["abc", "1280", "1281x720", "100x50", "0x0"])
def test_video_size_is_validated(bad):
    with pytest.raises(SystemExit, match="video-size"):
        parse_size(bad)


def test_every_scene_renders_at_the_requested_size_and_looks_different():
    for size in ((640, 360), (1920, 1080)):
        sc = Scenes(Style(size), plan(), "Test Song")
        imgs = {k: sc.render(state, "verse 1") for k, state in {
            "title": ("title", "verse 1"), "text": ("text", "Hello there"), "lesson": ("lesson", 0, False, 1, 2),
            "lesson_tr": ("lesson", 0, True, 1, 2), "song": ("song", (0, 1), 1), "song_idle": ("song", (0, 1), None)}.items()}
        assert all(isinstance(i, Image.Image) and i.size == size for i in imgs.values())
        assert imgs["lesson"].tobytes() != imgs["lesson_tr"].tobytes() and imgs["song"].tobytes() != imgs["song_idle"].tobytes()
        assert len({i.tobytes() for i in imgs.values()}) == len(imgs)
        assert len(imgs["lesson_tr"].getcolors(1 << 20)) > 3                      # not a blank canvas
    p = sc.progress(imgs["title"], 12.0, 24.0)
    assert p.tobytes() != imgs["title"].tobytes() and p.size == imgs["title"].size


@needs_ffmpeg
def test_make_video_produces_an_mp4_with_the_audio_and_the_right_length(tmp_path):
    audio = tmp_path / "a.mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={TOTAL}", "-b:a", "128k", str(audio)], check=True)
    out = tmp_path / "v.mp4"
    n = make_video(timeline(), plan(), audio, out, Settings(video_size="640x360", video_fps=5))
    assert out.exists() and n >= 8                                                     # title, text, lessons, song, outro...
    info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,duration", "-of", "csv=p=0", str(out)],
                          capture_output=True, text=True).stdout.split()
    assert any(x.startswith("video,640,360") for x in info) and any(x.startswith("audio") for x in info)
    assert all(abs(float(x.split(",")[-1]) - TOTAL) < 0.5 for x in info)
    frame = lambda t: subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "bmp", "-"],
                                     capture_output=True).stdout
    assert frame(2.0) != frame(7.0) != frame(15.0)                                    # the picture really changes with the audio
