
import numpy as np
import pytest

from lessong import lyrics
from lessong.lyrics import (
    HOP,
    align,
    annotate_vocals,
    build_lines,
    fill_gaps,
    from_transcript,
    norm,
    only_sung,
    parse_lyrics,
    vocal_threshold,
)


def W(text, start, dur=0.3):
    return {"type": "word", "text": text, "start": start, "end": start + dur}


def test_norm_keeps_apostrophes_and_accents():
    assert norm("Don’t,") == "don't" and norm("Été!") == "été"


def test_parse_lyrics_strips_timestamps_headers_and_blank_lines():
    txt = "[ar:Someone]\n[00:12.50] first line here\n\n[Chorus]\n[01:02.00] second line here\n(third line)"
    assert parse_lyrics(txt) == ["first line here", "second line here", "(third line)"]


def test_align_times_lines_and_ignores_extra_transcript_words():
    words = [W(t, i * 0.5) for i, t in enumerate("uh oh red green blue yellow purple".split())]
    out, matched, total = align(["red green blue", "yellow purple"], words)
    assert matched == total == 5 and out[0]["start"] == 1.0 and out[1]["start"] == 2.5 and all(l["timed"] for l in out)


def test_align_repeated_lines_follow_song_order():
    words = [W(t, i) for i, t in enumerate("la la la one two la la la three four la la la".split())]
    out, _, _ = align(["la la la", "one two", "la la la", "three four", "la la la"], words)
    starts = [l["start"] for l in out]
    assert starts == sorted(starts) and starts[0] == 0 and starts[2] == 5 and starts[4] == 10


def test_align_stray_word_far_away_does_not_stretch_a_line():
    words = [W("hello", 1.0)] + [W(t, 17 + i * 0.4) for i, t in enumerate("hello there my friend".split())]
    out, _, _ = align(["Hello there my friend"], words)
    assert out[0]["start"] >= 17 and out[0]["end"] < 19


def test_only_sung_drops_words_on_silence():
    act = np.zeros(int(30 / HOP), bool)
    act[int(16.9 / HOP):int(19.5 / HOP)] = True
    kept = only_sung([W("ghost", 1.0), W("real", 17.0)], act)
    assert [w["text"] for w in kept] == ["real"]


def test_fill_gaps_interpolates_untimed_lines():
    ls = [{"start": 1, "end": 2}, {"start": None, "end": None}, {"start": 6, "end": 7}]
    fill_gaps(ls)
    assert 2 < ls[1]["start"] < ls[1]["end"] < 6


def test_from_transcript_splits_on_pauses_and_sentence_ends():
    words = [W("a", 0), W("b", 0.4), W("c", 3.0), W("d.", 3.4), W("e", 3.8)]
    lines = from_transcript(words)
    assert [l["text"] for l in lines] == ["a b", "c d.", "e"]


@pytest.mark.parametrize("silence_floor", [-180.0, -90.0, -45.0])
def test_vocal_threshold_sits_between_silence_and_singing_whatever_the_floor(silence_floor):
    rng = np.random.default_rng(0)
    level = np.concatenate([np.full(500, silence_floor) + rng.normal(0, 1, 500), -30 + rng.normal(0, 2, 1500)])
    t = vocal_threshold(level)
    assert max(silence_floor, -80) < t < -30 - 5


def test_annotate_vocals_finds_real_silence_and_word_gaps():
    act = np.zeros(int(40 / HOP), bool)
    for a, b in ((1.0, 4.0), (5.0, 8.0), (8.1, 11.0)):       # line 1 | 1.0 s silence | line 2 | tiny 0.1 s dip | line 3
        act[int(a / HOP):int(b / HOP)] = True
    lines = [{"start": 1.0, "end": 3.8}, {"start": 5.0, "end": 7.9}, {"start": 8.15, "end": 10.9}]
    annotate_vocals(lines, act)
    assert lines[0]["vgap"] >= 0.9 and 4.0 < lines[0]["vgap_mid"] < 5.0 and lines[0]["sing_end"] < 4.3
    assert lines[1]["vgap"] < 0.2 and 7.9 <= lines[1]["vgap_mid"] <= 8.15          # no real pause: the cut falls in the word gap
    assert all(l["sing_start"] <= l["start"] and l["sing_end"] >= l["end"] for l in lines)


def test_fetch_candidates_caches_the_response(monkeypatch, tmp_path):
    calls = []

    class R:
        status_code = 200
        def json(self):
            return [{"syncedLyrics": "[00:01.00] one two\n[00:03.00] three four", "instrumental": False},
                    {"plainLyrics": "one two\nthree four"}, {"instrumental": True, "plainLyrics": "x"}]

    monkeypatch.setattr(lyrics.requests, "get", lambda *a, **k: calls.append(1) or R())
    cache = tmp_path / "lrclib.json"
    first = lyrics.fetch_candidates("Title", "Artist", cache)
    again = lyrics.fetch_candidates("Title", "Artist", cache)
    assert first == again == [["one two", "three four"]] and len(calls) == 1      # duplicate text merged, instrumental dropped, cached


def test_fetch_candidates_survives_an_unreachable_service(monkeypatch):
    monkeypatch.setattr(lyrics.requests, "get", lambda *a, **k: (_ for _ in ()).throw(lyrics.requests.ConnectionError()))
    monkeypatch.setattr(lyrics.time, "sleep", lambda s: None)
    assert lyrics.fetch_candidates("T", None) == []


def test_build_lines_prefers_the_provided_lyrics_and_falls_back_to_the_transcript(synth):
    words = synth["words"]
    lines, src = build_lines(None, None, synth["lyrics"], False, words)
    assert src == "lyrics file" and len(lines) == 8 and lines[0]["start"] == pytest.approx(24.0, abs=0.01)
    lines, src = build_lines("T", None, None, True, words)          # no lookup, no file: use what was heard
    assert src == "transcript" and len(lines) >= 8


def test_poor_lyric_match_falls_back_to_the_transcript(synth):
    lines, src = build_lines(None, None, "completely unrelated text\nnothing in common at all", False, synth["words"])
    assert src == "transcript"
