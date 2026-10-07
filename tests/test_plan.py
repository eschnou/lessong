import json

import pytest

from lessong import plan as plan_mod
from lessong.config import Settings
from lessong.plan import drop_repeats, make_sections, mark_narration, structured_sections, summarize


def mk(times, **extra):
    return [{"i": i, "text": f"line {i}", "start": a, "end": b, "vgap": 3.0, **extra} for i, (a, b) in enumerate(times)]


def test_every_line_is_narrated():
    ls = [{"text": "a"}, {"text": "a"}, {"text": "b"}]
    mark_narration(ls)
    assert all(l["narrate"] for l in ls)


def test_gap_mode_breaks_on_long_silence_caps_size_and_covers_everything():
    s = Settings(section_gap=6, min_lines=2, max_lines=4)
    lines = mk([(i * 3, i * 3 + 2) for i in range(6)] + [(60 + i * 3, 62 + i * 3) for i in range(3)])
    for l in lines:
        l["vgap"] = 1.0
    lines[5]["vgap"] = 50.0
    mark_narration(lines)
    secs = make_sections(lines, s)
    assert secs[-1] == [6, 7, 8] and all(len(x) <= 4 for x in secs) and sum(len(x) for x in secs) == 9


def _structure(monkeypatch, segs):
    monkeypatch.setattr(plan_mod, "llm_structure", lambda lines, s: segs)


def test_structured_sections_keep_each_section_whole_and_teach_every_line(monkeypatch):
    lines = mk([(0, 2), (3, 5), (10, 12), (13, 15), (20, 22), (23, 25)])
    mark_narration(lines)
    _structure(monkeypatch, [dict(label="verse", first=0, last=1), dict(label="chorus", first=2, last=3), dict(label="verse again", first=4, last=5)])
    out = structured_sections(lines, Settings(max_lines=8))
    assert [o["lines"] for o in out] == [[0, 1], [2, 3], [4, 5]] and all(lines[i]["narrate"] for o in out for i in o["lines"])


def test_repeats_skip_removes_a_section_whose_text_already_appeared():
    lines = mk([(0, 2), (3, 5), (10, 12), (20, 22)])
    for l, t in zip(lines, "a b c a".split()):
        l["text"] = t
    secs = [{"label": "one", "lines": [0, 1]}, {"label": "two", "lines": [2]}, {"label": "one again", "lines": [3]}]
    assert [s["label"] for s in drop_repeats(lines, secs)] == ["one", "two"]


def test_sections_sung_without_a_pause_are_joined(monkeypatch):
    lines = mk([(0, 2), (2.1, 4), (10, 12)])
    for l, v in zip(lines, (0.0, 3.0, 3.0)):
        l["vgap"] = v
    mark_narration(lines)
    _structure(monkeypatch, [dict(label="a", first=0, last=0), dict(label="b", first=1, last=1), dict(label="c", first=2, last=2)])
    assert [o["lines"] for o in structured_sections(lines, Settings())] == [[0, 1], [2]]


def test_an_oversized_join_is_cut_at_the_best_real_pause(monkeypatch):
    lines = mk([(i * 3, i * 3 + 2) for i in range(20)])
    for l in lines:
        l["vgap"] = 0.0
    lines[9]["vgap"] = 0.6
    mark_narration(lines)
    _structure(monkeypatch, [dict(label="a", first=0, last=9), dict(label="b", first=10, last=19)])
    out = structured_sections(lines, Settings(max_lines=8))
    assert [o["lines"][-1] for o in out][:1] == [9] and sum(len(o["lines"]) for o in out) == 20


def test_isolated_ad_lib_is_dropped(monkeypatch):
    lines = mk([(0, 2), (3, 5), (30, 30.3), (50, 52)])
    lines[2]["text"] = "hey"
    mark_narration(lines)
    _structure(monkeypatch, [dict(label="v", first=0, last=1), dict(label="adlib", first=2, last=2), dict(label="v2", first=3, last=3)])
    out = structured_sections(lines, Settings())
    assert [o["lines"] for o in out] == [[0, 1], [3]]


def test_llm_failure_falls_back_to_pause_based_sections(monkeypatch):
    lines = mk([(i * 3, i * 3 + 2) for i in range(6)])
    mark_narration(lines)
    _structure(monkeypatch, None)
    out = structured_sections(lines, Settings(min_lines=2, max_lines=8))
    assert sum(len(o["lines"]) for o in out) == 6


class FakeOpenAI:
    """Minimal stand-in for openai.OpenAI().chat.completions.create"""
    answers: list = []

    def __init__(self, *a, **k):
        outer = self
        self.chat = type("C", (), {"completions": type("X", (), {"create": staticmethod(outer._create)})()})()

    def _create(self, **kw):
        msg = type("M", (), {"content": FakeOpenAI.answers.pop(0)})()
        return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()


@pytest.fixture
def fake_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    return FakeOpenAI


def test_translate_retries_until_every_line_is_answered(fake_openai):
    lines = [{"i": 0, "text": "one"}, {"i": 1, "text": "two"}]
    fake_openai.answers = [json.dumps({"lines": [{"i": 0, "translation": "un"}]}),
                           json.dumps({"lines": [{"i": 0, "translation": "un"}, {"i": 1, "translation": "deux"}]})]
    assert plan_mod.translate(lines, "en", "fr", "m") == ["un", "deux"]


def test_translate_gives_up_after_three_bad_answers(fake_openai):
    fake_openai.answers = ["not json", "{}", json.dumps({"lines": []})]
    with pytest.raises(RuntimeError):
        plan_mod.translate([{"i": 0, "text": "one"}], "en", "fr", "m")


def test_llm_structure_rejects_answers_that_do_not_cover_the_lyrics(fake_openai):
    lines = mk([(0, 1), (2, 3), (4, 5)])
    bad = json.dumps({"sections": [{"label": "a", "first": 0, "last": 0}, {"label": "b", "first": 2, "last": 2}]})
    good = json.dumps({"sections": [{"label": "b", "first": 2, "last": 2}, {"label": "a", "first": 0, "last": 1}]})   # unordered but valid
    fake_openai.answers = [bad, bad, bad]
    assert plan_mod.llm_structure(lines, Settings()) is None
    fake_openai.answers = [good]
    assert [(g["first"], g["last"]) for g in plan_mod.llm_structure(lines, Settings())] == [(0, 1), (2, 2)]


def test_make_plan_reuses_known_translations_and_gives_identical_lines_the_same_one():
    lines = mk([(0, 1), (2, 3), (4, 5)])
    for l in lines:
        l["text"] = "same words"
    calls = []
    s = Settings(section_mode="gap")
    p = plan_mod.make_plan(lines, s, {}, translate_fn=lambda *a: calls.append(a) or ["x", "y", "z"])
    assert len(calls) == 1 and {l["translation"] for l in p["lines"]} == {"x"}
    lines2 = mk([(0, 1)])
    lines2[0]["text"] = "same words"
    p2 = plan_mod.make_plan(lines2, s, {}, translate_fn=lambda *a: pytest.fail("must not call the LLM"), known={"samewords": "kept"})
    assert p2["lines"][0]["translation"] == "kept"


def test_summarize_lists_every_section_and_the_tts_cost():
    lines = mk([(0, 1), (2, 3)])
    for l in lines:
        l.update(narrate=True, translation="xx")
    txt = summarize({"lines": lines, "sections": [{"label": "verse", "lines": [0, 1]}]})
    assert "section 1" in txt and "verse" in txt and "characters" in txt
