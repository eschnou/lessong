import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lessong import cli
from lessong import pipeline as pl
from lessong import plan as plan_mod


def test_scribe_language_codes_map_to_two_letter_codes():
    assert [pl.iso1(x) for x in ("fra", "nld", "eng", "ita", "zho", "fr", "IT")] == ["fr", "nl", "en", "it", "zh", "fr", "it"]
    assert pl.iso1("xyz") is None and pl.iso1(None) is None and pl.iso1("") is None


def test_default_target_is_english_unless_the_song_is_english():
    assert [pl.default_target(x) for x in ("fr", "nl", "ja", "en")] == ["en", "en", "en", "fr"]


@pytest.mark.parametrize("frm,to,expected", [(None, None, False), ("fr", None, False), (None, "en", False), ("fr", "en", False),
                                             (None, "it", True), ("en", None, True), ("en", "it", True)])
def test_only_languages_that_were_asked_for_can_conflict_with_a_plan(frm, to, expected):
    assert pl.conflicts(("fr", "en"), frm, to) is expected


def seed(tmp_path: Path, synth: dict, name="song", pair=("fr", "en")) -> Path:
    """A project folder with a song file and a work folder that already holds separated stems and a plan for `pair`."""
    (tmp_path / "proj").mkdir()
    base = tmp_path / "proj" / ".lessong" / name
    shutil.copytree(synth["dir"], base)
    (base / "plan.json").write_text(json.dumps({"source_lang": pair[0], "target_lang": pair[1], "lines": [], "sections": [], "meta": {}}))
    (base / "meta.json").write_text(json.dumps({"title": "Song", "artist": None}))
    (tmp_path / "proj" / f"{name}.mp3").write_bytes(b"not really audio: the stems are already separated")
    return tmp_path / "proj"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_a_new_target_language_gets_its_own_folder_and_leaves_the_old_plan_alone(tmp_path, synth, monkeypatch):
    proj = seed(tmp_path, synth)
    monkeypatch.chdir(proj)
    base_plan = proj / ".lessong" / "song" / "plan.json"
    before = sha(base_plan)
    ws = pl.choose_workspace(Path("song.mp3"), None, None, "it")
    assert ws.dir == Path(".lessong") / "song-fr-it" and ws.dir != base_plan.parent
    assert (ws.dir / "stems" / "vocals.wav").exists() and (ws.dir / "source.wav").exists() and (ws.dir / "meta.json").exists()   # stems copied
    assert not (ws.dir / "plan.json").exists() and sha(base_plan) == before                                                        # nothing overwritten


@pytest.mark.parametrize("frm,to", [(None, None), (None, "en"), ("fr", "en"), ("fr", None)])
def test_the_same_or_unspecified_languages_keep_using_the_existing_folder(tmp_path, synth, monkeypatch, frm, to):
    monkeypatch.chdir(seed(tmp_path, synth))
    assert pl.choose_workspace(Path("song.mp3"), None, frm, to).dir == Path(".lessong") / "song"


def test_an_explicit_workdir_or_a_folder_input_is_never_redirected(tmp_path, synth, monkeypatch):
    proj = seed(tmp_path, synth)
    monkeypatch.chdir(proj)
    explicit = Path(".lessong") / "song"
    assert pl.choose_workspace(Path("song.mp3"), explicit, None, "it").dir == explicit
    assert pl.choose_workspace(explicit, None, None, "it").dir == explicit


def test_a_conflict_in_an_explicit_folder_is_reported_with_the_ways_out(tmp_path, synth, monkeypatch):
    monkeypatch.chdir(seed(tmp_path, synth))
    from lessong.config import Settings
    ws = pl.Workspace(Path(".lessong") / "song", None)
    with pytest.raises(SystemExit) as e:
        pl.resolve_languages(ws, Settings(), None, "it", replanning=False)
    msg = str(e.value)
    assert "fr -> en" in msg and "fr -> it" in msg and "--replan" in msg and "--workdir" in msg


def test_resolve_languages_adopts_the_plan_or_leaves_detection_to_the_transcription(tmp_path, synth, monkeypatch):
    from lessong.config import Settings
    monkeypatch.chdir(seed(tmp_path, synth))
    ws = pl.Workspace(Path(".lessong") / "song", None)
    s = Settings()
    pl.resolve_languages(ws, s, None, None, replanning=False)
    assert (s.source_lang, s.target_lang) == ("fr", "en")                              # the plan decides
    s = Settings()
    pl.resolve_languages(ws, s, "es", "it", replanning=True)                           # replanning: what was asked for
    assert (s.source_lang, s.target_lang) == ("es", "it")
    s = Settings()
    pl.resolve_languages(ws, s, "ja", None, replanning=True)
    assert (s.source_lang, s.target_lang) == ("ja", "en")                              # target derived from the given source
    s = Settings()
    pl.resolve_languages(pl.Workspace(Path(".lessong") / "empty", None), s, None, None, replanning=False)
    assert (s.source_lang, s.target_lang) == (None, None)                              # nothing known yet: detect later


@pytest.fixture
def project(tmp_path, synth, monkeypatch, wired):
    """A project with separated stems but no plan yet, and the API clients faked."""
    proj = tmp_path / "proj"
    proj.mkdir()
    shutil.copytree(synth["dir"], proj / ".lessong" / "song")
    (proj / "song.mp3").write_bytes(b"x")
    monkeypatch.chdir(proj)
    return proj


def run_plan(*extra):
    return cli.main(["plan", "song.mp3", "--no-lookup", "--sections", "gap", "--min-lines", "2", *extra])


def read_plan(folder: Path) -> dict:
    return json.loads((folder / "plan.json").read_text())


def test_the_song_language_is_detected_and_the_target_defaults_sensibly(project):
    assert run_plan() == 0
    p = read_plan(project / ".lessong" / "song")
    assert (p["source_lang"], p["target_lang"]) == ("en", "fr")                         # detected English -> French (the default pairing)


def test_asking_for_another_language_after_a_first_run_makes_a_second_lesson_without_touching_the_first(project):
    run_plan("--to", "es")
    base = project / ".lessong" / "song"
    before = sha(base / "plan.json")
    run_plan("--to", "it")
    other = project / ".lessong" / "song-en-it"
    assert read_plan(other)["target_lang"] == "it" and read_plan(base)["target_lang"] == "es" and sha(base / "plan.json") == before
    assert all(l["translation"].startswith("[it]") for l in read_plan(other)["lines"])          # really translated into the new language
    run_plan()                                                                                  # no language asked for: back to the first one
    assert sha(base / "plan.json") == before and not (project / ".lessong" / "song-en-es").exists()
    run_plan("--to", "it")                                                                      # asking again reuses the second folder
    assert {p.name for p in (project / ".lessong").iterdir()} == {"song", "song-en-it"}          # exactly two folders (a set: directory order is arbitrary)


def test_the_old_behaviour_that_silently_ignored_the_language_is_gone(project, capsys):
    run_plan("--to", "es")
    run_plan("--to", "it")
    assert "using .lessong/song-en-it" in capsys.readouterr().err.replace("\\", "/")


def test_translation_prompt_still_names_the_languages(monkeypatch):
    assert plan_mod.lang_name("it") == "Italian" and plan_mod.lang_name("nl") == "Dutch"
