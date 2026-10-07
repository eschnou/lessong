import json
import subprocess

import pytest
import soundfile as sf
from conftest import needs_ffmpeg

from lessong import cli
from lessong import plan as plan_mod


def test_version_and_help_exit_cleanly(capsys):
    for flag in ("--version", "--help"):
        with pytest.raises(SystemExit) as e:
            cli.main([flag])
        assert e.value.code == 0
    assert "lessong" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("cmd", ["build", "plan", "render", "loops", "doctor"])
def test_every_subcommand_has_help(cmd, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([cmd, "--help"])
    assert e.value.code == 0 and capsys.readouterr().out


def test_unknown_command_is_an_error():
    with pytest.raises(SystemExit) as e:
        cli.main(["frobnicate"])
    assert e.value.code == 2


def test_render_without_a_plan_explains_what_to_do(workspace):
    with pytest.raises(SystemExit, match="run `lessong plan`"):
        cli.main(["render", str(workspace), "-o", str(workspace / "x.mp3")])


def test_doctor_reports_missing_keys_and_fails(monkeypatch, capsys):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "ELEVENLABS_API_KEY" in out and "OPENAI_API_KEY" in out and "!!" in out


def test_missing_openai_key_is_reported_before_any_work(workspace, monkeypatch, fake_eleven):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "x")
    monkeypatch.setattr(cli, "ElevenLabs", lambda key: fake_eleven)
    (workspace / "lyr.txt").write_text("alpha beta gamma delta epsilon")
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        cli.main(["plan", str(workspace), "--lyrics-file", str(workspace / "lyr.txt")])


def fake_translate(lines, src, dst, model):
    return [f"[{dst}] {l['text']}" for l in lines]


@pytest.fixture
def wired(monkeypatch, fake_eleven):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "x")
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(cli, "ElevenLabs", lambda key: fake_eleven)
    monkeypatch.setattr(plan_mod, "translate", fake_translate)
    return fake_eleven


@needs_ffmpeg
@pytest.mark.slow
def test_build_end_to_end_on_a_synthetic_song(workspace, wired, synth, tmp_path):
    out = tmp_path / "lesson.mp3"
    (workspace / "lyrics.txt").write_text(synth["lyrics"])
    rc = cli.main(["build", str(workspace), "--lyrics-file", str(workspace / "lyrics.txt"), "--sections", "gap", "--min-lines", "2",
                   "--max-lines", "4", "--section-gap", "3", "--loop-bars", "4", "--to", "xx", "-o", str(out)])
    assert rc == 0 and out.exists()
    # plan: 8 lines, 2 sections (4 + 4) split at the long instrumental break, translations present
    plan = json.loads((workspace / "plan.json").read_text())
    assert len(plan["lines"]) == 8 and [len(s["lines"]) for s in plan["sections"]] == [4, 4]
    assert all(l["translation"].startswith("[xx]") for l in plan["lines"]) and all(l["narrate"] for l in plan["lines"])
    # audio: longer than the narration alone, no clipping, stereo, hand-overs recorded and aligned
    info = sf.info(str(workspace / "mix.wav"))
    assert info.channels == 2 and 60 < info.duration < 200
    y, _ = sf.read(workspace / "mix.wav")
    assert abs(y).max() < 1.5 and abs(y).max() > 0.1
    rep = json.loads((workspace / "transitions.json").read_text())
    assert len(rep) == 2 and rep[0]["entry"] is not None and rep[0]["entry"] < 24.0       # the song comes in during the intro
    assert rep[0]["exit"] is not None or rep[1]["entry"] is not None
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                               capture_output=True, text=True).stdout)
    assert dur == pytest.approx(info.duration, abs=1.0)
    n_tts = len(wired.tts_calls)
    assert n_tts == 16 and wired.scribe_calls == 1                                         # 8 lines x (song language + translation)


@needs_ffmpeg
@pytest.mark.slow
def test_rerender_with_other_mix_settings_reuses_everything_cached(workspace, wired, synth, tmp_path):
    (workspace / "lyrics.txt").write_text(synth["lyrics"])
    plan_args = ["--sections", "gap", "--min-lines", "2", "--max-lines", "4", "--section-gap", "3"]
    args = ["--loop-bars", "4", "--to", "xx"]
    cli.main(["build", str(workspace), "--lyrics-file", str(workspace / "lyrics.txt"), *plan_args, *args, "-o", str(tmp_path / "a.mp3")])
    calls = len(wired.tts_calls)
    loop_json = (workspace / "loop.json").read_text()
    cli.main(["render", str(workspace), *args, "--bed-db", "-25", "--voice-gender", "female", "-o", str(tmp_path / "b.mp3")])
    assert len(wired.tts_calls) == calls and (workspace / "loop.json").read_text() == loop_json and (tmp_path / "b.mp3").exists()


@needs_ffmpeg
@pytest.mark.slow
def test_replan_keeps_translations_and_can_drop_repeated_sections(workspace, wired, synth, monkeypatch):
    (workspace / "lyrics.txt").write_text(synth["lyrics"])
    base = ["plan", str(workspace), "--lyrics-file", str(workspace / "lyrics.txt"), "--sections", "gap", "--min-lines", "2", "--to", "xx"]
    cli.main(base)
    p = json.loads((workspace / "plan.json").read_text())
    p["lines"][0]["translation"] = "edited by hand"
    (workspace / "plan.json").write_text(json.dumps(p))
    monkeypatch.setattr(plan_mod, "translate", lambda *a: pytest.fail("a replan must not re-translate"))
    cli.main([*base, "--replan"])
    assert json.loads((workspace / "plan.json").read_text())["lines"][0]["translation"] == "edited by hand"
    assert wired.scribe_calls == 1                                                         # Scribe is not re-run on a replan
