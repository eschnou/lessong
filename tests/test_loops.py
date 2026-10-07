import numpy as np
import pytest
import soundfile as sf

from lessong.audio import SR
from lessong.config import Settings
from lessong.loops import Loop, analyse, candidates, pick, render, seam_ratio

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def analysis(synth):
    d = synth["dir"]
    return analyse(d / "stems" / "no_vocals.wav", d / "stems" / "vocals.wav", 4)


def test_beat_tracking_finds_the_tempo(analysis):
    assert analysis["bar"] == pytest.approx(2.0, abs=0.05)


def test_the_loop_is_cut_from_the_voice_free_intro_and_is_seamless(analysis):
    loop = pick(analysis, Settings())
    assert isinstance(loop, Loop) and loop.bars == 4 and loop.length == pytest.approx(8.0, abs=0.1)
    assert 0 <= loop.t and loop.t + loop.length + 0.5 < 24.0          # entirely before the first sung line
    assert loop.seam < 1.5 and loop.sim > 0.5


def test_a_manual_loop_start_is_honoured(analysis):
    loop = pick(analysis, Settings(loop_start=2.0))
    assert loop.t == 2.0 and loop.note == "manual start"


def test_falls_back_to_a_shorter_loop_when_the_song_has_no_room(analysis):
    loop = pick(analysis, Settings(loop_bars=32))                     # 64 s of loop in a 24 s intro: impossible
    assert loop.bars < 32 and loop.t + loop.length < 24.5


def test_candidates_are_ranked_and_pass_the_voice_filter(analysis):
    cs, note = candidates(analysis, 2, Settings())
    assert cs and all(c.t + c.length < 24.5 for c in cs[:5]) and cs[0].sim >= cs[-1].sim - 0.5


def test_render_is_a_cycle_whose_end_continues_into_its_start(synth, analysis):
    nov, _ = sf.read(synth["dir"] / "stems" / "no_vocals.wav")
    loop = render(nov, 4.0, 8.0, 0.4)
    assert loop.shape == (8 * SR, 2)
    jump = np.abs(loop[0] - nov[int(12.0 * SR)]).max()                # first sample of the loop == the music right after its end
    assert jump < 0.3
    assert seam_ratio(nov, loop, 4.0, 8.0) < 1.5
