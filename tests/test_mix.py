import numpy as np
import pytest
from conftest import needs_ffmpeg

from lessong.audio import SR, db_to_gain, fade, rms
from lessong.config import Settings
from lessong.mix import Track, Voices, learn_block, smoothstep, song_windows, trim


def mk(times):
    return [{"i": i, "text": f"w{i}", "start": a, "end": b, "sing_start": a, "sing_end": b, "vgap": 1.0, "vgap_mid": b + 0.5, "narrate": True}
            for i, (a, b) in enumerate(times)]


def test_trim_removes_edge_silence_but_keeps_the_sound():
    y = np.concatenate([np.zeros(4096), np.ones(2048), np.zeros(4096)]).astype("float32")
    assert 2048 <= len(trim(y, 35)) <= 3072 and trim(np.zeros(100, "float32"), 35).shape == (100,)


def test_smoothstep_is_monotonic_from_zero_to_one():
    x = smoothstep(1000)
    assert x[0] == 0 and x[-1] == 1 and np.all(np.diff(x) >= 0) and len(smoothstep(0)) == 2


def test_fade_ramps_mono_and_stereo_in_place():
    for shape in ((SR,), (SR, 2)):
        y = np.ones(shape)
        fade(y, 0.1, 0.1)
        assert y[0].max() == 0 and y[-1].max() == 0 and y[SR // 2].min() == 1


def test_song_windows_cut_in_pauses_and_fade_only_after_singing():
    L = mk([(10, 12), (14, 16), (18, 20), (22, 24)])
    L[1].update(sing_end=17.2, vgap=2.0, vgap_mid=17.5)       # held note, then a real pause
    plan = {"lines": L, "sections": [{"lines": [0, 1]}, {"lines": [2, 3]}]}
    w0, w1 = song_windows(plan, Settings(song_preroll=2, song_tail=1.5, song_fade_out=1.5), 100)
    assert w0["end"] == 17.5 and w1["start"] == 17.5
    assert w0["end"] - w0["fade_out"] >= 17.2 and w1["end"] == 25.5 and not w0["cut_in_singing"]


def test_song_windows_flag_a_cut_inside_continuous_singing():
    L = mk([(10, 12), (12.1, 14)])
    L[0].update(vgap=0.05)
    plan = {"lines": L, "sections": [{"lines": [0]}, {"lines": [1]}]}
    assert song_windows(plan, Settings(), 100)[0]["cut_in_singing"]


def test_track_overlaps_pieces_and_grows():
    t = Track()
    t.add(np.ones((SR, 2)), 0)
    t.add(np.ones((SR, 2)), SR // 2)
    t.add(np.ones((10, 2)), 40 * SR)            # beyond the initial buffer
    a = t.audio()
    assert len(a) == 40 * SR + 10 and a[SR // 4, 0] == 1 and a[int(SR * 0.75), 0] == 2


class StubVoices:
    """speak() returns a constant-level tone: lets us test the block layout without any TTS."""
    def __init__(self, s):
        self.s = s

    def speak(self, text, lang):
        t = np.arange(int(0.6 * SR)) / SR
        return np.sin(2 * np.pi * 300 * t) * db_to_gain(self.s.voice_level_db) * 1.4142


def _loop():
    t = np.arange(4 * SR) / SR
    x = 0.3 * np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    return np.stack([x, x], 1)


def test_learn_block_keeps_the_bed_under_the_voices_and_ducks_it():
    s = Settings(lead_in=1.0, learn_tail=1.0)
    block, T0 = learn_block([{"text": "a", "translation": "b"}], StubVoices(s), _loop(), s)
    assert T0 == len(block) and block.shape[1] == 2
    bed_only = block[int(0.2 * SR):int(0.9 * SR)]                                # lead-in: bed alone
    voice = block[int(1.1 * SR):int(1.5 * SR)]                                   # first line over the bed
    assert rms(voice) > 5 * rms(bed_only)


def test_learn_block_with_transition_swells_then_hands_over_on_the_requested_phase():
    s = Settings(lead_in=1.0, learn_tail=0.5, swell_bars=1.0)
    loop, bar = _loop(), len(_loop()) / 2
    trans = dict(bar=bar, xfade=1.0, target_rms=0.2, phase=0.7)
    block, T0 = learn_block([{"text": "a", "translation": "b"}], StubVoices(s), loop, s, trans)
    assert (T0 - 0.7 * SR) % bar == pytest.approx(0, abs=2) or (T0 - 0.7 * SR) % bar == pytest.approx(bar, abs=2)   # entry beat == loop phase at T0
    sw = int(bar)
    a, b = rms(block[T0 - sw - SR // 4:T0 - sw]), rms(block[T0 - SR // 4:T0])
    assert b > 3 * a and b == pytest.approx(0.2, rel=0.5)                       # volume grew to the song's level
    assert rms(block[-SR // 20:]) < 0.1 * b                                     # and the bed is gone by the end of the crossfade


def test_learn_block_after_a_song_starts_at_the_song_level_and_settles_before_the_first_voice():
    s = Settings(settle_bars=1.0)
    loop, bar = _loop(), len(_loop()) / 2
    intro = dict(bar=bar, xfade=0.5, phase=0.3, target_rms=0.25)
    block, _ = learn_block([{"text": "a", "translation": "b"}], StubVoices(s), loop, s, None, intro)
    start = rms(block[int(0.3 * SR):int(0.5 * SR)])
    settled = rms(block[int((0.5 + bar / SR + 0.1) * SR):int((0.5 + bar / SR + 0.25) * SR)])
    assert start == pytest.approx(0.25, rel=0.6) and settled < 0.3 * start


@needs_ffmpeg
def test_voices_cache_tts_so_a_second_render_costs_nothing(tmp_path, fake_eleven):
    s = Settings()
    v = Voices(fake_eleven, s, tmp_path / "tts", "src", "dst")
    a = v.speak("hello there", "en")
    n = len(fake_eleven.tts_calls)
    b = v.speak("hello there", "en")
    assert len(fake_eleven.tts_calls) == n == 1 and np.allclose(a, b) and v.chars == len("hello there")
    assert rms(a) == pytest.approx(db_to_gain(s.voice_level_db), rel=0.01)
    v.speak("hello there", "fr")                      # a different voice is a different cache entry
    assert len(fake_eleven.tts_calls) == 2
