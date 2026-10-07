import numpy as np

from lessong.transition import find_entry


def features(n_beats=200, bpm=120, bar_pattern=(1.0, 0.3, 0.6, 0.3)):
    """Perfectly periodic synthetic features: a 4-beat accent pattern and a different 'timbre' per beat position."""
    beat = 60 / bpm
    bt = np.arange(n_beats) * beat
    efps, fr = 344.5, 43.07
    env = np.zeros(int(n_beats * beat * efps) + 800)
    for j, t in enumerate(bt):
        env[int(t * efps):int(t * efps) + 6] = bar_pattern[j % 4]
    S = np.zeros((16, int(n_beats * beat * fr) + 40))
    for j, t in enumerate(bt):
        S[(j % 4) * 4:(j % 4) * 4 + 4, int(t * fr):int((t + beat) * fr)] = 1.0
    return dict(bt=bt, env=env, S=S, fr=fr, efps=efps), beat


def test_entry_lands_on_a_beat_and_keeps_the_bar_position():
    feat, beat = features()
    e = find_entry(feat, loop_t=beat * 4, beats_per_bar=4, lo=20.0, hi=30.0, X=1.0)
    assert e and 20.0 <= e["t"] and e["t"] + 1.0 <= 30.0
    j = round(e["t"] / beat)
    assert abs(e["t"] - j * beat) < 0.02                    # on the beat grid
    assert e["onset_corr"] > 0.9 and abs(e["phase"] - (e["q"] * beat)) < 0.02


def test_every_beat_position_is_reachable_in_a_narrow_window():
    feat, beat = features()
    seen = {find_entry(feat, beat * 4, 4, lo=t0, hi=t0 + 0.75, X=0.25)["q"] for t0 in np.arange(20.0, 22.0, 0.25) if find_entry(feat, beat * 4, 4, t0, t0 + 0.75, 0.25)}
    assert seen == {0, 1, 2, 3}


def test_a_few_milliseconds_of_beat_tracker_error_are_corrected():
    feat, beat = features()
    feat["bt"] = feat["bt"] + 0.02                          # the tracker thinks every beat is 20 ms late
    e = find_entry(feat, beat * 4 - 0.02 + 0.02, 4, lo=20.0, hi=30.0, X=1.0)
    assert e and abs(e["lag"]) <= 0.05


def test_no_room_means_no_entry():
    feat, beat = features()
    assert find_entry(feat, beat * 4, 4, lo=20.0, hi=20.2, X=1.0) is None
