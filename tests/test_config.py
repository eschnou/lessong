import argparse
from dataclasses import fields

import pytest

from lessong import config
from lessong.config import Settings


def parser(stages):
    p = argparse.ArgumentParser()
    config.add_arguments(p, stages)
    return p


def test_every_setting_has_a_flag_for_its_stage():
    for stages in ({"plan", "both"}, {"render", "both"}, {"plan", "render", "both"}):
        p = parser(stages)
        dests = {a.dest for a in p._actions}
        for f in fields(Settings):
            assert (f.name in dests) == (f.metadata["stage"] in stages), f.name


def test_defaults_round_trip_through_the_parser():
    s = config.from_args(parser({"plan", "render", "both"}).parse_args([]))
    assert s == Settings()


def test_flag_spellings_and_types():
    s = config.from_args(parser({"plan", "render", "both"}).parse_args(
        ["--from", "es", "--to", "de", "--bed-db", "-20", "--loop-bars", "2", "--loop-start", "12.5", "--sections", "gap",
         "--voice-gender", "female", "--repeats", "skip", "--plain-entry"]))
    assert (s.source_lang, s.target_lang, s.bed_db, s.loop_bars, s.loop_start) == ("es", "de", -20.0, 2, 12.5)
    assert (s.section_mode, s.voice_gender, s.repeats, s.plain_entry) == ("gap", "female", "skip", True)


@pytest.mark.parametrize("bad", [["--sections", "nope"], ["--voice-gender", "robot"], ["--repeats", "maybe"], ["--loop-bars", "x"]])
def test_invalid_values_are_rejected(bad):
    with pytest.raises(SystemExit):
        parser({"plan", "render", "both"}).parse_args(bad)


def test_help_text_never_breaks_argparse_formatting():
    for stages in ({"plan", "both"}, {"render", "both"}):
        parser(stages).format_help()      # a stray '%' in a help string raises here
