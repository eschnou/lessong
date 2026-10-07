import pytest

from lessong import elevenlabs
from lessong.elevenlabs import DEFAULT_VOICES, ElevenLabs


def voices():
    return [dict(voice_id="m1", name="Marco", labels=dict(gender="male"), verified_languages=[dict(language="it")]),
            dict(voice_id="f1", name="Giulia", labels=dict(gender="female"), verified_languages=[dict(language="it")]),
            dict(voice_id="a1", name="Another", labels=dict(gender="female", language="pl"), verified_languages=[])]


def client():
    el = ElevenLabs("key")
    el._voices = voices()
    return el


def test_missing_key_is_a_clear_error():
    with pytest.raises(SystemExit, match="ELEVENLABS_API_KEY"):
        ElevenLabs(None)


def test_default_voices_follow_language_and_gender():
    el = client()
    for lang in ("en", "fr"):
        for g in ("male", "female"):
            assert el.resolve_voice(None, lang, g) == DEFAULT_VOICES[(lang, g)]


def test_other_languages_are_resolved_from_the_library_by_gender():
    el = client()
    assert el.resolve_voice(None, "it", "female") == "f1" and el.resolve_voice(None, "it", "male") == "m1"
    assert el.resolve_voice(None, "pl", "male") == "a1"                 # no male voice: falls back to what exists
    assert el.resolve_voice(None, "xx", "female") == DEFAULT_VOICES[("en", "female")]


def test_explicit_voice_wins_over_gender_and_unknown_names_fail():
    el = client()
    assert el.resolve_voice("marco", "it", "female") == "m1" and el.resolve_voice("a1", "it", "male") == "a1"
    assert el.resolve_voice("AbCdEfGhIjKlMnOpQrSt", "it") == "AbCdEfGhIjKlMnOpQrSt"
    with pytest.raises(SystemExit, match="no voice matching"):
        el.resolve_voice("nobody", "it")


def test_requests_retry_on_rate_limit_then_succeed(monkeypatch):
    seq = [429, 503, 200]

    class R:
        def __init__(self, c): self.status_code, self.text, self.content = c, "x", b"ok"

    monkeypatch.setattr(elevenlabs.requests, "request", lambda *a, **k: R(seq.pop(0)))
    monkeypatch.setattr(elevenlabs.time, "sleep", lambda s: None)
    assert client()._req("GET", "/x").content == b"ok" and not seq


def test_http_errors_raise_with_the_body(monkeypatch):
    class R:
        status_code, text, content = 401, "missing permission", b""

    monkeypatch.setattr(elevenlabs.requests, "request", lambda *a, **k: R())
    with pytest.raises(RuntimeError, match="missing permission"):
        client()._req("GET", "/x")


def test_tts_only_sends_voice_settings_that_were_asked_for(monkeypatch):
    sent = []
    monkeypatch.setattr(ElevenLabs, "_req", lambda self, m, p, **kw: sent.append(kw["json"]) or type("R", (), {"content": b""})())
    el = client()
    el.tts("hi", "v", "model", None, None)
    el.tts("hi", "v", "model", 0.9, None)
    assert "voice_settings" not in sent[0] and sent[1]["voice_settings"] == {"speed": 0.9}


def test_a_native_voice_is_preferred_over_one_that_merely_speaks_the_language():
    el = ElevenLabs("key")
    el._voices = [dict(voice_id="fr_accent", name="Premade", labels=dict(gender="male", language="en"), verified_languages=[dict(language="nl")]),
                  dict(voice_id="native_m", name="Native", labels=dict(gender="male", language="nl", accent="flemish"), verified_languages=[dict(language="nl")]),
                  dict(voice_id="native_f", name="Nativa", labels=dict(gender="female", language="nl"), verified_languages=[dict(language="nl")])]
    assert el.resolve_voice(None, "nl", "male") == "native_m" and el.resolve_voice(None, "nl", "female") == "native_f"
    el._voices = el._voices[:1]                                                # no native voice at all: the speaker of the language is still used
    assert el.resolve_voice(None, "nl", "male") == "fr_accent"
