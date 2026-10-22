"""Caption skip, audio timeline logic, and on-screen text report (no GPU, no network)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import audio
import download
import ocr
import watch

SONG = {"id": "1", "title": "Tavern", "artist": "Colm", "url": "https://shazam/1", "offset": 30.0}


def _url_run(monkeypatch, seen, *extra):
    monkeypatch.setenv("WATCH_WHISPER_BACKEND", "fasterwhisper")
    monkeypatch.setattr(watch, "has_cuda", lambda: True)
    monkeypatch.setattr(audio, "available", lambda: True)
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(watch, "fetch_captions", lambda *a, **kw: seen.update(kw) or {"info": {"title": "t"}})
    monkeypatch.setattr(watch, "download", lambda *a, **kw: {"video_path": "v.mp4"})
    monkeypatch.setattr(watch, "get_metadata", lambda *a: {"duration_seconds": 30, "has_audio": True, "has_video": False})
    monkeypatch.setattr(watch, "transcribe_video", lambda *a, **kw: ([{"start": 1, "end": 2, "text": "hi"}], "fasterwhisper"))
    monkeypatch.setattr(sys, "argv", ["watch", "https://youtu.be/abc", "--detail", "efficient", "--no-ocr", *extra])


def test_gpu_whisper_skips_captions(monkeypatch, capsys):
    seen = {}
    _url_run(monkeypatch, seen, "--no-audio")
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert seen["captions"] is False and "**Captions:** skipped" in out and "via whisper (fasterwhisper)" in out


def test_explicit_sub_lang_still_fetches_captions(monkeypatch, capsys):
    seen = {}
    _url_run(monkeypatch, seen, "--no-audio", "--sub-lang", "en")
    assert watch.main() == 0
    assert seen["captions"] is True and "Captions:** skipped" not in capsys.readouterr().out


def test_metadata_only_fetch_never_requests_a_track(monkeypatch, tmp_path):
    calls = []
    def run(cmd, timeout=None):
        calls.append(cmd)
        run_dir = cmd[cmd.index("-o") + 1].rsplit("/video.", 1)[0]
        (tmp_path / run_dir).joinpath("video.info.json").write_text(json.dumps(
            {"id": "x", "subtitles": {"en": [{"url": "u", "ext": "vtt"}]}}))
    monkeypatch.setattr(download, "_run", run)
    result = download.fetch_captions("https://youtu.be/x", tmp_path, captions=False)
    assert len(calls) == 1 and result["subtitle_path"] is None and result["info"]["id"] == "x"


def test_audio_report_reaches_the_watch_report(monkeypatch, capsys):
    seen = {}
    _url_run(monkeypatch, seen)
    monkeypatch.setattr(watch.audio, "cached", lambda *a: {
        "music": [{"start": 0.0, "end": 20.0, "speech": 0.9, "bpm": 112, "labels": ["Violin, fiddle"],
                   "under_voice_db": -11.6, "songs": [{**SONG, "via": "mix", "heard_at": [0.0]}]}],
        "events": [{"start": 4.0, "end": 5.0, "label": "Laughter", "score": 0.8},
                   {"start": 9.0, "end": 9.0, "label": "Whoosh, swoosh, swish", "score": 0.5}]})
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert "music 67% of the range in 1 stretch(es), 1 song(s) named, 1 laugh(s), 1 other sound(s)" in out
    assert '**"Tavern" by Colm**' in out and "~112 BPM" in out and "12 dB below it" in out
    assert "**Laughter:** 0:04-0:05" in out and "- 0:09 Whoosh, swoosh, swish (0.50)" in out


def test_audio_failure_is_reported_not_hidden(monkeypatch, capsys):
    seen = {}
    _url_run(monkeypatch, seen)
    def boom(*a):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(watch.audio, "cached", boom)
    assert watch.main() == 0
    assert "Audio timeline unavailable: RuntimeError: CUDA out of memory" in capsys.readouterr().out


def test_transcript_detail_skips_the_audio_timeline(monkeypatch, capsys):
    seen = {}
    _url_run(monkeypatch, seen)
    monkeypatch.setattr(watch.audio, "cached", lambda *a: pytest.fail("transcript-only runs skip audio"))
    monkeypatch.setattr(sys, "argv", ["watch", "https://youtu.be/abc", "--detail", "transcript"])
    assert watch.main() == 0
    assert "## Audio" not in capsys.readouterr().out


def test_hung_audio_times_out_instead_of_blocking(monkeypatch, capsys):
    import threading
    seen, release = {}, threading.Event()
    _url_run(monkeypatch, seen)
    monkeypatch.setattr(watch, "AUDIO_TIMEOUT", 0.2)
    monkeypatch.setattr(watch.audio, "cached", lambda *a: release.wait(5))
    assert watch.main() == 0
    release.set()
    assert "Audio timeline unavailable: still running after 0.2s" in capsys.readouterr().out


def test_event_spanning_the_range_start_is_kept():
    result = {"music": [], "events": [{"start": 2.0, "end": 8.0, "label": "Laughter", "score": 0.5}]}
    assert "1 laugh(s)" in audio.summary(result, 5.0, 20.0)
    assert "**Laughter:** 0:02-0:08" in "\n".join(audio.report(result, 5.0, 20.0))


def _fake_shazam(monkeypatch, answers):
    """answers: queue of per-call lists, one list per shazam_all call."""
    calls = []
    def fake(clips, work, failures):
        calls.append(len(clips))
        return answers.pop(0)
    monkeypatch.setattr(audio, "shazam_all", fake)
    return calls


def _stereo(seconds=60):
    return np.zeros((2, audio.HQ_SR * seconds), np.float32)


def test_mix_match_needs_no_retry(monkeypatch, tmp_path):
    calls = _fake_shazam(monkeypatch, [[SONG], []])
    out = audio.identify([(0.0, 12.0, True)], _stereo(), _stereo(), tmp_path, [])
    assert out[0]["via"] == "mix" and calls == [1, 0]


def test_voice_removed_match_counts_only_when_confirmed(monkeypatch, tmp_path):
    later = {**SONG, "offset": 35.0}
    _fake_shazam(monkeypatch, [[None, None], [SONG, later, SONG, {**SONG, "id": "2"}]])
    out = audio.identify([(0.0, 12.0, True), (20.0, 32.0, True)], _stereo(), _stereo(), tmp_path, [])
    assert out[0]["via"] == "voice removed" and out[1] is None


def test_unvoiced_miss_is_not_retried(monkeypatch, tmp_path):
    calls = _fake_shazam(monkeypatch, [[None], []])
    assert audio.identify([(0.0, 12.0, False)], _stereo(), _stereo(), tmp_path, []) == [None]
    assert calls == [1, 0]


def test_events_split_laughter_from_effects_and_merge_runs():
    labels = {i: f"L{i}" for i in range(527)}
    times = np.array([0.5, 1.0, 1.5, 3.0])
    voice, rest = np.zeros((4, 527)), np.zeros((4, 527))
    voice[0:2, 16] = 0.9          # laughter, two adjacent windows
    voice[:, 0] = 0.99            # speech never becomes an event
    rest[3, 400] = 0.6            # an effect
    rest[:, audio.MUSIC] = 0.9    # music never becomes an event
    loud = np.zeros(4)
    out = audio.events(times, voice, rest, labels, 0.5, loud, loud)
    assert out == [{"start": 0.5, "end": 1.0, "label": "Laughter", "score": 0.9},
                   {"start": 3.0, "end": 3.0, "label": "L400", "score": 0.6}]
    # The same scores on near-silent leftovers (talking-head noise) are not events.
    quiet = np.full(4, -30.0)
    assert audio.events(times, voice, rest, labels, 0.5, quiet, quiet) == []
    # A laugh track ~20 dB under the dialogue still counts; a quiet effect does not.
    under = np.full(4, -20.0)
    assert [e["label"] for e in audio.events(times, voice, rest, labels, 0.5, under, under)] == ["Laughter"]


def _burst(seconds, at, hz, amp, sr=audio.HQ_SR):
    """Stereo bed of faint noise with one 80ms tone burst."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.002, int(seconds * sr)).astype(np.float32)
    n = int(0.08 * sr)
    x[int(at * sr):int(at * sr) + n] += amp * np.sin(2 * np.pi * hz * np.arange(n) / sr).astype(np.float32)
    return np.stack([x, x])


def test_quiet_ding_is_a_hit_but_a_low_thump_is_not():
    ref = 20 * np.log10(0.05)  # a mix averaging -26 dBFS
    ding = audio.hits(_burst(6, 3.0, 6000, 0.1), ref, [], np.array([]))
    assert len(ding) == 1 and abs(ding[0]["time"] - 3.0) < 0.1 and not ding[0]["on_cut"]
    # A mic pop or plosive is low only: it never jumps above 4 kHz.
    assert audio.hits(_burst(6, 3.0, 90, 0.3), ref, [], np.array([])) == []


def test_hit_under_music_needs_a_video_cut():
    ref = 20 * np.log10(0.05)
    bed = _burst(6, 3.0, 6000, 0.1)
    assert audio.hits(bed, ref, [[0.0, 6.0]], np.array([])) == []
    assert audio.hits(bed, ref, [[0.0, 6.0]], np.array([3.1]))[0]["on_cut"] is True


def test_quiet_hit_report_line():
    result = {"music": [], "events": [{"start": 7.2, "end": 7.2, "label": "Ding", "score": 0.1,
                                       "hit": {"db": -9.0, "on_cut": True}}]}
    assert "- 0:07.2 Ding (0.10) · quiet hit, 9 dB under the mix average, on a video cut" in audio.report(result, 0, 30)


def test_spans_join_short_gaps_and_drop_blips():
    times = np.arange(0.5, 12, 1.0)
    on = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0], bool)
    assert audio.spans(times, on, 1.0, 2.0, 3.0) == [[0.0, 6.0]]


def test_audio_report_limits_to_range_and_flags_shazam_failures():
    result = {"music": [{"start": 0.0, "end": 10.0, "speech": 0.0, "bpm": None, "labels": [], "songs": []},
                        {"start": 50.0, "end": 60.0, "speech": 0.0, "bpm": 90, "labels": [], "songs": []}],
              "events": [{"start": 55.0, "end": 55.0, "label": "Boom", "score": 0.7}],
              "shazam_failures": ["FailedDecodeJson: Failed to decode json"]}
    text = "\n".join(audio.report(result, 40.0, 70.0))
    assert "0:00 → 0:10" not in text and "0:50 → 1:00" in text and "no Shazam match" in text
    assert "Shazam failed on 1 lookup(s)" in text and "- 0:55 Boom (0.70)" in text


def test_cached_audio_recomputes_on_version_change(monkeypatch, tmp_path):
    saved = tmp_path / "v.audio.json"
    saved.write_text(json.dumps({"version": audio.VERSION - 1, "music": [], "events": []}))
    monkeypatch.setattr(audio, "analyze", lambda media, work, has_video: {"music": ["fresh"], "events": [], "shazam_failures": []})
    assert audio.cached("v.mp4", saved, tmp_path, True)["music"] == ["fresh"]
    monkeypatch.setattr(audio, "analyze", lambda media, work, has_video: pytest.fail("cache hit must not recompute"))
    assert audio.cached("v.mp4", saved, tmp_path, True)["music"] == ["fresh"]


def test_result_with_shazam_failures_is_not_cached(monkeypatch, tmp_path):
    saved = tmp_path / "v.audio.json"
    monkeypatch.setattr(audio, "analyze", lambda media, work, has_video: {"music": [], "events": [], "shazam_failures": ["429"]})
    audio.cached("v.mp4", saved, tmp_path, True)
    assert not saved.exists()


def test_shazam_call_retries_then_reports(monkeypatch, tmp_path):
    import shazamio
    tries = []
    async def recognize(self, path):
        tries.append(path)
        if len(tries) < 3:
            raise shazamio.exceptions.FailedDecodeJson("Failed to decode json")
        return {"track": {"key": 7, "title": "T", "subtitle": "A", "url": "u"}, "matches": [{"offset": 1.0}]}
    monkeypatch.setattr(shazamio.Shazam, "recognize", recognize)
    monkeypatch.setattr(audio, "SHAZAM_BACKOFF", (0.0, 0.0))
    failures = []
    assert audio.shazam_all([_stereo(5)], tmp_path, failures)[0]["title"] == "T" and not failures
    tries.clear()
    monkeypatch.setattr(audio, "SHAZAM_BACKOFF", (0.0,))
    assert audio.shazam_all([_stereo(5)], tmp_path, failures) == [None] and len(failures) == 1


def test_screen_text_prints_only_new_lines():
    stamps = ["00:01", "00:02", "00:03"]
    per_frame = [["WHERE STATE LIVES", "agent sqlite"], ["where state lives", "agent sqlite"], ["GATHER", "agent sqlite"]]
    assert ocr.report(stamps, per_frame) == ["- 00:01 WHERE STATE LIVES | agent sqlite", "- 00:03 GATHER"]
    assert ocr.report(stamps, [[], [], []]) == ["- No readable text."]
    # OCR flicker on a held caption ("new." / "new:" / "new.") prints once.
    assert ocr.report(stamps, [["new."], ["new:"], ["new."]]) == ["- 00:01 new."]


def test_screen_text_is_capped(monkeypatch):
    monkeypatch.setattr(ocr, "MAX_CHARS", 30)
    out = ocr.report(["a", "b", "c"], [["x" * 20], ["y" * 20], ["z"]])
    assert out[0].startswith("- a x") and out[-1].startswith("- … more text cut")
