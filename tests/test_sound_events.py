from app import DetectionEngine
import unittest


def test_sound_once_per_risk_episode():
    engine = DetectionEngine()
    engine._update_sound_event("MONITORING", "CAUTION")
    first = dict(engine.latest_sound_event)
    engine._update_sound_event("CAUTION", "CAUTION")
    assert engine.latest_sound_event == first
    engine._update_sound_event("CAUTION", "DANGER")
    danger = dict(engine.latest_sound_event)
    assert danger["cue"] == "DANGER_TWO_PULSE"
    engine._update_sound_event("DANGER", "CAUTION")
    assert engine.latest_sound_event == danger


def test_new_episode_can_emit_new_caution_chime():
    engine = DetectionEngine()
    engine._update_sound_event("MONITORING", "CAUTION")
    first_id = engine.latest_sound_event["id"]
    engine._update_sound_event("CAUTION", "VEHICLE TRACKED")
    engine._update_sound_event("VEHICLE TRACKED", "CAUTION")
    assert engine.latest_sound_event["id"] > first_id
    assert engine.latest_sound_event["cue"] == "CAUTION_CHIME"


class SoundUnittestAdapter(unittest.TestCase):
    def test_sound_deduplication(self):
        test_sound_once_per_risk_episode()
        test_new_episode_can_emit_new_caution_chime()

