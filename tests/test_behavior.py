from collections import deque

from behavior import PoseBehaviorAnalyzer, unavailable_observation
import unittest


def test_unavailable_observation_is_conservative():
    observation = unavailable_observation("missing")
    assert observation["activity"] == "POSE NOT AVAILABLE"
    assert observation["orientation"] == "HEAD ORIENTATION UNKNOWN"
    assert observation["awareness"] == "CANNOT BE INFERRED"
    assert observation["available"] is False


def test_activity_needs_multiple_frames():
    assert PoseBehaviorAnalyzer.classify_activity([(0.5, 0.5, 0.8)] * 3) == ("MOTION UNCERTAIN", "LOW")


def test_activity_classifies_standing_and_walking():
    standing = [(0.5 + index * 0.002, 0.5, 0.8) for index in range(5)]
    walking = [(0.5 + index * 0.08, 0.5, 0.8) for index in range(5)]
    assert PoseBehaviorAnalyzer.classify_activity(standing) == ("LIKELY STANDING", "MEDIUM")
    assert PoseBehaviorAnalyzer.classify_activity(walking) == ("LIKELY WALKING", "MEDIUM")


def test_orientation_is_always_low_confidence_or_unknown():
    unknown = PoseBehaviorAnalyzer.classify_orientation(0.5, 0.45, 0.55, 1)
    toward = PoseBehaviorAnalyzer.classify_orientation(0.68, 0.42, 0.52, 1)
    assert unknown == ("HEAD ORIENTATION UNKNOWN", "UNKNOWN")
    assert toward == ("ORIENTED TOWARD VEHICLE — LOW-CONFIDENCE PROXY", "LOW")


class BehaviorUnittestAdapter(unittest.TestCase):
    def test_all_behavior_cases(self):
        test_unavailable_observation_is_conservative()
        test_activity_needs_multiple_frames()
        test_activity_classifies_standing_and_walking()
        test_orientation_is_always_low_confidence_or_unknown()

