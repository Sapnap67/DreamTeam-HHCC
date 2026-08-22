from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from behavior import PoseBehaviorAnalyzer, empty_analysis


class PoseBehaviorTests(unittest.TestCase):
    @staticmethod
    def landmarks() -> list[SimpleNamespace]:
        points = [SimpleNamespace(x=0.2, y=0.4, visibility=1.0, presence=1.0) for _ in range(33)]
        values = {
            0: (0.25, 0.18), 7: (0.19, 0.18), 8: (0.21, 0.18),
            11: (0.16, 0.30), 12: (0.24, 0.30), 23: (0.17, 0.50),
            24: (0.23, 0.50), 25: (0.17, 0.70), 26: (0.23, 0.70),
            27: (0.17, 0.90), 28: (0.23, 0.90),
        }
        for index, (x, y) in values.items():
            points[index] = SimpleNamespace(x=x, y=y, visibility=1.0, presence=1.0)
        return points

    def setUp(self) -> None:
        self.analyzer = PoseBehaviorAnalyzer(Path("missing.task"))
        self.person = {"box": [100, 100, 300, 500], "motion_track_key": "person:1"}
        self.truck = {"box": [500, 100, 900, 500]}

    def test_missing_model_degrades_without_breaking_warning_system(self) -> None:
        self.assertFalse(self.analyzer.start())
        self.assertEqual(self.analyzer.unavailable_reason, "POSE MODEL NOT INSTALLED")

    def test_awareness_is_never_inferred(self) -> None:
        self.assertEqual(empty_analysis()["awareness"], "CANNOT BE INFERRED")

    def test_head_orientation_is_explicitly_a_low_confidence_proxy(self) -> None:
        label, confidence = self.analyzer._head_orientation(self.landmarks(), self.person, self.truck)
        self.assertEqual(label, "ORIENTED TOWARD VEHICLE (PROXY)")
        self.assertEqual(confidence, "LOW")

    def test_stationary_pose_requires_history(self) -> None:
        result = ("", "")
        for timestamp in (0, 100, 200, 300):
            result = self.analyzer._classify_activity(self.landmarks(), self.person, timestamp, 1000, 600)
        self.assertEqual(result, ("LIKELY STANDING", "MEDIUM"))

    def test_moving_hips_are_likely_walking(self) -> None:
        result = ("", "")
        for timestamp, hip_x in ((0, 0.20), (100, 0.23), (200, 0.26), (300, 0.30)):
            points = self.landmarks()
            points[23].x = hip_x - 0.03
            points[24].x = hip_x + 0.03
            result = self.analyzer._classify_activity(points, self.person, timestamp, 1000, 600)
        self.assertEqual(result, ("LIKELY WALKING", "MEDIUM"))


if __name__ == "__main__":
    unittest.main()
