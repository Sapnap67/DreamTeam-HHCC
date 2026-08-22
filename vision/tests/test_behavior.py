from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from behavior import PoseBehaviorAnalyzer, crossing_advisory, empty_analysis


class CrossingAdvisoryTests(unittest.TestCase):
    def test_danger_never_reports_safe(self) -> None:
        advisory, reason = crossing_advisory("DANGER")
        self.assertEqual(advisory, "DO NOT CROSS")
        self.assertIn("conflict zone", reason)

    def test_truck_present_stays_conservative(self) -> None:
        advisory, _reason = crossing_advisory("TRUCK TRACKED")
        self.assertEqual(advisory, "WAIT — TRUCK PRESENT")

    def test_monitoring_requires_independent_checks(self) -> None:
        advisory, reason = crossing_advisory("MONITORING")
        self.assertEqual(advisory, "CHECK SIGNAL AND TRAFFIC")
        self.assertNotIn("SAFE", advisory)
        self.assertIn("cannot declare", reason)

    def test_awareness_is_never_inferred(self) -> None:
        analysis = empty_analysis("MONITORING")
        self.assertEqual(analysis["awareness"], "CANNOT BE INFERRED")


class PoseHeuristicTests(unittest.TestCase):
    @staticmethod
    def landmarks() -> list[SimpleNamespace]:
        points = [SimpleNamespace(x=0.2, y=0.4, visibility=1.0, presence=1.0) for _ in range(33)]
        points[0] = SimpleNamespace(x=0.25, y=0.18, visibility=1.0, presence=1.0)
        points[7] = SimpleNamespace(x=0.19, y=0.18, visibility=1.0, presence=1.0)
        points[8] = SimpleNamespace(x=0.21, y=0.18, visibility=1.0, presence=1.0)
        points[11] = SimpleNamespace(x=0.16, y=0.30, visibility=1.0, presence=1.0)
        points[12] = SimpleNamespace(x=0.24, y=0.30, visibility=1.0, presence=1.0)
        points[23] = SimpleNamespace(x=0.17, y=0.50, visibility=1.0, presence=1.0)
        points[24] = SimpleNamespace(x=0.23, y=0.50, visibility=1.0, presence=1.0)
        points[25] = SimpleNamespace(x=0.17, y=0.70, visibility=1.0, presence=1.0)
        points[26] = SimpleNamespace(x=0.23, y=0.70, visibility=1.0, presence=1.0)
        points[27] = SimpleNamespace(x=0.17, y=0.90, visibility=1.0, presence=1.0)
        points[28] = SimpleNamespace(x=0.23, y=0.90, visibility=1.0, presence=1.0)
        return points

    def setUp(self) -> None:
        self.analyzer = PoseBehaviorAnalyzer(Path("unused.task"))
        self.person = {"box": [100, 100, 300, 500]}
        self.truck = {"box": [500, 100, 900, 500]}

    def test_head_orientation_is_only_a_proxy(self) -> None:
        label, confidence = self.analyzer._head_orientation(self.landmarks(), self.person, self.truck)
        self.assertEqual(label, "ORIENTED TOWARD TRUCK (PROXY)")
        self.assertEqual(confidence, "LOW")

    def test_stationary_pose_needs_multiple_samples(self) -> None:
        points = self.landmarks()
        result = ("", "")
        for timestamp in (0, 100, 200, 300):
            result = self.analyzer._classify_activity(points, self.person, timestamp, 1000, 600)
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

