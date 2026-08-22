from __future__ import annotations

import unittest

from app import DetectionEngine, RELEVANT_CLASSES


class SceneContextTests(unittest.TestCase):
    def test_road_context_classes_are_requested_from_yolo(self) -> None:
        self.assertEqual(RELEVANT_CLASSES[2], "car")
        self.assertEqual(RELEVANT_CLASSES[5], "bus")
        self.assertEqual(RELEVANT_CLASSES[9], "traffic light")
        self.assertEqual(RELEVANT_CLASSES[11], "stop sign")

    def test_scene_context_separates_controls_from_vehicles(self) -> None:
        context = DetectionEngine._scene_context(
            [
                {"class": "car"},
                {"class": "bus"},
                {"class": "traffic light"},
                {"class": "stop sign"},
                {"class": "person"},
            ]
        )
        self.assertEqual(context["vehicles"], ["car", "bus"])
        self.assertEqual(context["traffic_controls"], ["traffic light", "stop sign"])


if __name__ == "__main__":
    unittest.main()

