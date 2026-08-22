from __future__ import annotations

import unittest

import numpy as np

from scene import RoadSidewalkAnalyzer


class SceneDraftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = RoadSidewalkAnalyzer()
        self.road = np.zeros((400, 600), dtype=np.uint8)
        self.road[60:390, 20:450] = 1
        self.sidewalk = np.zeros_like(self.road)
        self.sidewalk[60:390, 451:590] = 1

    def test_draft_zones_follow_the_road_sidewalk_boundary(self) -> None:
        draft = self.analyzer._draft_zones(self.road, self.sidewalk)
        self.assertEqual(set(draft or {}), {
            "TRUCK_TURN_ZONE",
            "ROAD_USER_APPROACH_ZONE",
            "CONFLICT_ZONE",
        })
        self.assertTrue(all(len(points) >= 3 for points in (draft or {}).values()))

    def test_draft_is_not_created_without_a_sidewalk(self) -> None:
        self.assertIsNone(self.analyzer._draft_zones(self.road, np.zeros_like(self.road)))


if __name__ == "__main__":
    unittest.main()

