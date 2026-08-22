from __future__ import annotations

import unittest

from app import validate_surfaces


class SurfaceMapTests(unittest.TestCase):
    def test_accepts_any_number_of_sidewalk_polygons(self) -> None:
        sidewalks = [
            [[0.01 * index, 0.1], [0.01 * index + 0.005, 0.2], [0.01 * index, 0.3]]
            for index in range(12)
        ]
        surfaces = {
            "road_polygons": [[[0.1, 0.1], [0.8, 0.1], [0.8, 0.8]]],
            "sidewalk_polygons": sidewalks,
        }
        self.assertEqual(len(validate_surfaces(surfaces)["sidewalk_polygons"]), 12)

    def test_rejects_incomplete_surface_polygon(self) -> None:
        surfaces = {
            "road_polygons": [[[0.1, 0.1], [0.8, 0.1]]],
            "sidewalk_polygons": [],
        }
        with self.assertRaises(ValueError):
            validate_surfaces(surfaces)


if __name__ == "__main__":
    unittest.main()

