from __future__ import annotations

import unittest

from app import ZONE_NAMES, validate_zones


VALID_ZONES = {
    "TRUCK_TURN_ZONE": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]],
    "ROAD_USER_APPROACH_ZONE": [[0.4, 0.1], [0.6, 0.1], [0.6, 0.3]],
    "CONFLICT_ZONE": [[0.2, 0.4], [0.4, 0.4], [0.4, 0.6]],
}


class FixedZoneValidationTests(unittest.TestCase):
    def test_accepts_three_normalized_polygons(self) -> None:
        validated = validate_zones(VALID_ZONES)
        self.assertEqual(tuple(validated), ZONE_NAMES)
        self.assertEqual(validated["CONFLICT_ZONE"][1], [0.4, 0.4])

    def test_rejects_missing_zone(self) -> None:
        incomplete = dict(VALID_ZONES)
        incomplete.pop("CONFLICT_ZONE")
        with self.assertRaises(ValueError):
            validate_zones(incomplete)

    def test_rejects_out_of_frame_point(self) -> None:
        invalid = {name: [point[:] for point in points] for name, points in VALID_ZONES.items()}
        invalid["CONFLICT_ZONE"][0][0] = 1.2
        with self.assertRaises(ValueError):
            validate_zones(invalid)


if __name__ == "__main__":
    unittest.main()

