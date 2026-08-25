import unittest
from collections import deque

import numpy as np

from app import DANGER_CLEAR_FRAMES, DetectionEngine


FRAME_WIDTH = 1000
FRAME_HEIGHT = 600


def tracked_item(engine, key, class_name, confidence, start, end, box_width=100, box_height=100):
    history = deque(maxlen=8)
    for frame_index in range(4):
        ratio = frame_index / 3
        point = np.asarray(start, dtype=np.float32) + ratio * (
            np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
        )
        history.append((frame_index, point))
    center_x = end[0] * FRAME_WIDTH
    bottom_y = end[1] * FRAME_HEIGHT
    box = [
        center_x - box_width / 2,
        bottom_y - box_height,
        center_x + box_width / 2,
        bottom_y,
    ]
    engine.motion_tracks[key] = {"history": history}
    return {
        "class": class_name,
        "confidence": confidence,
        "box": box,
        "contact": list(end),
        "smoothed_contact": list(end),
        "motion_track_key": key,
    }


class AllVehicleRiskTests(unittest.TestCase):
    def setUp(self):
        self.engine = DetectionEngine()

    def test_car_and_pedestrian_can_trigger_danger(self):
        car = tracked_item(self.engine, "car", "car", 0.9, (0.20, 0.70), (0.32, 0.70), 120)
        person = tracked_item(self.engine, "person", "person", 0.9, (0.55, 0.70), (0.43, 0.70), 42, 120)
        state, reason, evidence = self.engine._evaluate_risk([car, person], None, FRAME_WIDTH, FRAME_HEIGHT, "right")
        self.assertEqual(state, "DANGER")
        self.assertEqual(evidence["vehicle_class"], "car")
        self.assertIn("Car", reason)

    def test_motorcycle_and_cyclist_can_trigger_caution(self):
        motorcycle = tracked_item(self.engine, "moto", "motorcycle", 0.88, (0.30, 0.72), (0.31, 0.72), 80)
        bicycle = tracked_item(self.engine, "bike", "bicycle", 0.86, (0.45, 0.72), (0.435, 0.72), 45, 105)
        state, reason, evidence = self.engine._evaluate_risk([motorcycle, bicycle], None, FRAME_WIDTH, FRAME_HEIGHT, "right")
        self.assertEqual(state, "CAUTION")
        self.assertEqual(evidence["vehicle_class"], "motorcycle")
        self.assertIn("cyclist", reason)

    def test_truck_and_pedestrian_still_trigger_danger(self):
        truck = tracked_item(self.engine, "truck", "truck", 0.94, (0.18, 0.68), (0.32, 0.68), 150, 160)
        person = tracked_item(self.engine, "person", "person", 0.91, (0.58, 0.68), (0.44, 0.68), 40, 115)
        state, _, evidence = self.engine._evaluate_risk([truck, person], truck, FRAME_WIDTH, FRAME_HEIGHT, "right")
        self.assertEqual(state, "DANGER")
        self.assertTrue(evidence["heavy_vehicle_detected"])

    def test_nearby_non_converging_objects_do_not_trigger_danger(self):
        car = tracked_item(self.engine, "car", "car", 0.9, (0.28, 0.70), (0.32, 0.70), 110)
        person = tracked_item(self.engine, "person", "person", 0.9, (0.40, 0.70), (0.44, 0.70), 40, 110)
        state, _, evidence = self.engine._evaluate_risk([car, person], None, FRAME_WIDTH, FRAME_HEIGHT, "right")
        self.assertNotEqual(state, "DANGER")
        self.assertFalse(evidence["distance_decreasing"])

    def test_highest_risk_pair_is_selected(self):
        car = tracked_item(self.engine, "car", "car", 0.91, (0.20, 0.68), (0.32, 0.68), 120)
        person = tracked_item(self.engine, "person", "person", 0.92, (0.55, 0.68), (0.43, 0.68), 40, 115)
        truck = tracked_item(self.engine, "truck", "truck", 0.95, (0.64, 0.76), (0.65, 0.76), 130, 150)
        bicycle = tracked_item(self.engine, "bike", "bicycle", 0.87, (0.82, 0.76), (0.80, 0.76), 45, 105)
        state, _, evidence = self.engine._evaluate_risk(
            [car, person, truck, bicycle], truck, FRAME_WIDTH, FRAME_HEIGHT, "right"
        )
        self.assertEqual(state, "DANGER")
        self.assertEqual(evidence["vehicle_class"], "car")
        self.assertEqual(evidence["road_user_class"], "person")

    def test_warning_hysteresis_prevents_rapid_flicker(self):
        visible, reason, danger, caution, safe = "MONITORING", "clear", 0, 0, 0
        for _ in range(3):
            visible, reason, danger, caution, safe = self.engine._apply_warning_hysteresis(
                "DANGER", "risk", visible, reason, danger, caution, safe
            )
        self.assertEqual(visible, "CAUTION")
        visible, reason, danger, caution, safe = self.engine._apply_warning_hysteresis(
            "DANGER", "risk", visible, reason, danger, caution, safe
        )
        self.assertEqual(visible, "DANGER")
        visible, reason, danger, caution, safe = self.engine._apply_warning_hysteresis(
            "MONITORING", "clear", visible, reason, danger, caution, safe
        )
        self.assertEqual(visible, "DANGER")
        for _ in range(DANGER_CLEAR_FRAMES - 1):
            visible, reason, danger, caution, safe = self.engine._apply_warning_hysteresis(
                "MONITORING", "clear", visible, reason, danger, caution, safe
            )
        self.assertEqual(visible, "MONITORING")


if __name__ == "__main__":
    unittest.main()
