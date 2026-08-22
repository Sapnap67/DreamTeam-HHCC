"""Semantic scene analysis for a stationary intersection camera.

The model labels road and sidewalk pixels.  It produces draft polygons only;
the application deliberately requires an operator to accept or adjust them
before they become warning zones.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np


SCENE_MODEL_ID = os.environ.get(
    "SCENE_SEGMENTATION_MODEL", "nvidia/segformer-b0-finetuned-ade-512-512"
)
MIN_COMPONENT_AREA = 0.001


def empty_scene(status: str = "NOT ANALYZED", error: str | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "error": error,
        "road_polygons": [],
        "sidewalk_polygons": [],
        "suggested_zones": None,
        "model": SCENE_MODEL_ID,
    }


class RoadSidewalkAnalyzer:
    """Lazy ADE20K segmentation, run once for each fixed-camera clip."""

    def __init__(self) -> None:
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._road_class_ids: list[int] = []
        self._sidewalk_class_ids: list[int] = []

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        self._torch = torch
        self._processor = SegformerImageProcessor.from_pretrained(SCENE_MODEL_ID)
        self._model = SegformerForSemanticSegmentation.from_pretrained(SCENE_MODEL_ID)
        self._model.eval()
        labels = {int(class_id): str(name).lower() for class_id, name in self._model.config.id2label.items()}
        self._road_class_ids = [class_id for class_id, name in labels.items() if name == "road"]
        self._sidewalk_class_ids = [
            class_id for class_id, name in labels.items() if name in {"sidewalk", "pavement"}
        ]
        if not self._road_class_ids or not self._sidewalk_class_ids:
            raise ValueError("The scene model must provide road and sidewalk labels.")

    @staticmethod
    def _normalized_contours(mask: np.ndarray, limit: int | None = None) -> list[list[list[float]]]:
        height, width = mask.shape[:2]
        minimum_area = width * height * MIN_COMPONENT_AREA
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[tuple[float, list[list[float]]]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < minimum_area:
                continue
            simplified = cv2.approxPolyDP(contour, 0.012 * cv2.arcLength(contour, True), True)
            if len(simplified) < 3:
                continue
            points = [
                [round(float(point[0][0]) / width, 5), round(float(point[0][1]) / height, 5)]
                for point in simplified
            ]
            polygons.append((area, points))
        ordered = [points for _, points in sorted(polygons, key=lambda item: item[0], reverse=True)]
        return ordered if limit is None else ordered[:limit]

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            return np.zeros_like(mask, dtype=np.uint8)
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == component).astype(np.uint8)

    @staticmethod
    def _polygon_from_mask(mask: np.ndarray) -> list[list[float]] | None:
        polygons = RoadSidewalkAnalyzer._normalized_contours(mask, limit=1)
        return polygons[0] if polygons else None

    def _draft_zones(self, road: np.ndarray, sidewalk: np.ndarray) -> dict[str, list[list[float]]] | None:
        """Make conservative, editable drafts at the road/sidewalk boundary.

        A road shape alone cannot reveal a truck's exact sweep path.  These
        polygons are intentionally marked as a draft in the API/UI.
        """
        height, width = road.shape[:2]
        road = self._largest_component(road)
        sidewalk = (cv2.morphologyEx(sidewalk.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0).astype(np.uint8)
        if not road.any() or not sidewalk.any():
            return None

        # Road pixels nearest a sidewalk are a practical first approximation
        # of the crossing / interaction band for a fixed view.
        distance_to_sidewalk = cv2.distanceTransform((1 - sidewalk).astype(np.uint8), cv2.DIST_L2, 5)
        road_distance_values = distance_to_sidewalk[road.astype(bool)]
        if road_distance_values.size == 0:
            return None
        boundary_distance = max(18.0, float(np.percentile(road_distance_values, 12)))
        conflict = ((road > 0) & (distance_to_sidewalk <= boundary_distance * 1.25)).astype(np.uint8)
        conflict = cv2.morphologyEx(conflict, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        conflict = self._largest_component(conflict)
        if not conflict.any():
            return None

        # The approach draft is the adjacent sidewalk band.  The truck draft
        # is the road-side envelope leading into the same boundary band.
        near_conflict = cv2.dilate(conflict, np.ones((max(25, width // 18), max(25, height // 18)), np.uint8))
        approach = ((sidewalk > 0) & (near_conflict > 0)).astype(np.uint8)
        approach = self._largest_component(approach)
        truck_turn = ((road > 0) & (cv2.dilate(conflict, np.ones((max(55, width // 8), max(55, height // 8)), np.uint8)) > 0)).astype(np.uint8)
        truck_turn = self._largest_component(truck_turn)

        polygons = {
            "TRUCK_TURN_ZONE": self._polygon_from_mask(truck_turn),
            "ROAD_USER_APPROACH_ZONE": self._polygon_from_mask(approach),
            "CONFLICT_ZONE": self._polygon_from_mask(conflict),
        }
        if any(points is None or len(points) < 3 for points in polygons.values()):
            return None
        return polygons  # type: ignore[return-value]

    def analyze(self, frame: np.ndarray) -> dict[str, Any]:
        try:
            self._load()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = self._processor(images=rgb, return_tensors="pt")
            with self._torch.no_grad():
                logits = self._model(**inputs).logits
            logits = self._torch.nn.functional.interpolate(
                logits, size=frame.shape[:2], mode="bilinear", align_corners=False
            )
            labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            road = np.isin(labels, self._road_class_ids).astype(np.uint8)
            sidewalk = np.isin(labels, self._sidewalk_class_ids).astype(np.uint8)
            return {
                "status": "DRAFT READY",
                "error": None,
                "road_polygons": self._normalized_contours(road),
                "sidewalk_polygons": self._normalized_contours(sidewalk),
                "suggested_zones": self._draft_zones(road, sidewalk),
                "model": SCENE_MODEL_ID,
            }
        except Exception as exc:
            return empty_scene("UNAVAILABLE", str(exc))

