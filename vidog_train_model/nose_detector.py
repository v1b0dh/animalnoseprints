"""
nose_detector.py - nose localization helpers for DogID.

Drop your trained detector at checkpoints/nose_detector.onnx later and replace
_detect_with_model with the detector-specific postprocessing. Until then, the
app uses a conservative center crop so the registration/identification workflow
is already wired around nose crops instead of full images.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image


BBox = Tuple[int, int, int, int]


@dataclass
class NoseCropResult:
    crop: Image.Image
    bbox: BBox
    confidence: float
    source: str


class NoseDetector:
    def __init__(
        self,
        weights_path: str = "checkpoints/nose_detector.onnx",
        input_size: int = 640,
        confidence_threshold: float = 0.60,
        fallback_crop_ratio: float = 0.55,
        padding_ratio: float = 0.18,
    ):
        self.weights_path = weights_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.fallback_crop_ratio = fallback_crop_ratio
        self.padding_ratio = padding_ratio
        self.model = self._load_model(weights_path)

    @property
    def has_model(self) -> bool:
        return self.model is not None

    def detect_and_crop(self, img: Image.Image) -> NoseCropResult:
        image = img.convert("RGB")
        if self.model is not None:
            detected = self._detect_with_model(image)
            if detected is not None:
                bbox, confidence = detected
                bbox = self._pad_bbox(bbox, image.size, self.padding_ratio)
                return NoseCropResult(
                    crop=image.crop(bbox),
                    bbox=bbox,
                    confidence=confidence,
                    source="model",
                )

        bbox = self._center_bbox(image.size, self.fallback_crop_ratio)
        return NoseCropResult(
            crop=image.crop(bbox),
            bbox=bbox,
            confidence=0.50,
            source="center_fallback",
        )

    def _load_model(self, weights_path: str):
        if not os.path.exists(weights_path):
            return None

        try:
            import cv2

            return cv2.dnn.readNetFromONNX(weights_path)
        except Exception:
            return None

    def _detect_with_model(self, img: Image.Image) -> Optional[Tuple[BBox, float]]:
        """Run a common YOLO-style ONNX detector and return the best nose box."""
        try:
            import cv2

            img_np = np.array(img.convert("RGB"))
            orig_h, orig_w = img_np.shape[:2]
            blob = cv2.dnn.blobFromImage(
                img_np,
                scalefactor=1.0 / 255.0,
                size=(self.input_size, self.input_size),
                swapRB=False,
                crop=False,
            )
            self.model.setInput(blob)
            outputs = self.model.forward()
            boxes = self._parse_yolo_outputs(outputs, orig_w, orig_h)
            if not boxes:
                return None
            return max(boxes, key=lambda item: item[1])
        except Exception:
            return None

    def _parse_yolo_outputs(self, outputs, orig_w: int, orig_h: int):
        arr = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        arr = np.asarray(arr)

        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            return []

        # YOLOv8 exports are often (features, boxes); transpose to (boxes, features).
        if arr.shape[0] < arr.shape[1] and arr.shape[0] in (5, 6, 84, 85):
            arr = arr.T

        boxes = []
        for row in arr:
            if row.shape[0] < 5:
                continue

            x, y, w, h = [float(v) for v in row[:4]]
            confidence = self._row_confidence(row)
            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = self._xywh_to_xyxy(
                x, y, w, h, orig_w, orig_h, self.input_size
            )
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(((x1, y1, x2, y2), confidence))
        return boxes

    @staticmethod
    def _row_confidence(row: np.ndarray) -> float:
        if row.shape[0] == 5:
            return float(row[4])
        if row.shape[0] == 6:
            return float(row[4])

        objectness = float(row[4])
        class_scores = row[5:]
        class_conf = float(np.max(class_scores)) if len(class_scores) else 1.0
        return objectness * class_conf if objectness <= 1.0 else class_conf

    @staticmethod
    def _xywh_to_xyxy(
        x: float,
        y: float,
        w: float,
        h: float,
        orig_w: int,
        orig_h: int,
        input_size: int,
    ) -> BBox:
        # Normalized exports stay in [0, 1]; pixel exports are scaled from model input.
        if max(x, y, w, h) <= 1.5:
            x *= orig_w
            w *= orig_w
            y *= orig_h
            h *= orig_h
        else:
            scale_x = orig_w / float(input_size)
            scale_y = orig_h / float(input_size)
            x *= scale_x
            w *= scale_x
            y *= scale_y
            h *= scale_y

        x1 = int(round(x - w / 2))
        y1 = int(round(y - h / 2))
        x2 = int(round(x + w / 2))
        y2 = int(round(y + h / 2))
        return (
            max(0, min(orig_w, x1)),
            max(0, min(orig_h, y1)),
            max(0, min(orig_w, x2)),
            max(0, min(orig_h, y2)),
        )

    @staticmethod
    def _center_bbox(size: Tuple[int, int], crop_ratio: float) -> BBox:
        w, h = size
        side = int(min(w, h) * crop_ratio)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        return x1, y1, x2, y2

    @staticmethod
    def _pad_bbox(bbox: BBox, size: Tuple[int, int], padding_ratio: float) -> BBox:
        w, h = size
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w, x2 + pad_x),
            min(h, y2 + pad_y),
        )
