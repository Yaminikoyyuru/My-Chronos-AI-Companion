import os
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Box = Tuple[int, int, int, int]


# --------------------------------------------------------------------
# Face detector (YuNet DNN preferred, Haar cascade as fallback)
# --------------------------------------------------------------------
def build_face_detector():
    """Returns detect(frame, gray) -> list of (x, y, w, h), or None if no model found."""
    onnx_path = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")
    if os.path.exists(onnx_path) and hasattr(cv2, "FaceDetectorYN"):
        try:
            yunet = cv2.FaceDetectorYN.create(onnx_path, "", (320, 320), 0.7)

            def detect_yunet(frame, gray, d=yunet):
                h, w = frame.shape[:2]
                d.setInputSize((w, h))
                _, found = d.detect(frame)
                if found is None:
                    return []
                return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in found]

            print("Face detector: YuNet (OpenCV DNN)")
            return detect_yunet
        except Exception as e:
            print("YuNet load failed:", e)

    candidates = []
    try:
        candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    except Exception:
        pass
    candidates.append(os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml"))
    for path in candidates:
        if os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                print(f"Face detector: Haar cascade ({path})")

                def detect_haar(frame, gray, c=cascade):
                    return [tuple(int(v) for v in f)
                            for f in c.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))]
                return detect_haar
    return None


# --------------------------------------------------------------------
# Motion analysis (dense optical flow)
# --------------------------------------------------------------------
class MotionAnalyzer:
    """Returns the fraction of pixels moving faster than mag_thr (px per frame @160x120)."""

    def __init__(self, size=(160, 120), mag_thr=0.5):
        self.size = size
        self.mag_thr = mag_thr
        self.prev = None
        self.use_flow = True

    def update(self, gray) -> float:
        small = cv2.GaussianBlur(cv2.resize(gray, self.size), (5, 5), 0)
        if self.prev is None:
            self.prev = small
            return 0.0
        frac = None
        if self.use_flow:
            try:
                flow = cv2.calcOpticalFlowFarneback(self.prev, small, None, 0.5, 2, 15, 2, 5, 1.1, 0)
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                frac = float(np.count_nonzero(mag > self.mag_thr)) / mag.size
            except Exception:
                self.use_flow = False  # fall back to frame differencing
        if frac is None:
            diff = cv2.absdiff(self.prev, small)
            frac = float(np.count_nonzero(diff > 20)) / diff.size
        self.prev = small
        return frac


# --------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------
@dataclass
class Perception:
    t: float
    faces: List[Box]
    motion: float
    immobile_s: float
    face_lost_s: float
    fall_suspected: bool
    trigger: Optional[str] = None


class PerceptionPipeline:
    """Turns raw frames into incident triggers:
    FALL_SUSPECTED, IMMOBILE, FACE_LOST (or None)."""

    def __init__(self, detector, immobile_s=30.0, face_lost_s=15.0,
                 motion_thr=0.01, min_presence_s=5.0):
        self.detector = detector
        self.immobile_thr = immobile_s
        self.face_lost_thr = face_lost_s
        self.motion_thr = motion_thr
        self.min_presence_s = min_presence_s
        self.motion_an = MotionAnalyzer()
        self.reset()

    def reset(self):
        self.last_t = None
        self.last_face_t = None
        self.present_s = 0.0
        self.immobile_s = 0.0
        self.face_lost_s = 0.0
        self.last_cy_ratio = None
        self.hist = deque()
        self.drop_at = None
        self.fall_hold = 0.0
        self.armed = True

    def rearm(self):
        self.reset()

    def process(self, frame, t: Optional[float] = None) -> Perception:
        t = time.time() if t is None else t
        dt = 0.0 if self.last_t is None else max(0.0, min(t - self.last_t, 1.0))
        self.last_t = t

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(frame, gray)
        motion = self.motion_an.update(gray)
        frame_h = frame.shape[0]
        face = max(faces, key=lambda f: f[2] * f[3]) if faces else None

        if face:
            self.last_face_t = t
            cy = face[1] + face[3] / 2.0
            self.last_cy_ratio = cy / frame_h
            self.hist.append((t, cy, face[3]))
        while self.hist and t - self.hist[0][0] > 1.5:
            self.hist.popleft()

        present = self.last_face_t is not None and (t - self.last_face_t) < 1.5

        # presence, immobility, face-lost timers
        if present:
            self.present_s += dt
            self.face_lost_s = 0.0
            if motion < self.motion_thr:
                self.immobile_s += dt
            else:
                self.immobile_s = 0.0
        else:
            self.immobile_s = 0.0
            if self.present_s >= self.min_presence_s:
                self.face_lost_s += dt

        # fall heuristic: fast downward face movement, then absent or still
        if face and len(self.hist) >= 2:
            ys = [y for _, y, _ in self.hist]
            drop = ys[-1] - min(ys)
            if drop > 1.2 * face[3] and motion > self.motion_thr * 3:
                self.drop_at = t
        fall_cond = False
        if self.drop_at is not None:
            since = t - self.drop_at
            if since > 5.0:
                self.drop_at = None
            elif since >= 1.0 and (not present or motion < self.motion_thr):
                fall_cond = True
        self.fall_hold = self.fall_hold + dt if fall_cond else 0.0
        fall = self.fall_hold >= 1.5

        trigger = None
        if self.armed:
            if fall:
                trigger = "FALL_SUSPECTED"
            elif self.immobile_s >= self.immobile_thr:
                trigger = "IMMOBILE"
            elif (self.face_lost_s >= self.face_lost_thr
                  and (self.last_cy_ratio or 0.0) > 0.55):
                trigger = "FACE_LOST"
            if trigger:
                self.armed = False

        return Perception(t, faces, motion, self.immobile_s, self.face_lost_s, fall, trigger)