"""Kaartdetector (YOLO-pose, 4 hoekpunten) gebruiken en de kaart rechttrekken."""
import cv2
import numpy as np


def detecteer(detector, frame_bgr: np.ndarray, min_conf: float):
    """Beste kaart in beeld: (4x2 hoekpunten in framepixels, confidence) of None."""
    res = detector(frame_bgr, imgsz=320, conf=min_conf, verbose=False)[0]
    if res.keypoints is None or len(res.boxes) == 0:
        return None
    i = int(res.boxes.conf.argmax())
    return res.keypoints.xy[i].cpu().numpy().astype(np.float32), float(res.boxes.conf[i])


def recht_trekken(frame_bgr: np.ndarray, hoeken: np.ndarray, hoogte: int, breedte: int) -> np.ndarray:
    """Hoekpunten (LB, RB, RO, LO in beeld) naar een plat rechthoekig kaartbeeld."""
    doel = np.float32([[0, 0], [breedte, 0], [breedte, hoogte], [0, hoogte]])
    M = cv2.getPerspectiveTransform(hoeken, doel)
    return cv2.warpPerspective(frame_bgr, M, (breedte, hoogte), flags=cv2.INTER_LINEAR)
