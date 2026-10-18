from pathlib import Path

import cv2
import numpy as np


def lees_rgb(pad: Path) -> np.ndarray:
    # imdecode i.p.v. imread: werkt ook met niet-ASCII paden op Windows.
    img = cv2.imdecode(np.fromfile(pad, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def lees_bgr(pad: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(pad, dtype=np.uint8), cv2.IMREAD_COLOR)


def schrijf_jpg(pad: Path, img_bgr: np.ndarray, kwaliteit: int = 95) -> None:
    # imencode + tofile: ook niet-ASCII paden op Windows.
    pad.parent.mkdir(parents=True, exist_ok=True)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, kwaliteit])
    buf.tofile(pad)
