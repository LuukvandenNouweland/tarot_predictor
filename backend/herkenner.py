"""Camerathread: frames lezen, kaart detecteren + classificeren, resultaat bijhouden.

Python is de enige eigenaar van de camera; de webserver leest alleen de
laatste JPEG en de laatste status uit dit object.
"""
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from backend.stabiel import Stabilisator
from tarot.beeld import lees_bgr
from tarot.classifier import laad_model, voorspel
from tarot.config import CLASSIFIER_PAD, DETECTOR_PAD
from tarot.detector import detecteer, recht_trekken

FOTO_EXTENSIES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Instellingen:
    bron: str = "0"            # camera-index, of pad naar een video/foto (om te testen)
    stabiel_n: int = 5         # frames op rij voordat een nieuwe kaart gemeld wordt
    weg_n: int = 5             # frames zonder kaart voordat "geen kaart" gemeld wordt
    min_conf: float = 0.5      # minimale detector-confidence
    min_kans: float = 0.5      # minimale classifier-kans om een waarneming mee te tellen
    foto_fps: float = 15.0     # tempo waarin een foto/video als "camera" wordt afgespeeld
    jpeg_kwaliteit: int = 80


def nette_naam(kaart: str) -> str:
    """the_tower -> The Tower, three_of_cups -> Three of Cups."""
    return " ".join(w if w == "of" else w.capitalize() for w in kaart.split("_"))


class Bron:
    """Camera, videobestand (loopt rond) of foto (steeds hetzelfde frame)."""

    def __init__(self, bron: str, foto_fps: float):
        self.foto = None
        self.cap = None
        self.wacht = 0.0  # camera's bepalen hun eigen tempo
        pad = Path(bron)
        if bron.isdigit():
            self.cap = cv2.VideoCapture(int(bron))
        elif pad.suffix.lower() in FOTO_EXTENSIES:
            self.foto = lees_bgr(pad)
            self.wacht = 1 / foto_fps
        else:
            self.cap = cv2.VideoCapture(str(pad))
            self.wacht = 1 / (self.cap.get(cv2.CAP_PROP_FPS) or foto_fps)
        if self.foto is None and not self.cap.isOpened():
            raise RuntimeError(f"Kan bron {bron!r} niet openen")
        self.is_video = self.cap is not None and not bron.isdigit()

    def lees(self):
        if self.wacht:
            time.sleep(self.wacht)
        if self.foto is not None:
            return self.foto.copy()
        ok, frame = self.cap.read()
        if not ok and self.is_video:  # video opnieuw afspelen
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def sluit(self):
        if self.cap is not None:
            self.cap.release()


class Herkenner:
    def __init__(self, inst: Instellingen):
        self.inst = inst
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier, self.klassen, self.hoogte, self.breedte = laad_model(CLASSIFIER_PAD, self.device)
        self.detector = YOLO(str(DETECTOR_PAD))
        self.stabiel = Stabilisator(inst.stabiel_n, inst.weg_n)
        print(f"Modellen geladen ({len(self.klassen)} kaarten) op {self.device}")

        self._slot = threading.Lock()
        self._jpeg: bytes | None = None
        self.frame_nr = 0
        self.versie = 0  # +1 bij elke wijziging van de aangekondigde kaart
        self._status = self._leeg_status((0, 0))
        self._kans = 0.0
        self._draait = False
        self._thread: threading.Thread | None = None

    # --- publiek, thread-safe ---

    def start(self):
        self._draait = True
        self._thread = threading.Thread(target=self._lus, daemon=True, name="camera")
        self._thread.start()

    def stop(self):
        self._draait = False
        if self._thread is not None:
            self._thread.join(timeout=5)

    def laatste_jpeg(self) -> tuple[bytes | None, int]:
        with self._slot:
            return self._jpeg, self.frame_nr

    def status(self) -> tuple[dict, int]:
        with self._slot:
            return dict(self._status), self.versie

    # --- camerathread ---

    @staticmethod
    def _leeg_status(frame_size):
        return {"card": None, "label": None, "reversed": False, "confidence": 0.0,
                "box": None, "corners": None, "frame_size": list(frame_size)}

    def _lus(self):
        bron = Bron(self.inst.bron, self.inst.foto_fps)
        try:
            while self._draait:
                frame = bron.lees()
                if frame is None:
                    time.sleep(0.1)
                    continue
                self._verwerk(frame)
        finally:
            bron.sluit()

    def _verwerk(self, frame):
        fh, fw = frame.shape[:2]
        detectie = detecteer(self.detector, frame, self.inst.min_conf)

        waarneming, kans, hoeken = None, 0.0, None
        if detectie is not None:
            hoeken = detectie[0]
            uitsnede = recht_trekken(frame, hoeken, self.hoogte, self.breedte)
            [(idx, kans)] = voorspel(self.classifier, uitsnede, self.hoogte, self.breedte, self.device, k=1)
            if kans >= self.inst.min_kans:
                waarneming = self.klassen[idx]
        veranderd = self.stabiel.update(waarneming)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.inst.jpeg_kwaliteit])
        kaart = self.stabiel.aangekondigd
        if kaart is not None and waarneming == kaart:
            self._kans = kans  # kans van de laatste waarneming van déze kaart
        status = self._leeg_status((fw, fh))
        if kaart is not None:
            status.update(card=kaart, label=nette_naam(kaart), confidence=round(self._kans, 3))
        if hoeken is not None:
            x0, y0 = hoeken.min(axis=0)
            x1, y1 = hoeken.max(axis=0)
            status["box"] = [round(float(v), 1) for v in (x0, y0, x1, y1)]
            status["corners"] = [[round(float(x), 1), round(float(y), 1)] for x, y in hoeken]

        with self._slot:
            self._jpeg = buf.tobytes()
            self.frame_nr += 1
            self._status = status
            if veranderd:
                self.versie += 1
        if veranderd:
            print(f"Kaart: {status['label'] or '(geen)'}")
