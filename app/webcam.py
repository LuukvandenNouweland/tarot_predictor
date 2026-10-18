"""Live kaartherkenning via de webcam.

Houd de kaart in het groene kader. Elke --interval seconden wordt het kader
geclassificeerd; de top-3 staat in beeld en wordt ook in de terminal geprint.
Stoppen: q of Esc.

Gebruik:
    python webcam.py
    python webcam.py --interval 0.5 --camera 1 --kader 0.8

Nodig op de webcam-pc (zonder albumentations):
    pip install torch timm opencv-python
    (opencv-python, niet -headless: die kan geen venster tonen)
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255


def laad_model(pad: Path, device):
    ckpt = torch.load(pad, map_location=device)
    model = timm.create_model(ckpt["model"], pretrained=False, num_classes=len(ckpt["klassen"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["klassen"], ckpt["hoogte"], ckpt["breedte"]


def kader_coords(frame_h: int, frame_w: int, fractie: float, hoogte: int, breedte: int):
    """Gecentreerd kader met de verhouding van een kaart, fractie = deel van de framehoogte."""
    kh = int(frame_h * fractie)
    kw = int(kh * breedte / hoogte)
    y0, x0 = (frame_h - kh) // 2, (frame_w - kw) // 2
    return x0, y0, kw, kh


@torch.no_grad()
def voorspel(model, uitsnede_bgr: np.ndarray, hoogte: int, breedte: int, device, k=3):
    rgb = cv2.cvtColor(uitsnede_bgr, cv2.COLOR_BGR2RGB)
    # Lineair vergroten, net als de Downscale-stap in de training.
    rgb = cv2.resize(rgb, (breedte, hoogte), interpolation=cv2.INTER_LINEAR)
    x = (rgb.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
    kansen = model(x).softmax(dim=1)[0]
    p, idx = kansen.topk(k)
    return list(zip(idx.tolist(), p.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/tarot_classifier.pt"))
    parser.add_argument("--interval", type=float, default=1.0, help="seconden tussen voorspellingen")
    parser.add_argument("--camera", type=int, default=0, help="camera-index")
    parser.add_argument("--kader", type=float, default=0.85, help="kaderhoogte als deel van het beeld")
    parser.add_argument("--venster-breedte", type=int, default=960, help="weergavebreedte in pixels")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, klassen, hoogte, breedte = laad_model(args.model, device)
    print(f"Model geladen ({len(klassen)} kaarten) op {device}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Kan camera {args.camera} niet openen; probeer --camera 1")

    top = []
    laatste = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Geen beeld van de camera")
            break
        fh, fw = frame.shape[:2]
        x0, y0, kw, kh = kader_coords(fh, fw, args.kader, hoogte, breedte)

        if time.time() - laatste >= args.interval:
            laatste = time.time()
            top = voorspel(model, frame[y0:y0 + kh, x0:x0 + kw], hoogte, breedte, device)
            print("  ".join(f"{klassen[i]} {p:.0%}" for i, p in top))

        # Weergave vergroten: 320x180 is anders onleesbaar klein.
        schaal = args.venster_breedte / fw
        beeld = cv2.resize(frame, (args.venster_breedte, int(fh * schaal)))
        cv2.rectangle(beeld, (int(x0 * schaal), int(y0 * schaal)),
                      (int((x0 + kw) * schaal), int((y0 + kh) * schaal)), (0, 255, 0), 2)
        for r, (i, p) in enumerate(top):
            tekst = f"{klassen[i].replace('_', ' ')}  {p:.0%}"
            pos = (10, 35 + r * 32)
            grootte, dikte = (0.9, 2) if r == 0 else (0.7, 1)
            cv2.putText(beeld, tekst, pos, cv2.FONT_HERSHEY_SIMPLEX, grootte, (0, 0, 0), dikte + 3)
            cv2.putText(beeld, tekst, pos, cv2.FONT_HERSHEY_SIMPLEX, grootte, (255, 255, 255), dikte)

        cv2.imshow("Tarot herkenner (q = stoppen)", beeld)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
