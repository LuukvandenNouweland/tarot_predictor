"""Live kaartherkenning met detector: kaart vinden, rechttrekken, classificeren.

Webcam -> YOLO-pose (4 hoekpunten) -> perspectiefcorrectie -> classifier.
De detector draait op elk frame (omtrek volgt de kaart), de classifier
elke --interval seconden. Rechtsboven staat de rechtgetrokken uitsnede:
dat is precies wat de classifier ziet. Stoppen: q of Esc.

Gebruik:
    python webcam_detectie.py
    python webcam_detectie.py --interval 0.5 --min-conf 0.6
    python webcam_detectie.py --foto example/WIN_....jpg     # test op een foto

Nodig op de webcam-pc:
    pip install torch timm ultralytics opencv-python
    models/tarot_classifier.pt en models/kaart_detector.pt
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from webcam import laad_model, voorspel


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


def tekst(beeld, regel, pos, grootte=0.7, dikte=1):
    cv2.putText(beeld, regel, pos, cv2.FONT_HERSHEY_SIMPLEX, grootte, (0, 0, 0), dikte + 3)
    cv2.putText(beeld, regel, pos, cv2.FONT_HERSHEY_SIMPLEX, grootte, (255, 255, 255), dikte)


def teken(frame, detectie, top, klassen, uitsnede, venster_breedte: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    schaal = venster_breedte / fw
    beeld = cv2.resize(frame, (venster_breedte, int(fh * schaal)))

    if detectie is None:
        tekst(beeld, "geen kaart", (10, 35), 0.9, 2)
        return beeld

    hoeken, conf = detectie
    cv2.polylines(beeld, [(hoeken * schaal).astype(np.int32)], True, (0, 255, 0), 2)
    tekst(beeld, f"kaart {conf:.0%}", (10, int(beeld.shape[0]) - 12), 0.6)
    for r, (i, p) in enumerate(top):
        grootte, dikte = (0.9, 2) if r == 0 else (0.7, 1)
        tekst(beeld, f"{klassen[i].replace('_', ' ')}  {p:.0%}", (10, 35 + r * 32), grootte, dikte)

    if uitsnede is not None:  # rechtsboven: wat de classifier ziet
        h = beeld.shape[0] // 2
        klein = cv2.resize(uitsnede, (int(h * uitsnede.shape[1] / uitsnede.shape[0]), h))
        beeld[5:5 + h, -klein.shape[1] - 5:-5] = klein
    return beeld


def verwerk(frame, detector, classifier, klassen, hoogte, breedte, device, min_conf):
    detectie = detecteer(detector, frame, min_conf)
    if detectie is None:
        return None, [], None
    uitsnede = recht_trekken(frame, detectie[0], hoogte, breedte)
    return detectie, voorspel(classifier, uitsnede, hoogte, breedte, device), uitsnede


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=Path, default=Path("models/tarot_classifier.pt"))
    parser.add_argument("--detector", type=Path, default=Path("models/kaart_detector.pt"))
    parser.add_argument("--interval", type=float, default=1.0, help="seconden tussen classificaties")
    parser.add_argument("--min-conf", type=float, default=0.5, help="minimale detector-confidence")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--venster-breedte", type=int, default=960)
    parser.add_argument("--foto", type=Path, default=None,
                        help="geen webcam: verwerk deze foto en sla het resultaat op in previews/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, klassen, hoogte, breedte = laad_model(args.classifier, device)
    detector = YOLO(str(args.detector))
    print(f"Modellen geladen ({len(klassen)} kaarten) op {device}")

    if args.foto is not None:
        frame = cv2.imdecode(np.fromfile(args.foto, dtype=np.uint8), cv2.IMREAD_COLOR)
        detectie, top, uitsnede = verwerk(frame, detector, classifier, klassen, hoogte, breedte,
                                          device, args.min_conf)
        if detectie is None:
            print("Geen kaart gevonden")
        else:
            print(f"kaart {detectie[1]:.0%}  |  " + "  ".join(f"{klassen[i]} {p:.0%}" for i, p in top))
        uit = Path("previews") / f"detectie_{args.foto.stem}.jpg"
        uit.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(uit), teken(frame, detectie, top, klassen, uitsnede, args.venster_breedte))
        print(f"Resultaat opgeslagen: {uit}")
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Kan camera {args.camera} niet openen; probeer --camera 1")

    top, uitsnede, laatste = [], None, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Geen beeld van de camera")
            break

        detectie = detecteer(detector, frame, args.min_conf)
        if detectie is None:
            top, uitsnede = [], None
        elif time.time() - laatste >= args.interval:
            laatste = time.time()
            uitsnede = recht_trekken(frame, detectie[0], hoogte, breedte)
            top = voorspel(classifier, uitsnede, hoogte, breedte, device)
            print(f"kaart {detectie[1]:.0%}  |  " + "  ".join(f"{klassen[i]} {p:.0%}" for i, p in top))

        cv2.imshow("Tarot herkenner (q = stoppen)",
                   teken(frame, detectie, top, klassen, uitsnede, args.venster_breedte))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
