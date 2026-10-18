"""Kaartdetector trainen: YOLO-pose die de 4 hoekpunten van een kaart vindt.

Gebruik (vanuit de projectmap):
    python -m dataset.maak_yolo_dataset --genereer    # eerst de dataset
    python -m training.train_detector
Resultaat: runs/kaart_pose/weights/best.pt; kopieer naar models/ met:
    python -m training.train_detector --installeer runs/kaart_pose/weights/best.pt
Onderbroken? Verder gaan met:
    python -m training.train_detector --hervat runs/kaart_pose/weights/last.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import strip_optimizer

from tarot.config import DETECTOR_PAD, RUNS_DIR, YOLO_DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=YOLO_DATA_DIR / "kaarten.yaml")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="voorgetraind op COCO")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hervat", type=Path, default=None,
                        help="onderbroken training voortzetten vanaf dit last.pt")
    parser.add_argument("--installeer", type=Path, default=None,
                        help="niet trainen: dit best.pt (zonder optimizer-stand) naar models/ kopiëren")
    args = parser.parse_args()

    if args.installeer is not None:
        strip_optimizer(str(args.installeer), s=str(DETECTOR_PAD))
        return

    if args.hervat is not None:
        # Neemt alle instellingen (epochs, map, optimizer-stand) over uit de onderbroken run.
        YOLO(str(args.hervat)).train(resume=True)
        return

    # Basismodel in runs/ downloaden i.p.v. in de map waar je toevallig staat.
    model = YOLO(args.model if Path(args.model).exists() else str(RUNS_DIR / args.model))
    model.train(
        data=str(args.data),
        imgsz=320,          # webcam is 320x180; wordt aangevuld tot 320x320
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        # Absoluut pad: een relatief pad zet ultralytics onder runs/pose/.
        project=str(RUNS_DIR),
        name="kaart_pose",
        exist_ok=True,
        # De dataset bevat al rotatie en perspectief; YOLO's eigen spiegelen
        # (met flip_idx uit de yaml) en mozaïek zijn prima, extra draaien niet:
        # dat zou de hoekvolgorde (0 = linksboven in beeld) verstoren.
        degrees=0,
        fliplr=0.5,
    )


if __name__ == "__main__":
    main()
