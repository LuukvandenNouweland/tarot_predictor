"""Kaartdetector trainen: YOLO-pose die de 4 hoekpunten van een kaart vindt.

Eerst de dataset maken:
    python maak_yolo_dataset.py --genereer
Dan trainen:
    python train_yolo.py
Resultaat: runs_yolo/kaart_pose/weights/best.pt

Nodig: pip install ultralytics
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset_yolo/kaarten.yaml"))
    parser.add_argument("--model", default="yolo11n-pose.pt", help="voorgetraind op COCO")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        imgsz=320,          # webcam is 320x180; wordt aangevuld tot 320x320
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        project="runs_yolo",
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
