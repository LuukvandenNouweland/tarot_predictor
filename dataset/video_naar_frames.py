"""Frames uit achtergrondvideo's halen voor maak_yolo_dataset.py.

Neem je kamer op ZONDER kaart in beeld en zet de video in data/achtergronden/.
    python -m dataset.video_naar_frames
Schrijft data/achtergronden/frames/<video>_<nr>.jpg op 640x360 (de renderresolutie
van de dataset-generator; groter heeft geen zin en kost alleen geheugen).
"""
import argparse

import cv2

from tarot.config import ACHTERGROND_DIR

GROOTTE = (640, 360)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elke", type=int, default=1, help="alleen elke n-de frame bewaren")
    args = parser.parse_args()

    uit = ACHTERGROND_DIR / "frames"
    uit.mkdir(parents=True, exist_ok=True)
    for video in sorted(ACHTERGROND_DIR.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video))
        nr = bewaard = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if nr % args.elke == 0:
                frame = cv2.resize(frame, GROOTTE, interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                buf.tofile(uit / f"{video.stem}_{nr:05d}.jpg")
                bewaard += 1
            nr += 1
        cap.release()
        print(f"{video.name}: {bewaard} van {nr} frames opgeslagen in {uit}/")


if __name__ == "__main__":
    main()
