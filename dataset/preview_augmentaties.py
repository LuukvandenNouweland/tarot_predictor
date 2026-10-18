"""Preview van de classifier-augmentaties (tarot/augmentaties.py).

Gebruik (vanuit de projectmap):
    python -m dataset.preview_augmentaties            # previews/augment_preview.jpg
    python -m dataset.preview_augmentaties --n 8      # meer voorbeelden per kaart
    python -m dataset.preview_augmentaties --genereer # vaste dataset in data/classifier/
                                                      # (niet nodig: train_classifier augmenteert zelf)
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from tarot.augmentaties import build_transform
from tarot.beeld import lees_rgb, schrijf_jpg
from tarot.config import BREEDTE, CLASSIFIER_DATA_DIR, HOOGTE, KAARTEN_DIR, PREVIEWS_DIR


def maak_preview(n_kaarten: int, n_per_kaart: int, uit: Path) -> None:
    transform = build_transform()
    paden = random.sample(sorted(KAARTEN_DIR.glob("*.jpg")), n_kaarten)
    rijen = []
    for pad in paden:
        img = lees_rgb(pad)
        origineel = cv2.resize(img, (BREEDTE, HOOGTE))
        varianten = [transform(image=img)["image"] for _ in range(n_per_kaart)]
        rijen.append(np.hstack([origineel, *varianten]))
    schrijf_jpg(uit, cv2.cvtColor(np.vstack(rijen), cv2.COLOR_RGB2BGR))
    print(f"Preview opgeslagen: {uit} (eerste kolom = origineel)")


def genereer(uit: Path, n_train: int, n_val: int) -> None:
    """Schrijft uit/train/<kaart>/*.jpg en uit/val/<kaart>/*.jpg (ImageFolder-indeling)."""
    transform = build_transform()
    paden = sorted(KAARTEN_DIR.glob("*.jpg"))
    for i, pad in enumerate(paden, 1):
        img = lees_rgb(pad)
        for split, n in (("train", n_train), ("val", n_val)):
            for j in range(n):
                aug = cv2.cvtColor(transform(image=img)["image"], cv2.COLOR_RGB2BGR)
                schrijf_jpg(uit / split / pad.stem / f"{j:04d}.jpg", aug)
        print(f"[{i}/{len(paden)}] {pad.stem}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genereer", action="store_true", help="dataset maken i.p.v. preview")
    parser.add_argument("--kaarten", type=int, default=4)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--uit", type=Path, default=None)
    args = parser.parse_args()
    if args.genereer:
        genereer(args.uit or CLASSIFIER_DATA_DIR, args.n_train, args.n_val)
    else:
        maak_preview(args.kaarten, args.n, args.uit or PREVIEWS_DIR / "augment_preview.jpg")
