"""Webcam-achtige augmentaties voor de tarotkaart-classifier.

Gebruik:
    python augment.py            # schrijft previews/augment_preview.jpg
    python augment.py --n 8      # meer voorbeelden per kaart
"""
import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

KAARTEN_DIR = Path("Tarot_kaarten/1920s (Pam B)")
# Kaarten zijn ~0.6 breed/hoog; houd die verhouding aan.
HOOGTE, BREEDTE = 384, 224


def build_transform():
    return A.Compose([
        A.Resize(HOOGTE, BREEDTE),

        # Geometrie: de detector + rechttrekken is nooit perfect.
        # GEEN HorizontalFlip (gespiegelde kaarten bestaan niet) en geen
        # 180°-rotatie (dat is een aparte klasse "omgekeerd").
        A.Affine(rotate=(-8, 8), scale=(0.9, 1.1), translate_percent=(-0.05, 0.05),
                 border_mode=cv2.BORDER_REPLICATE, p=0.8),
        A.Perspective(scale=(0.02, 0.06), border_mode=cv2.BORDER_REPLICATE, p=0.5),

        # Optiek: onscherpte en lage resolutie van de webcam.
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 9)),
            A.Defocus(radius=(1, 3)),
        ], p=0.7),
        # Webcam (320x180) levert een kaart van ~90-160 px hoog: altijd naar
        # dat detailniveau omlaag en weer terug omhoog schalen.
        A.Downscale(scale_range=(90 / HOOGTE, 160 / HOOGTE),
                    interpolation_pair={"downscale": cv2.INTER_AREA,
                                        "upscale": cv2.INTER_LINEAR},
                    p=1.0),

        # Kleur en belichting: verzadiging, witbalans, helderheid.
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=(-20, 40),
                             val_shift_limit=10, p=0.7),
        A.PlanckianJitter(p=0.3),
        A.RandomGamma(gamma_limit=(70, 140), p=0.3),

        # Sensor en compressie.
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.06)),
            A.ISONoise(),
        ], p=0.5),
        A.ImageCompression(quality_range=(30, 85), p=0.5),

        # Vingers / glans die een stuk afdekken.
        # A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.25),
        #                 hole_width_range=(0.1, 0.25), fill="random_uniform", p=0.3),
    ])


def lees_rgb(pad: Path) -> np.ndarray:
    # imdecode i.p.v. imread: werkt ook met niet-ASCII paden op Windows.
    img = cv2.imdecode(np.fromfile(pad, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def maak_preview(n_kaarten: int, n_per_kaart: int, uit: Path) -> None:
    transform = build_transform()
    paden = random.sample(sorted(KAARTEN_DIR.glob("*.jpg")), n_kaarten)
    rijen = []
    for pad in paden:
        img = lees_rgb(pad)
        origineel = cv2.resize(img, (BREEDTE, HOOGTE))
        varianten = [transform(image=img)["image"] for _ in range(n_per_kaart)]
        rijen.append(np.hstack([origineel, *varianten]))
    uit.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(uit), cv2.cvtColor(np.vstack(rijen), cv2.COLOR_RGB2BGR))
    print(f"Preview opgeslagen: {uit} (eerste kolom = origineel)")


def genereer(uit: Path, n_train: int, n_val: int) -> None:
    """Schrijft uit/train/<kaart>/*.jpg en uit/val/<kaart>/*.jpg (ImageFolder-indeling)."""
    transform = build_transform()
    paden = sorted(KAARTEN_DIR.glob("*.jpg"))
    for i, pad in enumerate(paden, 1):
        img = lees_rgb(pad)
        for split, n in (("train", n_train), ("val", n_val)):
            map_ = uit / split / pad.stem
            map_.mkdir(parents=True, exist_ok=True)
            for j in range(n):
                aug = cv2.cvtColor(transform(image=img)["image"], cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
                buf.tofile(map_ / f"{j:04d}.jpg")
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
        genereer(args.uit or Path("dataset"), args.n_train, args.n_val)
    else:
        maak_preview(args.kaarten, args.n, args.uit or Path("previews/augment_preview.jpg"))
