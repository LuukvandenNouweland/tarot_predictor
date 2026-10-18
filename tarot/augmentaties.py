"""Webcam-achtige augmentaties voor de tarotkaart-classifier.

Bekijken: python -m dataset.preview_augmentaties
"""
import albumentations as A
import cv2

from tarot.config import BREEDTE, HOOGTE


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
