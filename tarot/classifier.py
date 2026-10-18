"""Getrainde classifier laden en een kaartuitsnede classificeren."""
from pathlib import Path

import cv2
import numpy as np
import timm
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255


def laad_model(pad: Path, device):
    """Geeft (model, klassen, hoogte, breedte) uit een checkpoint van train_classifier."""
    ckpt = torch.load(pad, map_location=device)
    model = timm.create_model(ckpt["model"], pretrained=False, num_classes=len(ckpt["klassen"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["klassen"], ckpt["hoogte"], ckpt["breedte"]


@torch.no_grad()
def voorspel(model, uitsnede_bgr: np.ndarray, hoogte: int, breedte: int, device, k=3):
    """Top-k als lijst van (klasse-index, kans)."""
    rgb = cv2.cvtColor(uitsnede_bgr, cv2.COLOR_BGR2RGB)
    # Lineair vergroten, net als de Downscale-stap in de training.
    rgb = cv2.resize(rgb, (breedte, hoogte), interpolation=cv2.INTER_LINEAR)
    x = (rgb.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
    kansen = model(x).softmax(dim=1)[0]
    p, idx = kansen.topk(k)
    return list(zip(idx.tolist(), p.tolist()))
