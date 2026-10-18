"""Tarotkaart-classifier trainen met augmentaties die elke epoch opnieuw gemaakt worden.

Gebruik (vanuit de projectmap):
    python -m training.train_classifier                       # standaardinstellingen
    python -m training.train_classifier --epochs 30 --model convnext_tiny

Optioneel: zet echte webcam-uitsneden in data/test_echt/<kaartnaam>/*.jpg
(kaartnaam zoals de bestandsnaam van de scan, bv. data/test_echt/the_moon/1.jpg).
Die worden dan elke epoch ook getest; dat is de score die er echt toe doet.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tarot.augmentaties import build_transform
from tarot.beeld import lees_rgb
from tarot.config import BREEDTE, CLASSIFIER_PAD, HOOGTE, KAARTEN_DIR, TEST_ECHT_DIR

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def laad_scans() -> tuple[list[str], list[np.ndarray]]:
    # Eén keer inlezen en verkleinen; de Resize in build_transform is dan gratis.
    paden = sorted(KAARTEN_DIR.glob("*.jpg"))
    klassen = [p.stem for p in paden]
    scans = [cv2.resize(lees_rgb(p), (BREEDTE, HOOGTE), interpolation=cv2.INTER_AREA)
             for p in paden]
    return klassen, scans


class KaartDataset(Dataset):
    """Elke __getitem__ maakt een nieuwe willekeurige variant van kaart i % 78."""

    def __init__(self, scans: list[np.ndarray], per_kaart: int):
        self.scans = scans
        self.per_kaart = per_kaart
        self.transform = build_transform()

    def __len__(self):
        return len(self.scans) * self.per_kaart

    def __getitem__(self, i):
        label = i % len(self.scans)
        img = self.transform(image=self.scans[label])["image"]
        return torch.from_numpy(img).permute(2, 0, 1), label  # uint8 CHW


def worker_init(_):
    # Op Windows krijgt elke worker een kopie van de transform met dezelfde
    # random-toestand; zonder dit maken alle workers dezelfde varianten.
    info = torch.utils.data.get_worker_info()
    info.dataset.transform.set_random_seed(torch.initial_seed() % 2**32)


def maak_val_set(scans: list[np.ndarray], per_kaart: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Vaste (geseede) set augmentaties, zodat de val-score tussen epochs vergelijkbaar is."""
    transform = build_transform()
    transform.set_random_seed(1234)
    beelden, labels = [], []
    for label, scan in enumerate(scans):
        for _ in range(per_kaart):
            beelden.append(transform(image=scan)["image"])
            labels.append(label)
    return torch.from_numpy(np.stack(beelden)).permute(0, 3, 1, 2), torch.tensor(labels)


def laad_test_echt(klassen: list[str]) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not TEST_ECHT_DIR.is_dir():
        return None
    beelden, labels = [], []
    for pad in sorted(TEST_ECHT_DIR.glob("*/*")):
        if pad.parent.name not in klassen:
            print(f"Let op: onbekende kaartnaam in {pad}, overgeslagen")
            continue
        img = cv2.resize(lees_rgb(pad), (BREEDTE, HOOGTE), interpolation=cv2.INTER_AREA)
        beelden.append(img)
        labels.append(klassen.index(pad.parent.name))
    if not beelden:
        return None
    print(f"{len(beelden)} echte webcam-uitsneden gevonden in {TEST_ECHT_DIR}")
    return torch.from_numpy(np.stack(beelden)).permute(0, 3, 1, 2), torch.tensor(labels)


def normaliseer(x: torch.Tensor) -> torch.Tensor:
    # Op de GPU i.p.v. in de workers: de CPU heeft het al druk met augmenteren.
    return (x.float() - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


@torch.no_grad()
def evalueer(model, beelden: torch.Tensor, labels: torch.Tensor, device, batch=128):
    model.eval()
    top1 = top5 = 0
    for s in range(0, len(beelden), batch):
        x = normaliseer(beelden[s:s + batch].to(device))
        y = labels[s:s + batch].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        beste5 = logits.topk(5, dim=1).indices
        top1 += (beste5[:, 0] == y).sum().item()
        top5 += (beste5 == y[:, None]).any(dim=1).sum().item()
    return top1 / len(beelden), top5 / len(beelden)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--per-kaart", type=int, default=300, help="varianten per kaart per epoch")
    parser.add_argument("--val-per-kaart", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    # Gemeten: meer dan ~4 workers levert op Windows nauwelijks meer op, wel langere opstart.
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--uit", type=Path, default=CLASSIFIER_PAD)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    klassen, scans = laad_scans()
    print(f"{len(klassen)} kaarten geladen")
    val_x, val_y = maak_val_set(scans, args.val_per_kaart)
    test_echt = laad_test_echt(klassen)

    loader = DataLoader(
        KaartDataset(scans, args.per_kaart), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, worker_init_fn=worker_init,
        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda", drop_last=True,
    )

    model = timm.create_model(args.model, pretrained=True, num_classes=len(klassen))
    # Geen channels_last: gemeten 9x trager voor EfficientNet op deze GPU.
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    args.uit.parent.mkdir(parents=True, exist_ok=True)
    beste = -1.0
    print("starting training")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch: {epoch}")
        model.train()
        start, totaal_loss = time.time(), 0.0
        for stap, (x, y) in enumerate(loader, 1):
            x = normaliseer(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            totaal_loss += loss.item()
            if stap % 25 == 0:
                verstreken = time.time() - start
                geheugen = (f", GPU-geheugen {torch.cuda.max_memory_reserved() / 2**30:.1f} GB"
                            if device.type == "cuda" else "")
                print(f"  stap {stap}/{len(loader)}  loss {loss.item():.3f}  "
                      f"{stap * args.batch / verstreken:.0f} img/s{geheugen}")

        val1, val5 = evalueer(model, val_x, val_y, device)
        regel = (f"epoch {epoch:2d}/{args.epochs}  loss {totaal_loss / len(loader):.3f}  "
                 f"val top1 {val1:.3f} top5 {val5:.3f}")
        score = val1
        if test_echt is not None:
            echt1, echt5 = evalueer(model, *test_echt, device)
            regel += f"  | ECHT top1 {echt1:.3f} top5 {echt5:.3f}"
            score = echt1  # echte webcambeelden zijn de enige test die telt
        regel += f"  ({time.time() - start:.0f}s)"

        if score >= beste:
            beste = score
            torch.save({"model": args.model, "state_dict": model.state_dict(), "klassen": klassen,
                        "hoogte": HOOGTE, "breedte": BREEDTE}, args.uit)
            regel += "  *opgeslagen"
        print(regel)

    print(f"Klaar. Beste model staat in {args.uit}")


if __name__ == "__main__":
    main()
