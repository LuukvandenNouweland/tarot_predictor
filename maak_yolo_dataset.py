"""Synthetische dataset voor de kaartdetector (YOLO-pose: 4 hoekpunten per kaart).

Elke afbeelding is 320x180 (zoals de webcam): een kaartscan met willekeurige
rotatie en perspectief op een achtergrond, daarna webcam-effecten over het geheel.

Hoekpunten staan in beeldvolgorde: 0 = linksboven, 1 = rechtsboven,
2 = rechtsonder, 3 = linksonder (zoals de kaart in beeld staat). Een omgekeerde
kaart wordt na rechttrekken dus ook omgekeerd uitgesneden.

Achtergronden: worden procedureel gemaakt. Staan er foto's in achtergronden/
(bv. webcamframes van je kamer ZONDER kaart), dan worden die ook gebruikt.

Gebruik:
    python maak_yolo_dataset.py                  # preview: previews/yolo_preview.jpg
    python maak_yolo_dataset.py --genereer       # dataset_yolo/ + kaarten.yaml
"""
import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

from augment import KAARTEN_DIR, lees_rgb

FRAME_B, FRAME_H = 320, 180
RENDER = 2  # eerst op 2x resolutie tekenen en dan verkleinen: nette randen
RB, RH = FRAME_B * RENDER, FRAME_H * RENDER
ACHTERGROND_DIR = Path("achtergronden")
FLIP_IDX = [1, 0, 3, 2]  # bij horizontaal spiegelen wisselen links en rechts
HUIDKLEUREN = np.array([[224, 172, 105], [198, 134, 66], [141, 85, 36], [255, 219, 172]], np.float32)


def laad_kaarten() -> list[tuple[np.ndarray, np.ndarray]]:
    """Scans (verkleind) met een alfamasker met afgeronde hoeken."""
    kaarten = []
    for pad in sorted(KAARTEN_DIR.glob("*.jpg")):
        img = lees_rgb(pad)
        h = 400
        img = cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)
        w = img.shape[1]
        r = int(0.05 * w)
        masker = np.zeros((h, w), np.uint8)
        cv2.rectangle(masker, (r, 0), (w - 1 - r, h - 1), 255, -1)
        cv2.rectangle(masker, (0, r), (w - 1, h - 1 - r), 255, -1)
        for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
            cv2.circle(masker, (cx, cy), r, 255, -1)
        kaarten.append((img, masker))
    return kaarten


def laad_foto_achtergronden() -> list[np.ndarray]:
    if not ACHTERGROND_DIR.is_dir():
        return []
    paden = [p for p in ACHTERGROND_DIR.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return [lees_rgb(p) for p in paden]


def procedurele_achtergrond(rng) -> np.ndarray:
    """Kamerachtig: zachte gradient, vlekken, vlakken (meubels/deuren), soms een lamp."""
    # Gedempte kleuren: echte kamers zijn vooral grijs/beige/bruin, niet paars of groen.
    grijs = rng.uniform(40, 200)
    c1 = grijs + rng.uniform(-20, 20, 3)
    c2 = grijs + rng.uniform(-20, 20, 3)
    t = np.linspace(0, 1, RB)[None, :, None] if rng.random() < 0.5 else np.linspace(0, 1, RH)[:, None, None]
    img = np.broadcast_to(c1 * (1 - t) + c2 * t, (RH, RB, 3)).astype(np.float32).copy()
    klein = rng.uniform(-40, 40, (rng.integers(2, 7), rng.integers(2, 7), 1)).astype(np.float32)
    img += cv2.resize(klein, (RB, RH), interpolation=cv2.INTER_CUBIC)[..., None]
    for _ in range(rng.integers(2, 9)):
        kleur = (rng.uniform(20, 230) + rng.uniform(-20, 20, 3)).tolist()
        x0, y0 = rng.integers(-RB // 4, RB), rng.integers(-RH // 4, RH)
        x1, y1 = x0 + rng.integers(RB // 10, RB // 2), y0 + rng.integers(RH // 10, RH)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), kleur, -1)
    if rng.random() < 0.3:  # lamp / raam: fel wit
        cv2.ellipse(img, (int(rng.integers(0, RB)), int(rng.integers(0, RH // 3))),
                    (int(rng.integers(10, 60)), int(rng.integers(10, 40))), 0, 0, 360, (255, 255, 250), -1)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(1, 6))
    return np.clip(img, 0, 255)


def foto_achtergrond(rng, foto: np.ndarray) -> np.ndarray:
    h, w = foto.shape[:2]
    ch = int(min(h, w * RH / RB) * rng.uniform(0.6, 1.0))
    cw = int(ch * RB / RH)
    y, x = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    return cv2.resize(foto[y:y + ch, x:x + cw], (RB, RH)).astype(np.float32)


def plaats_kaart(rng, kaart_b: int, kaart_h: int):
    """Kies hoekpunten in render-pixels; geeft (eigen hoeken, beeldvolgorde-hoeken) of None."""
    for _ in range(20):
        h = rng.uniform(0.4, 0.95) * RH
        w = h * kaart_b / kaart_h
        hoek = np.deg2rad(rng.uniform(-30, 30))
        omgekeerd = rng.random() < 0.5
        if omgekeerd:
            hoek += np.pi
        # Eigen hoeken van de kaart: LB, RB, RO, LO.
        p = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        p += rng.normal(0, 0.04 * h, p.shape)  # perspectief: kaart iets gekanteld
        rot = np.array([[np.cos(hoek), -np.sin(hoek)], [np.sin(hoek), np.cos(hoek)]])
        p = p @ rot.T
        marge = 2 * RENDER
        lo, hi = -p.min(0) + marge, np.array([RB, RH]) - p.max(0) - marge
        if np.any(lo > hi):
            continue  # past niet in beeld, opnieuw proberen
        p += rng.uniform(lo, hi)
        beeld = np.roll(p, -2, axis=0) if omgekeerd else p
        return p.astype(np.float32), beeld.astype(np.float32)
    return None


def belichting(rng, kaart: np.ndarray) -> np.ndarray:
    """Licht verloop en soms een glansvlek op de kaart zelf."""
    h, w = kaart.shape[:2]
    klein = rng.uniform(0.7, 1.25, (2, 2)).astype(np.float32)
    kaart = kaart.astype(np.float32) * cv2.resize(klein, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    if rng.random() < 0.25:
        glans = np.zeros((h, w), np.float32)
        cv2.ellipse(glans, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                    (int(rng.integers(w // 10, w // 3)), int(rng.integers(h // 20, h // 6))),
                    float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        glans = cv2.GaussianBlur(glans, (0, 0), w / 15) * rng.uniform(0.4, 0.9)
        kaart = kaart * (1 - glans[..., None]) + 255 * glans[..., None]
    return np.clip(kaart, 0, 255)


def teken_hand(rng, img: np.ndarray, beeld_hoeken: np.ndarray) -> None:
    """Hand over de onderrand van de kaart, met een arm naar de onderkant van het beeld."""
    onder = (beeld_hoeken[2] + beeld_hoeken[3]) / 2
    breedte = np.linalg.norm(beeld_hoeken[2] - beeld_hoeken[3])
    masker = np.zeros(img.shape[:2], np.float32)
    centrum = onder + np.array([rng.uniform(-0.4, 0.4) * breedte, rng.uniform(-0.05, 0.15) * breedte])
    hand_b = breedte * rng.uniform(0.35, 0.7)
    cv2.ellipse(masker, (int(centrum[0]), int(centrum[1])),
                (int(hand_b), int(breedte * rng.uniform(0.2, 0.4))),
                float(rng.uniform(-30, 30)), 0, 360, 1.0, -1)
    # Pols/arm: van de hand schuin naar de onderrand van het beeld.
    arm_x = centrum[0] + rng.uniform(-1, 1) * breedte
    arm = np.array([[centrum[0] - hand_b * 0.6, centrum[1]], [centrum[0] + hand_b * 0.6, centrum[1]],
                    [arm_x + hand_b * 0.8, RH], [arm_x - hand_b * 0.8, RH]], np.int32)
    cv2.fillPoly(masker, [arm], 1.0)
    masker = cv2.GaussianBlur(masker, (0, 0), 2 * RENDER)[..., None]
    kleur = HUIDKLEUREN[rng.integers(len(HUIDKLEUREN))] * rng.uniform(0.6, 1.1)
    img[:] = img * (1 - masker) + kleur * masker


def webcam_effecten() -> A.Compose:
    # Alleen fotometrisch: de hoekpunten blijven kloppen.
    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=(3, 7)),
            A.Defocus(radius=(1, 2)),
        ], p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=(-30, 30), val_shift_limit=10, p=0.5),
        A.PlanckianJitter(p=0.4),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
        A.RandomGamma(gamma_limit=(70, 140), p=0.3),
        A.OneOf([A.GaussNoise(std_range=(0.01, 0.05)), A.ISONoise()], p=0.5),
        A.ImageCompression(quality_range=(40, 90), p=0.6),
    ])


def maak_voorbeeld(rng, kaarten, fotos, effecten, p_leeg=0.1):
    """Eén 320x180-beeld + hoekpunten (4x2, in frame-pixels) of None als er geen kaart is."""
    if fotos and rng.random() < 0.5:
        img = foto_achtergrond(rng, fotos[rng.integers(len(fotos))])
    else:
        img = procedurele_achtergrond(rng)

    hoeken = None
    if rng.random() >= p_leeg:
        kaart, masker = kaarten[rng.integers(len(kaarten))]
        plaatsing = plaats_kaart(rng, kaart.shape[1], kaart.shape[0])
        if plaatsing is not None:
            eigen, beeld = plaatsing
            h, w = kaart.shape[:2]
            M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), eigen)
            kaart_w = cv2.warpPerspective(belichting(rng, kaart), M, (RB, RH), flags=cv2.INTER_LINEAR)
            alfa = cv2.warpPerspective(masker, M, (RB, RH), flags=cv2.INTER_LINEAR)[..., None] / 255.0
            img = img * (1 - alfa) + kaart_w * alfa
            if rng.random() < 0.4:
                teken_hand(rng, img, beeld)
            hoeken = beeld / RENDER

    img = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (FRAME_B, FRAME_H), interpolation=cv2.INTER_AREA)
    return effecten(image=img)["image"], hoeken


def yolo_label(hoeken) -> str:
    """class cx cy w h  x1 y1 v1 ... x4 y4 v4, alles genormaliseerd."""
    if hoeken is None:
        return ""
    n = hoeken / np.array([FRAME_B, FRAME_H])
    x0, y0 = n.min(0)
    x1, y1 = n.max(0)
    kp = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in n)
    return f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f} {kp}\n"


def maak_preview(aantal: int, uit: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    effecten = webcam_effecten()
    if seed is not None:
        effecten.set_random_seed(seed)
    kaarten, fotos = laad_kaarten(), laad_foto_achtergronden()
    kleuren = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 0, 255)]  # RGB per hoekpunt
    tegels = []
    for _ in range(aantal):
        img, hoeken = maak_voorbeeld(rng, kaarten, fotos, effecten)
        groot = cv2.resize(img, (FRAME_B * 2, FRAME_H * 2), interpolation=cv2.INTER_NEAREST)
        if hoeken is not None:
            p = (hoeken * 2).astype(np.int32)
            cv2.polylines(groot, [p], True, (255, 255, 0), 1)
            for i, (x, y) in enumerate(p):
                cv2.circle(groot, (int(x), int(y)), 5, kleuren[i], -1)
                cv2.putText(groot, str(i), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, kleuren[i], 2)
        else:
            cv2.putText(groot, "geen kaart", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        tegels.append(cv2.copyMakeBorder(groot, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    kolommen = 3
    while len(tegels) % kolommen:
        tegels.append(np.zeros_like(tegels[0]))
    rijen = [np.hstack(tegels[i:i + kolommen]) for i in range(0, len(tegels), kolommen)]
    uit.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(uit), cv2.cvtColor(np.vstack(rijen), cv2.COLOR_RGB2BGR))
    print(f"Preview opgeslagen: {uit} ({len(fotos)} foto-achtergronden gevonden)")
    print("Hoekpunten: 0 = rood (linksboven), 1 = groen, 2 = blauw, 3 = paars")


def genereer(uit: Path, n_train: int, n_val: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    effecten = webcam_effecten()
    effecten.set_random_seed(seed)
    kaarten, fotos = laad_kaarten(), laad_foto_achtergronden()
    print(f"{len(kaarten)} kaarten, {len(fotos)} foto-achtergronden")
    for split, n in (("train", n_train), ("val", n_val)):
        (uit / "images" / split).mkdir(parents=True, exist_ok=True)
        (uit / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, hoeken = maak_voorbeeld(rng, kaarten, fotos, effecten)
            naam = f"{i:05d}"
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            buf.tofile(uit / "images" / split / f"{naam}.jpg")
            (uit / "labels" / split / f"{naam}.txt").write_text(yolo_label(hoeken))
            if (i + 1) % 1000 == 0:
                print(f"  {split}: {i + 1}/{n}")
    yaml = (f"path: {uit.resolve().as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "kpt_shape: [4, 3]\n"
            f"flip_idx: {FLIP_IDX}\n"
            "names:\n"
            "  0: kaart\n")
    (uit / "kaarten.yaml").write_text(yaml)
    print(f"Klaar: {uit}/ en {uit / 'kaarten.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genereer", action="store_true", help="dataset maken i.p.v. preview")
    parser.add_argument("--aantal", type=int, default=12, help="aantal beelden in de preview")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--uit", type=Path, default=None)
    args = parser.parse_args()
    if args.genereer:
        genereer(args.uit or Path("dataset_yolo"), args.n_train, args.n_val, args.seed or 0)
    else:
        maak_preview(args.aantal, args.uit or Path("previews/yolo_preview.jpg"), args.seed)
