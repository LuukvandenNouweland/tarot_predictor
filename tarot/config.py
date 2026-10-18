"""Alle paden en vaste maten op één plek.

Paden zijn absoluut (t.o.v. de projectmap), zodat scripts vanuit elke map werken.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = ROOT / "data"
KAARTEN_DIR = DATA_DIR / "kaarten"              # 78 scans, bestandsnaam = klassenaam
ACHTERGROND_DIR = DATA_DIR / "achtergronden"    # video's + frames van de kamer zonder kaart
VOORBEELDEN_DIR = DATA_DIR / "voorbeelden"      # losse webcamfoto's om mee te testen
TEST_ECHT_DIR = DATA_DIR / "test_echt"          # echte uitsneden: test_echt/<kaart>/*.jpg
YOLO_DATA_DIR = DATA_DIR / "yolo"               # gegenereerde detector-dataset
CLASSIFIER_DATA_DIR = DATA_DIR / "classifier"   # optionele vooraf gegenereerde classifier-dataset

# Modellen en uitvoer
MODELS_DIR = ROOT / "models"
CLASSIFIER_PAD = MODELS_DIR / "tarot_classifier.pt"
DETECTOR_PAD = MODELS_DIR / "kaart_detector.pt"
RUNS_DIR = ROOT / "runs"
PREVIEWS_DIR = ROOT / "previews"

# Classifier-invoer: kaarten zijn ~0.6 breed/hoog; houd die verhouding aan.
HOOGTE, BREEDTE = 384, 224

# Webcambeeld
FRAME_B, FRAME_H = 320, 180
