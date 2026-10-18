# Tarot herkenner

Herkent tarotkaarten (Rider-Waite, 1920s Pam B) die voor een webcam worden gehouden.

```
webcam → detector (YOLO-pose: 4 hoekpunten) → rechttrekken → classifier (EfficientNet-B0, 78 kaarten)
```

Beide modellen zijn alleen op **synthetische** data getraind: 78 scans met webcam-achtige
augmentaties (wazig, lage resolutie, kleurzweem, ruis), en voor de detector kaarten op
achtergronden van de eigen kamer.

## Mappen

| Map | Inhoud |
|---|---|
| `tarot/` | gedeelde code: `config.py` (alle paden), augmentaties, classifier, detector |
| `dataset/` | scripts die data maken |
| `training/` | scripts die modellen trainen |
| `app/` | de webcam-apps (OpenCV-venster) |
| `backend/` | FastAPI-server: camera, herkenning, MJPEG-stream en WebSocket |
| `frontend/` | webpagina met video en 3D-personage (Vite + TypeScript + three.js) |
| `models/` | getrainde modellen (in git) |
| `data/kaarten/` | de 78 scans; bestandsnaam = klassenaam (in git) |
| `data/achtergronden/`, `data/voorbeelden/`, `data/yolo/` | lokaal, niet in git |
| `runs/`, `previews/` | trainingslogs en previewplaatjes, niet in git |

## Installeren

Python 3.12. Training-pc (NVIDIA-GPU):

```
py -V:3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Webcam-pc (CPU is genoeg): `pip install -r requirements-webcam.txt`

## Gebruik

Alle scripts draaien als module **vanuit de projectmap** (`python -m map.script`),
zodat ze `tarot/` kunnen vinden.

### Webfrontend (video + 3D-personage)

Productie, één commando (de gebouwde frontend staat in `frontend/dist/`, Node is niet nodig):

```
python -m backend.main                         # open http://127.0.0.1:8000
python -m backend.main --bron 1                # andere camera
python -m backend.main --bron data/voorbeelden/foto.jpg   # testen zonder webcam
python -m backend.main --stabiel 8 --weg 5 --min-kans 0.6 # flikkeren tegengaan
python -m backend.main --host 0.0.0.0          # ook bereikbaar vanaf telefoon/tablet in het netwerk
```

Een kaart wordt pas gemeld na `--stabiel` frames op rij (standaard 5) met een
classifier-kans ≥ `--min-kans`; "geen kaart" na `--weg` frames zonder kaart.
Testpagina zonder frontend: http://127.0.0.1:8000/test

Ontwikkelen aan de frontend (Node 24 LTS, Vite-proxy naar de backend op :8000):

```
python -m backend.main                         # terminal 1
cd frontend
npm install                                    # eenmalig
npm run dev                                    # terminal 2, open http://localhost:5173
npm run build                                  # klaar: nieuwe build in frontend/dist/ (in git)
```

Animaties instellen: `frontend/src/animaties.ts` (welke animatie bij idle, een
nieuwe kaart, kaart weg, of per kaart). Daarna `npm run build`.

### Webcam (OpenCV-venster)

```
python -m app.webcam_detectie                  # met detector: kaart mag overal in beeld
python -m app.webcam                           # zonder detector: kaart in het groene kader
python -m app.webcam_detectie --foto data/voorbeelden/foto.jpg   # test op een foto
```

### Classifier trainen

```
python -m dataset.preview_augmentaties         # augmentaties bekijken: previews/augment_preview.jpg
python -m training.train_classifier            # → models/tarot_classifier.pt
```

Echte webcam-uitsneden in `data/test_echt/<kaartnaam>/*.jpg` worden elke epoch mee getest.

### Detector trainen

```
python -m dataset.video_naar_frames            # video's in data/achtergronden/ → frames
python -m dataset.maak_yolo_dataset            # preview: previews/yolo_preview.jpg
python -m dataset.maak_yolo_dataset --genereer # → data/yolo/
python -m training.train_detector              # → runs/kaart_pose/weights/best.pt
python -m training.train_detector --installeer runs/kaart_pose/weights/best.pt   # → models/kaart_detector.pt
```

Onderbroken: `python -m training.train_detector --hervat runs/kaart_pose/weights/last.pt`
