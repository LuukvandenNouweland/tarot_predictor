# CLAUDE.md

Tarotkaartherkenner: webcam → YOLO-pose-detector (4 hoekpunten) → perspectiefcorrectie →
EfficientNet-B0-classifier (78 kaarten) → FastAPI-backend → Vite/three.js-frontend met
3D-schedel. Zie README.md voor mappen, commando's en status.

## Werkwijze en conventies

- Code, commentaar, namen en CLI-opties zijn **Nederlands** (`laad_model`, `--bron`,
  `--hervat`). Kaartnamen/labels zijn Engels (`the_tower` → "The Tower").
- Scripts draaien als module vanuit de projectmap: `python -m training.train_classifier`.
  Scripts importeren alleen uit `tarot/`, nooit uit elkaar. Alle paden staan in
  `tarot/config.py` (absoluut t.o.v. de projectmap).
- `tarot/__init__.py` blijft leeg: de webcam-pc heeft geen albumentations.
- Python: `.venv` met Python 3.12 (`.\.venv\Scripts\python`). De systeem-Python is 3.15;
  daarvoor bestaan geen torch/albumentations-wheels.
- `frontend/dist/` staat bewust in git (webcam-pc heeft geen Node). Na elke wijziging in
  `frontend/src/`: `npm run build` en `dist/` meecommitten. Een wijziging in `src/` zonder
  build is op poort 8000 niet zichtbaar.
- In een sessie die al liep vóór de Node-installatie: PATH verversen met
  `$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")`.
- `frontend/tsconfig.json` heeft `erasableSyntaxOnly`: geen parameter properties
  (`constructor(private x)`) of enums.

## Twee pc's

- **Training-pc** (deze): RTX 3060 Ti, i7-12700K, **geen webcam**. Testen met
  `python -m backend.main --bron data/voorbeelden/<foto>.jpg` of
  `python -m app.webcam_detectie --foto ...`.
- **Webcam-pc**: draait alleen `backend/` en `app/`, via `git pull` en
  `requirements-webcam.txt`. Webcam is 320x180.
- Frontend visueel controleren kan met headless Edge:
  `C:\Program Files (x86)\Microsoft\EdgeCore\<versie>\msedge.exe --headless=new --use-angle=swiftshader --enable-unsafe-swiftshader --window-size=1400,800 --timeout=8000 --screenshot=<pad> http://127.0.0.1:<poort>/`
  (`--timeout` nodig: de MJPEG-stream laadt nooit "klaar"; per run een eigen
  `--user-data-dir`; CSS-transities lopen headless niet altijd af).

## Geleerde lessen (niet opnieuw uitzoeken)

- `channels_last` maakt EfficientNet op deze GPU 9x trager; `cudnn.benchmark` levert
  niets op. Meer dan ~4 DataLoader-workers helpt op Windows niet.
- DataLoader-workers op Windows krijgen dezelfde albumentations-seed; `worker_init` in
  `train_classifier.py` lost dat op. Niet weghalen.
- ultralytics zet een relatief `project`-pad onder `runs/<taak>/`; daarom absolute paden.
  Het downloadt bij elke training `weights/yolo26n.pt` in de werkmap (staat in .gitignore).
- `best.pt` van een onderbroken YOLO-run bevat de optimizer (27 MB); installeren via
  `train_detector --installeer` (strip, 5.6 MB).
- Detector: hoekpunten in beeldvolgorde (0 = linksboven in beeld). Detectordata is
  standaard alleen rechtop (±30°); `--alle-richtingen` voor zijwaarts/omgekeerd. De
  detector vindt omgekeerde kaarten ook zonder die data; de classifier niet.
- Classifier-confidence ~70-90% is normaal (label smoothing 0.1).
- `opencv-python` en `opencv-python-headless` staan allebei geïnstalleerd (albumentations
  vs ultralytics); dat werkt omdat de versies gelijk zijn.

## Open punten

- Omgekeerde kaarten in de classifier (156 klassen of 180°-augmentatie); de backend stuurt
  nu altijd `"reversed": false`.
- Echte webcam-uitsneden verzamelen in `data/test_echt/<kaart>/` als echte testset.
- Op de webcam-pc (CPU) draait de classifier op elk frame; als dat te traag is, om het
  andere frame classificeren.
