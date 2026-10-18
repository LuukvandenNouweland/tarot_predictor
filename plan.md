# Tarot-kaartherkenning via webcam

## Doel
Een kaart die voor de webcam wordt gehouden automatisch vinden en herkennen
(welke kaart, en of hij rechtop of omgekeerd is).

## Aanpak: twee stappen
1. **Detectie**: waar is de kaart in beeld?
   - Objectdetector met één klasse: "kaart" (YOLO via `ultralytics`)
   - Daarna uitsnijden en rechttrekken met perspectiefcorrectie (OpenCV)
2. **Classificatie**: welke kaart is het?
   - Voorgetraind netwerk fine-tunen (ResNet/EfficientNet via `torchvision` of `timm`)
   - 78 klassen, of 156 inclusief omgekeerde kaarten
   - Niet vanaf nul trainen: voortbouwen op een voorgetraind model

## Stap 1: brondata (klaar)
- 78 scans als .jpg, hernoemd naar `the_fool.jpg`, `three_of_cups.jpg`, enz.
- Volgorde: grote arcana (Rider-Waite: Strength = 8, Justice = 11),
  dan wands, cups, swords, pentacles (ace t/m ten, page, knight, queen, king)

## Stap 2: synthetische dataset maken (volgende)
Per kaart honderden varianten genereren met `albumentations` / OpenCV:
- perspectief- en rotatievervorming
- helderheid, contrast, kleurtemperatuur
- bewegingsonscherpte, ruis, JPEG-compressie
- glansvlekken
- stukken afdekken (vingers)
- kaart op willekeurige achtergronden plakken (ook bruikbaar voor de detector, omdat de positie dan bekend is)

## Stap 3: classifier trainen
- Fine-tunen op de synthetische data
- Testen op **echte** webcamfoto's (30–50 stuks, recht uitgesneden)
- Kijken naar top-1- en top-5-nauwkeurigheid en naar welke kaarten verward worden

## Stap 4: detector toevoegen
- Voortrainen op de synthetische composities
- Aanvullen met 100–200 gelabelde echte webcamfoto's (Roboflow / Label Studio)

## Stap 5: alles koppelen
Webcam → detectie → uitsnijden + rechttrekken → classificatie → resultaat in beeld

## Aandachtspunten
- Echte webcambeelden zijn de enige test die telt
- Kaart groot genoeg in beeld houden; lage resolutie is de grootste vijand
- Perspectiefcorrectie vóór classificatie helpt veel
- Lijkende kaarten (vooral genummerde kaarten) zijn het lastigst

## Stack
- PyTorch + torchvision / timm: classificatie
- ultralytics (YOLO): detectie
- OpenCV: webcam, uitsnijden, perspectiefcorrectie
- albumentations: augmentatie
- PyMuPDF: plaatjes uit pdf halen (indien nodig)