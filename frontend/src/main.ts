import "./style.css";

import { ANIMATIES, reactieOpKaart } from "./animaties";
import { Overlay } from "./overlay";
import { Personage } from "./personage";
import type { Status } from "./types";
import { verbind } from "./verbinding";

const video = document.querySelector<HTMLImageElement>("#video")!;
const verbindingTekst = document.querySelector<HTMLElement>("#verbinding")!;
const kaartnaam = document.querySelector<HTMLElement>("#kaartnaam")!;
const kaartdetail = document.querySelector<HTMLElement>("#kaartdetail")!;

const overlay = new Overlay(document.querySelector<HTMLCanvasElement>("#overlay")!);
const personage = new Personage(document.querySelector<HTMLElement>("#scene")!);
personage.laad("/models/Skull.gltf").catch((fout) => console.error("Model laden mislukt:", fout));

let vorigeKaart: string | null = null;

function toonKaart(s: Status): void {
  if (s.card) {
    kaartnaam.textContent = s.label;
    kaartdetail.textContent = s.reversed ? "omgekeerd" : `${Math.round(s.confidence * 100)}% zeker`;
    // Fade-in opnieuw starten: klasse weg, reflow forceren, klasse terug.
    kaartnaam.classList.remove("zichtbaar");
    void kaartnaam.offsetWidth;
    kaartnaam.classList.add("zichtbaar");
  } else {
    kaartnaam.classList.remove("zichtbaar");
    kaartdetail.textContent = "Houd een kaart voor de camera";
  }
}

function opBericht(s: Status): void {
  overlay.teken(s);
  if (s.type !== "change") return;

  toonKaart(s);
  if (s.card && s.card !== vorigeKaart) {
    const reactie = reactieOpKaart(s.card);
    if (reactie) personage.speelEenmalig(reactie);
  } else if (!s.card && vorigeKaart && ANIMATIES.kaartWeg) {
    personage.speelEenmalig(ANIMATIES.kaartWeg);
  }
  vorigeKaart = s.card;
}

let eerderVerbonden = false;
verbind(opBericht, (verbonden) => {
  verbindingTekst.textContent = verbonden ? "" : "geen verbinding met de backend, opnieuw proberen…";
  // Na een herstart van de backend is de MJPEG-stream dood: opnieuw openen.
  if (verbonden && eerderVerbonden) video.src = `/video?t=${Date.now()}`;
  eerderVerbonden ||= verbonden;
});
