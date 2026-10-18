/** Bericht van de backend via WS /ws (zie backend/main.py). */
export interface Status {
  type: "change" | "update";
  card: string | null; // bv. "the_tower"
  label: string | null; // bv. "The Tower"
  reversed: boolean;
  confidence: number;
  box: [number, number, number, number] | null; // x1, y1, x2, y2 in framepixels
  corners: [number, number][] | null; // 4 hoekpunten: LB, RB, RO, LO
  frame_size: [number, number]; // breedte, hoogte
}
