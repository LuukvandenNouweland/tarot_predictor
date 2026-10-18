/**
 * Welke animatie wanneer speelt. Pas dit object aan; meer hoeft niet.
 *
 * Namen moeten bestaan in het model. Bij het laden worden alle namen in de
 * console gelogd. Skull.gltf heeft: Idle, Yes, No, Jump, Dance, HitRecieve,
 * Death, Bite_Front, Bite_InPlace, Walk.
 */
export interface AnimatieConfig {
  /** Loopt continu als er niets anders speelt. */
  idle: string;
  /** Speelt één keer bij een nieuw herkende kaart; null = niets (blijft idle). */
  nieuweKaart: string | null;
  /** Speelt één keer als de kaart uit beeld gaat; null = niets. */
  kaartWeg: string | null;
  /** Eigen reactie per kaart; gaat vóór nieuweKaart. */
  perKaart: Record<string, string>;
  /** Overgangstijd (crossfade) tussen animaties, in seconden. */
  overgang: number;
}

export const ANIMATIES: AnimatieConfig = {
  idle: "Idle",
  nieuweKaart: "Yes", // bv. "Yes"
  kaartWeg: "No", // bv. "No"
  perKaart: {
    // the_tower: "Death",
    // the_sun: "Dance",
  },
  overgang: 0.3,
};

/** Animatie voor een nieuw herkende kaart, of null. */
export function reactieOpKaart(kaart: string): string | null {
  return ANIMATIES.perKaart[kaart] ?? ANIMATIES.nieuweKaart;
}
