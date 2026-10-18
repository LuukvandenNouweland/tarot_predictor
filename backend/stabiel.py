"""Flikkervrij aankondigen: een kaart pas melden als hij een paar frames stabiel is."""


class Stabilisator:
    """Houdt bij welke kaart 'aangekondigd' is.

    Per frame krijgt hij een waarneming: een kaartnaam, of None (geen kaart of
    te onzeker). Een nieuwe kaart wordt pas aangekondigd na `stabiel_n` gelijke
    waarnemingen op rij; "geen kaart" pas na `weg_n` keer None op rij.
    """

    def __init__(self, stabiel_n: int = 5, weg_n: int = 5):
        self.stabiel_n = stabiel_n
        self.weg_n = weg_n
        self.aangekondigd: str | None = None
        self._kandidaat: str | None = None
        self._teller = 0

    def update(self, waarneming: str | None) -> bool:
        """Verwerk één frame; geeft True als de aangekondigde kaart verandert."""
        if waarneming == self._kandidaat:
            self._teller += 1
        else:
            self._kandidaat, self._teller = waarneming, 1

        nodig = self.weg_n if waarneming is None else self.stabiel_n
        if self._teller >= nodig and waarneming != self.aangekondigd:
            self.aangekondigd = waarneming
            return True
        return False
