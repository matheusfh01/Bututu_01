"""Humor do Bututu: dois eixos (humor e energia) que reagem ao que acontece e voltam ao normal com o tempo.
O tédio não é guardado: é calculado pelo tempo desde a última interação."""
import re
import time

PRAISE = re.compile(r"\b(obrigad[oa]|valeu|vlw|parab[eé]ns|mandou bem|voc[eê] [eé] (demais|incr[ií]vel|o melhor|foda)|"
                    r"gostei|adorei|amei)\b", re.I)
INSULT = re.compile(r"\b(burro|idiota|in[uú]til|lixo|besta|cala a boca|odeio voc[eê])\b", re.I)
# evento -> (variação do humor, variação da energia)
EVENTS = {"conversa": (0.0, 0.05), "elogio": (0.25, 0.05), "ofensa": (-0.30, 0.10), "bom": (0.20, 0.0),
          "ruim": (-0.15, 0.0), "feliz": (0.15, 0.10), "triste": (-0.20, -0.10), "raiva": (-0.15, 0.15),
          "medo": (-0.05, 0.15)}
RESETS_BOREDOM = ("conversa", "elogio", "ofensa", "bom", "ruim")


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Mood:
    BASE_V, BASE_E, HALF_LIFE = 0.35, 0.70, 720.0  # meia-vida: 12 min

    def __init__(self, valence=BASE_V, energy=BASE_E, t=None, last=None):
        self.valence, self.energy = valence, energy
        self.t = t or time.time()
        self.last = last or self.t  # última interação

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(d.get("valence", cls.BASE_V), d.get("energy", cls.BASE_E), d.get("t"), d.get("last"))

    def to_dict(self):
        return {"valence": round(self.valence, 3), "energy": round(self.energy, 3), "t": self.t, "last": self.last}

    def tick(self, now=None):
        now = now or time.time()
        f = 0.5 ** (max(0.0, now - self.t) / self.HALF_LIFE)
        self.valence = self.BASE_V + (self.valence - self.BASE_V) * f
        self.energy = self.BASE_E + (self.energy - self.BASE_E) * f
        self.t = now

    def boredom(self, now=None):
        return _clamp(((now or time.time()) - self.last) / 1800.0, 0.0, 1.0)  # 30 min sem papo = tédio máximo

    def event(self, name, now=None):
        now = now or time.time()
        self.tick(now)
        dv, de = EVENTS.get(name, (0.0, 0.0))
        self.valence = _clamp(self.valence + dv, -1.0, 1.0)
        self.energy = _clamp(self.energy + de, 0.0, 1.0)
        if name in RESETS_BOREDOM:
            self.last = now

    def read_text(self, text, now=None):
        """Reage ao que o usuário escreveu (elogio, ofensa ou conversa normal)."""
        self.event("ofensa" if INSULT.search(text) else "elogio" if PRAISE.search(text) else "conversa", now)

    def describe(self, now=None):
        self.tick(now)
        v, e, b = self.valence, self.energy, self.boredom(now)
        humor = ("muito animado" if v > .6 else "de bom humor" if v > .2 else "normal" if v > -.2
                 else "meio chateado" if v > -.55 else "bem irritado")
        energia = "elétrico" if e > .85 else "com energia" if e > .5 else "meio sem energia" if e > .3 else "com sono"
        extra = ", entediado (faz tempo que ninguém fala com você)" if b > .6 else ""
        return f"{humor}, {energia}{extra}"

    def emotion(self):
        """Emoção para o sprite quando o humor está num extremo (senão None)."""
        if self.valence > .75:
            return "happy"
        if self.valence < -.55:
            return "angry" if self.energy > .6 else "sad"
        return None
