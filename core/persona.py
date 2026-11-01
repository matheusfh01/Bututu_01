"""Monta o prompt de sistema e interpreta as etiquetas que a IA escreve ([emoção:..], [som:..], [acao:..])."""
import re
import unicodedata
from pathlib import Path

EMOTIONS = {
    "happy": ("feliz", "alegre", "happy", "joy"),
    "angry": ("raiva", "bravo", "irritado", "angry"),
    "sad": ("triste", "chateado", "sad"),
    "scared": ("medo", "assustado", "scared", "fear"),
}
EMOTION_TAG = re.compile(r"\[\s*(?:emo[cç][aã]o|emotion)\s*:\s*([^\]\n]{1,20}?)\s*\]", re.I)
SOUND_TAG = re.compile(r"\[\s*som\s*:\s*([^\]\n]{1,40}?)\s*\]", re.I)
ACTION_TAG = re.compile(r"\[\s*acao\s*:\s*([a-z_]+)\s*(?:\|\s*([^\]\n]{0,120}))?\]", re.I)


def _norm(word):
    w = unicodedata.normalize("NFD", word.strip().lower())
    return "".join(c for c in w if not unicodedata.combining(c))


def build_system_prompt(cfg, sound_names=(), app_names=()):
    path = Path(__file__).resolve().parent.parent / "persona" / "persona.md"
    prompt = path.read_text(encoding="utf-8").replace("{name}", cfg["name"])
    prompt += ("\n\nEmoções: quando a fala tiver emoção clara, comece com UMA etiqueta: [emoção:feliz], "
               "[emoção:raiva], [emoção:triste] ou [emoção:medo]. Ela muda sua aparência e não é lida em voz alta.")
    if sound_names:
        prompt += ("\nSons: escreva [som:nome] no meio da resposta (no máximo 2) só quando combinar. "
                   "Disponíveis: " + ", ".join(sound_names[:40]) + ".")
    if app_names:
        prompt += ("\nAções: para abrir um app, escreva [acao:abrir|nome]. Apps permitidos: "
                   + ", ".join(app_names) + ". Nunca invente nomes. O usuário confirma antes de executar.")
    prompt += ("\nObservação automática da tela: se não houver nada relevante para comentar, "
               "responda exatamente [silencio].")
    return prompt


def parse_reply(text):
    """Retorna (texto_falado, emocao|None, sons[], acoes[(nome, arg)])."""
    emo = None
    for m in EMOTION_TAG.finditer(text):
        w = _norm(m.group(1))
        for key, names in EMOTIONS.items():
            if w in names and emo is None:
                emo = key
    sounds = [m.group(1).strip().lower() for m in SOUND_TAG.finditer(text)][:3]
    actions = [(m.group(1).lower(), (m.group(2) or "").strip()) for m in ACTION_TAG.finditer(text)][:2]
    clean = ACTION_TAG.sub("", SOUND_TAG.sub("", EMOTION_TAG.sub("", text)))
    return re.sub(r"[ \t]{2,}", " ", clean).strip(), emo, sounds, actions
