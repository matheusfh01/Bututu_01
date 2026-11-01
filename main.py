"""Bututu Local: assistente de desktop 100% local (cérebro + LLM + visão + voz + audição).
Uso:  python setup_brain.py   (uma vez)   e depois   python main.py
"""
import re
import threading
from pathlib import Path

import yaml

from core.actions import Actions
from core.brain import Brain
from core.llm import LocalLLM
from core.persona import parse_reply
from core.tts import Voice
from core.ui import Overlay
from core.vision import ScreenStream

ROOT = Path(__file__).resolve().parent
SOUND_EXTS = (".wav", ".mp3", ".ogg", ".flac")
BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n+")
HELP = ("Comandos: /tela [pergunta] /limpar /som nome /voz /calar /emocao nome /ver /sair\n"
        "Cérebro: /memoria /esquecer tudo|palavra /humor /espontaneo /bom /corrigir <resposta certa>")


def load_config():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["home"] = str(Path(cfg["home"]).expanduser())
    return cfg


def split_ready(buf):
    """Separa frases completas do que ainda está sendo escrito (não corta no meio de uma [etiqueta])."""
    if buf.count("[") > buf.count("]"):
        return [], buf
    parts = BOUNDARY.split(buf)
    return [p for p in parts[:-1] if p.strip()], parts[-1]


class Bututu:
    def __init__(self):
        self.cfg = load_config()
        self.home = Path(self.cfg["home"])
        for sub in ("sprites", "sounds", "voices"):
            (self.home / sub).mkdir(parents=True, exist_ok=True)
        self.llm = LocalLLM(self.cfg)
        self.busy = threading.Lock()
        self.ui = Overlay(self.cfg, self.home / "sprites", {
            "submit": self.submit, "look": lambda: self.run("O que você está vendo na minha tela?", True),
            "watch": self.toggle_watch, "voice": self.toggle_voice, "quit": self.quit, "status": self.status})
        self.voice = Voice(self.cfg, on_state=lambda s: self.ui.post(lambda: self._speaking(s)),
                           on_error=lambda m: self.ui.post(lambda: self.ui.say(f"[{m}]")))
        self.screen = ScreenStream(self.cfg, on_change=self.auto_look)
        self.actions = Actions(self.cfg, self.ui.ask)
        self.brain = Brain(self.cfg, self.llm, self.sounds, self.actions.app_names, ROOT / "data")
        self.screen.start()
        if self.cfg["stt"]["enabled"]:
            from core.stt import Ears
            Ears(self.cfg, lambda t: self.ui.post(lambda: self.submit(t, spoken=True)),
                 is_muted=lambda: self.voice.speaking or self.busy.locked()).start()
        self.ui.post(lambda: self.ui.say("Oi! Sou o " + self.cfg["name"] + ". Clique em mim para conversar."))
        self.ui.root.after(5000, self._tick)

    # ----- estado / menu
    def status(self):
        return {"watching": self.screen.auto_watch, "voice": self.voice.enabled}

    def toggle_watch(self):
        self.screen.auto_watch = not self.screen.auto_watch
        self.ui.say("Observação automática " + ("LIGADA (os quadros só são enviados ao modelo local)."
                                                if self.screen.auto_watch else "desligada."))

    def toggle_voice(self):
        self.voice.enabled = not self.voice.enabled
        if not self.voice.enabled:
            self.voice.stop()

    def _speaking(self, speaking):
        if not self.busy.locked():
            self.ui.set_state("speaking" if speaking else "idle")

    def quit(self):
        self.screen.stop()
        self.brain.close()
        self.ui.root.destroy()

    def sounds(self):
        d = self.home / "sounds"
        return {p.stem.lower(): p for p in sorted(d.iterdir()) if p.suffix.lower() in SOUND_EXTS}

    # ----- relógio do cérebro: consolidação de memória, humor e assunto espontâneo
    def _tick(self):
        try:
            prompt = self.brain.tick(busy=self.busy.locked() or self.voice.speaking)
            if prompt and self.llm.alive():
                self.run(prompt, False, mode="spont")
            emo = self.brain.mood.emotion()
            if emo and not self.ui.emotion:
                self.ui.set_emotion(emo, 5000)
        finally:
            self.ui.root.after(5000, self._tick)

    # ----- entrada do usuário
    def submit(self, text, spoken=False):
        text = text.strip()
        if not text:
            return
        if spoken:
            self.ui.say(text, who="user")
        if text.startswith("/"):
            return self.command(text)
        self.run(text, with_screen=self.brain.wants_screen(text))

    def command(self, text):
        cmd, _, arg = text[1:].partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        if cmd == "tela":
            self.run(arg or "O que você está vendo na minha tela?", True)
        elif cmd == "limpar":
            self.ui.say(self.brain.clear_recent())
        elif cmd == "som":
            path = self.sounds().get(arg.lower())
            self.voice.play(path) if path else self.ui.say("Sons: " + (", ".join(self.sounds()) or "nenhum"))
        elif cmd == "voz":
            self.toggle_voice()
        elif cmd == "calar":
            self.voice.stop()
        elif cmd in ("emocao", "emoção"):
            _, emo, _, _ = parse_reply(f"[emoção:{arg}]")
            self.ui.set_emotion(emo or None)
        elif cmd == "ver":
            self.toggle_watch()
        elif cmd == "memoria":
            self.ui.say(self.brain.memories_text())
        elif cmd == "esquecer":
            self.ui.say(self.brain.forget(arg) if arg else "Use: /esquecer tudo  ou  /esquecer <palavra>")
        elif cmd == "humor":
            self.ui.say(self.brain.mood_text())
        elif cmd == "espontaneo":
            self.brain.spontaneous = not self.brain.spontaneous
            self.ui.say("Vou puxar assunto sozinho de vez em quando." if self.brain.spontaneous else "Fico quieto até você chamar.")
        elif cmd == "bom":
            self.ui.say(self.brain.feedback("bom"))
        elif cmd == "corrigir":
            self.ui.say(self.brain.feedback("corrigir", arg))
        elif cmd == "sair":
            self.quit()
        else:
            self.ui.say(HELP)

    def auto_look(self):
        self.run("Observação automática da tela: comente em 1 ou 2 frases só se houver algo relevante.",
                 True, mode="auto")

    # ----- conversa com o modelo
    def run(self, text, with_screen=False, mode="chat"):
        if not self.busy.acquire(blocking=False):
            if mode == "chat":
                self.ui.say("(calma, ainda estou pensando…)")
            return
        self.ui.set_state("thinking")
        threading.Thread(target=self._worker, args=(text, with_screen, mode), daemon=True).start()

    def _worker(self, text, with_screen, mode):
        try:
            if not self.llm.alive():
                raise RuntimeError("Ollama não está rodando. Abra o Ollama e rode: python setup_brain.py")
            image, desc = None, None
            if with_screen:
                image = self.screen.snapshot_b64()
                if image is None:
                    raise RuntimeError("ainda não consegui capturar a tela")
                describer = self.cfg["vision"].get("describer_model")
                if describer:  # modelo de visão separado descreve; o modelo de personalidade responde
                    desc, image = self.llm.describe(image, text, describer), None
            turn = self.brain.build_turn(text, with_screen, mode, desc)   # memórias + humor + situação + histórico
            full, buf, first = "", "", True
            for piece in self.llm.stream(turn.system, turn.history, turn.user_text, image):
                full += piece
                if mode == "auto":
                    continue
                buf += piece
                ready, buf = split_ready(buf)
                for sent in ready:
                    self._emit(sent, append=not first)
                    first = False
            if mode == "auto" and re.match(r"\s*\[\s*sil[eê]ncio", full, re.I):
                return
            if mode == "auto":
                buf = full
            if buf.strip():
                self._emit(buf, append=not first)
            self.brain.after_reply(turn, full.strip())
        except Exception as exc:  # noqa: BLE001
            self.ui.post(lambda: self.ui.say(f"[falha: {exc}]"))
        finally:
            self.busy.release()
            self.ui.post(lambda: self.ui.set_state("speaking" if self.voice.speaking else "idle"))

    def _emit(self, sentence, append):
        clean, emo, sound_names, acts = parse_reply(sentence)
        if emo:
            self.ui.post(lambda: self.ui.set_emotion(emo))
        if clean:
            self.ui.post(lambda: self.ui.say(clean, append=append))
            self.voice.say(clean)
        known = self.sounds()
        for name in sound_names:
            if name in known:
                self.voice.play(known[name])
        for name, arg in acts:
            threading.Thread(target=lambda n=name, a=arg: self.ui.post(
                lambda r=self.actions.run(n, a): self.ui.say(r, append=True)), daemon=True).start()

    def run_forever(self):
        self.ui.run()


if __name__ == "__main__":
    Bututu().run_forever()
