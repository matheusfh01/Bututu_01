"""Audição local: microfone -> texto com faster-whisper. Detecta o fim da frase por silêncio (VAD simples)."""
import queue
import threading
import time

import numpy as np
import sounddevice as sd

RATE = 16000
BLOCK = 480  # 30 ms


class Ears:
    def __init__(self, cfg, on_text, is_muted=lambda: False):
        from faster_whisper import WhisperModel
        c = cfg["stt"]
        self.c = c
        self.on_text = on_text
        self.is_muted = is_muted  # ignora o microfone enquanto a IA fala (evita ouvir a si mesma)
        self.model = WhisperModel(c["model"], device=c["device"], compute_type="default")
        self.enabled = True
        self._q = queue.Queue()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        silence_blocks = int(self.c["silence_seconds"] * RATE / BLOCK)
        thr = float(self.c["energy_threshold"])
        buf, quiet, speaking = [], 0, False
        with sd.InputStream(samplerate=RATE, channels=1, blocksize=BLOCK, dtype="float32",
                            callback=lambda d, f, t, s: self._q.put(d[:, 0].copy())):
            while True:
                block = self._q.get()
                if not self.enabled or self.is_muted():
                    buf, quiet, speaking = [], 0, False
                    continue
                loud = float(np.sqrt(np.mean(block ** 2))) > thr
                if loud:
                    speaking, quiet = True, 0
                elif speaking:
                    quiet += 1
                if speaking:
                    buf.append(block)
                if speaking and quiet >= silence_blocks:
                    audio = np.concatenate(buf)
                    buf, quiet, speaking = [], 0, False
                    if len(audio) > RATE * 0.5:  # ignora estalos
                        self._transcribe(audio)

    def _transcribe(self, audio):
        try:
            segs, _ = self.model.transcribe(audio, language=self.c["language"], vad_filter=True)
            text = " ".join(s.text.strip() for s in segs).strip()
            if len(text) > 1:
                self.on_text(text)
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
