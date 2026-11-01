"""Voz (texto -> fala) 100% local + soundboard, numa fila única para manter a ordem das falas e dos sons."""
import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def _wav_to_array(path):
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
        return data, w.getframerate()


class Voice:
    def __init__(self, cfg, on_state=None, on_error=None):
        self.c = cfg["voice"]
        self.enabled = bool(self.c["enabled"])
        self.on_state = on_state or (lambda speaking: None)
        self.on_error = on_error or (lambda msg: None)
        self.q = queue.Queue()
        self.speaking = False
        self._cancel = threading.Event()
        self._engine = None
        self._warned = False
        threading.Thread(target=self._worker, daemon=True).start()

    # ----- API pública
    def say(self, text):
        if self.enabled and text.strip():
            self.q.put(("say", text.strip()))

    def play(self, path):
        self.q.put(("play", str(path)))

    def stop(self):
        self._cancel.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        sd.stop()

    # ----- motores de síntese: cada um devolve (amostras float32, taxa)
    def _load_engine(self):
        eng = self.c["engine"]
        if eng == "piper":
            model = Path(self.c["piper_model"]).expanduser()
            if not model.exists():
                raise FileNotFoundError(f"modelo Piper não encontrado: {model}")
            return lambda t: self._piper(t, model)
        if eng == "kokoro":
            from kokoro_onnx import Kokoro
            k = Kokoro(str(Path(self.c["kokoro_model"]).expanduser()), str(Path(self.c["kokoro_voices"]).expanduser()))
            return lambda t: k.create(t, voice=self.c["kokoro_voice"], speed=self.c["speed"], lang="pt-br")
        if eng == "xtts":
            from TTS.api import TTS  # licença do XTTS: uso não comercial
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            ref = str(Path(self.c["xtts_ref"]).expanduser())
            def run(t):
                out = tts.tts(text=t, speaker_wav=ref, language="pt")
                return np.asarray(out, dtype=np.float32), 24000
            return run
        raise ValueError(f"engine de voz desconhecido: {eng}")

    def _piper(self, text, model):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.wav"
            cmd = [sys.executable, "-m", "piper", "--model", str(model), "--output_file", str(out)]
            subprocess.run(cmd, input=" ".join(text.split()).encode("utf-8"), capture_output=True, check=True, timeout=60)
            return _wav_to_array(out)

    # ----- reprodução
    def _play_array(self, data, rate):
        data = np.asarray(data, dtype=np.float32) * float(self.c["volume"])
        sd.play(data, rate)
        end = time.time() + len(data) / rate + 0.3
        while time.time() < end and sd.get_stream().active and not self._cancel.is_set():
            time.sleep(0.03)
        sd.stop()

    def _worker(self):
        while True:
            kind, payload = self.q.get()
            self._cancel.clear()
            try:
                if kind == "say":
                    if self._engine is None:
                        self._engine = self._load_engine()
                    self._set(True)
                    data, rate = self._engine(payload)
                    if not self._cancel.is_set():
                        self._play_array(data, rate)
                else:
                    data, rate = sf.read(payload, dtype="float32", always_2d=False)
                    if getattr(data, "ndim", 1) > 1:
                        data = data.mean(axis=1)
                    self._play_array(data, rate)
            except Exception as exc:  # noqa: BLE001
                if not self._warned:
                    self._warned = True
                    self.on_error(f"voz falhou ({exc}). Confira o motor em config.yaml > voice")
            finally:
                if self.q.empty():
                    self._set(False)

    def _set(self, value):
        if self.speaking != value:
            self.speaking = value
            self.on_state(value)
