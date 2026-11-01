"""Visão de tela: captura contínua e leve (só guarda o último quadro) + detecção de mudança.
O modelo NÃO recebe vídeo: recebe o quadro mais recente quando você pergunta, ou quando a tela muda
(modo automático). É assim que se faz 'tempo real' com modelos locais sem travar o PC."""
import base64
import io
import threading
import time

import mss
from PIL import Image


def _grab(monitor):
    factory = getattr(mss, "MSS", None) or mss.mss  # nome mudou nas versões novas
    with factory() as sct:
        mons = sct.monitors
        shot = sct.grab(mons[monitor if 0 < monitor < len(mons) else 1])
    return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


def _signature(img):
    return img.convert("L").resize((24, 24)).tobytes()


def _diff(a, b):
    if a is None or b is None:
        return 999.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


class ScreenStream:
    def __init__(self, cfg, on_change=None):
        self.c = cfg["vision"]
        self.on_change = on_change          # chamado (em outra thread) quando a tela muda e auto_watch está ligado
        self.auto_watch = bool(self.c["auto_watch"])
        self._frame = None
        self._sig = None
        self._last_sent_sig = None
        self._last_comment = 0.0
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        period = 1.0 / max(0.2, float(self.c["fps"]))
        while not self._stop.is_set():
            try:
                img = _grab(self.c["monitor"])
                img.thumbnail((self.c["max_side"],) * 2)
                self._frame, self._sig = img, _signature(img)
                self._maybe_comment()
            except Exception:  # noqa: BLE001  (monitor desligado, sessão bloqueada...)
                time.sleep(2)
            time.sleep(period)

    def _maybe_comment(self):
        if not (self.auto_watch and self.on_change):
            return
        if time.time() - self._last_comment < self.c["watch_interval"]:
            return
        if _diff(self._sig, self._last_sent_sig) < self.c["change_threshold"]:
            return
        self._last_comment, self._last_sent_sig = time.time(), self._sig
        self.on_change()

    def snapshot_b64(self):
        """Último quadro em JPEG/base64 (ou None se ainda não houver)."""
        if self._frame is None:
            return None
        buf = io.BytesIO()
        self._frame.save(buf, "JPEG", quality=self.c["jpeg_quality"])
        self._last_sent_sig = self._sig
        return base64.b64encode(buf.getvalue()).decode()
