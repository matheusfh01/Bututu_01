"""Consciência de situação: o que o Bututu 'sente' do mundo real (hora, tempo sem conversar, disco cheio)."""
import shutil
import time
from datetime import datetime

DIAS = ("segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo")


def part_of_day(h):
    return "madrugada" if h < 6 else "manhã" if h < 12 else "tarde" if h < 18 else "noite"


def humanize(sec):
    if sec < 120:
        return "agora há pouco"
    if sec < 3600:
        return f"há {int(sec // 60)} min"
    if sec < 86400:
        return f"há {int(sec // 3600)} h"
    return f"há {int(sec // 86400)} dia(s)"


def disk(home):
    """(% usado, GB livres) do disco onde ficam os dados, ou None."""
    try:
        u = shutil.disk_usage(str(home))
        return u.used / u.total * 100, u.free / 1e9
    except OSError:
        return None


def snapshot(home, last_chat=0.0, now=None):
    now = now or time.time()
    d = datetime.fromtimestamp(now)
    parts = [f"{DIAS[d.weekday()]}, {d:%d/%m/%Y} {d:%H:%M} ({part_of_day(d.hour)})."]
    if last_chat:
        parts.append(f"Última conversa: {humanize(now - last_chat)}.")
    info = disk(home)
    if info and info[0] >= 80:  # só aparece quando é motivo de drama
        parts.append(f"Disco do usuário: {info[0]:.0f}% cheio ({info[1]:.0f} GB livres).")
    return " ".join(parts)
