"""Memória de longo prazo do cérebro: SQLite + busca por significado.
Embeddings: Ollama (/api/embed, ex.: nomic-embed-text) ou, sem ele, um fallback offline por hashing de n-gramas."""
import json
import math
import re
import sqlite3
import threading
import time
import unicodedata
import zlib

import numpy as np
import requests

# nunca vira memória: senhas, chaves, documentos, cartões
SECRET = re.compile(
    r"(?i)\b(senha|password|token|api[ _-]?key|chave de api|cpf|cnpj|cart[aã]o)\b|\b(?:\d[ -]?){13,19}\b"
    r"|\d{3}\.?\d{3}\.?\d{3}-?\d{2}|[A-Za-z0-9_\-]{32,}")
STOP = set("de a o que e do da em um uma para com nao os as se na no por mais mas foi ele ela eu voce meu minha "
           "seu sua ao dos das como ja tem ser esta isso esse essa usuario".split())
SCHEMA = """
CREATE TABLE IF NOT EXISTS memories(
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL UNIQUE, importance INTEGER NOT NULL,
  created REAL NOT NULL, last_used REAL NOT NULL, uses INTEGER NOT NULL DEFAULT 0, emb_name TEXT, emb TEXT);
CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);
"""


def fold(text):
    t = unicodedata.normalize("NFD", str(text).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _dump(vec):
    return json.dumps([round(float(x), 5) for x in vec])


class HashEmbedder:
    """Fallback offline: n-gramas de caracteres por palavra, em vetor de tamanho fixo. Pega só semelhança de palavras."""
    name, dim, min_sim, dup_sim = "hash512", 512, 0.2, 0.75

    def embed(self, texts):
        out = []
        for text in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in fold(text).split():
                if len(tok) < 3 or tok in STOP:
                    continue
                s = f"#{tok}#"
                for i in range(len(s) - 2):
                    v[zlib.crc32(s[i:i + 3].encode()) % self.dim] += 1.0  # crc32: estável entre execuções
            n = float(np.linalg.norm(v))
            out.append(v / n if n else v)
        return out


class OllamaEmbedder:
    min_sim, dup_sim = 0.45, 0.93

    def __init__(self, url, model):
        self.url, self.model = url.rstrip("/"), model
        self.name = f"ollama:{model}"

    def embed(self, texts):
        r = requests.post(self.url + "/api/embed", json={"model": self.model, "input": list(texts)}, timeout=30)
        r.raise_for_status()
        out = []
        for e in r.json()["embeddings"]:
            v = np.asarray(e, dtype=np.float32)
            n = float(np.linalg.norm(v))
            out.append(v / n if n else v)
        return out


def make_embedder(url, model, mode="auto"):
    """auto/ollama: tenta o modelo de embeddings do Ollama e, se não der, cai no fallback offline. hash: força o fallback."""
    if mode != "hash":
        try:
            emb = OllamaEmbedder(url, model)
            emb.embed(["teste"])
            return emb
        except Exception:  # noqa: BLE001
            pass
    return HashEmbedder()


class Memory:
    def __init__(self, path, embedder, max_items=2000):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.RLock()
        self.embedder, self.fallback, self.max_items = embedder, HashEmbedder(), max_items
        with self.lock:
            self.db.executescript(SCHEMA)
        self._items = {}
        for r in self.db.execute("SELECT id,kind,text,importance,created,last_used,uses,emb_name,emb FROM memories"):
            vec = np.asarray(json.loads(r[8]), dtype=np.float32) if r[8] else None
            self._items[r[0]] = dict(id=r[0], kind=r[1], text=r[2], imp=r[3], created=r[4], used=r[5], uses=r[6],
                                     emb_name=r[7], vec=vec)
        if any(m["emb_name"] != embedder.name or m["vec"] is None for m in self._items.values()):
            threading.Thread(target=self._reindex, daemon=True).start()  # trocou de motor de embeddings

    def _embed(self, texts):
        try:
            return self.embedder.embed(texts), self.embedder.name
        except Exception:  # noqa: BLE001  (Ollama ocupado/desligado)
            return self.fallback.embed(texts), self.fallback.name

    def _reindex(self):
        stale = [m for m in list(self._items.values()) if m["emb_name"] != self.embedder.name or m["vec"] is None]
        for i in range(0, len(stale), 32):
            batch = stale[i:i + 32]
            try:
                vecs = self.embedder.embed([m["text"] for m in batch])
            except Exception:  # noqa: BLE001
                return
            with self.lock:
                for m, v in zip(batch, vecs):
                    m["vec"], m["emb_name"] = v, self.embedder.name
                    self.db.execute("UPDATE memories SET emb_name=?, emb=? WHERE id=?",
                                    (m["emb_name"], _dump(v), m["id"]))
                self.db.commit()

    def _limits(self, name):
        e = self.embedder if name == self.embedder.name else self.fallback
        return e.min_sim, e.dup_sim

    # ----- escrita
    def add(self, kind, text, importance=3):
        """Guarda uma memória. Devolve True se for nova (duplicadas só reforçam a existente)."""
        text = " ".join(str(text).split())[:240]
        if len(text) < 8 or SECRET.search(text):
            return False
        (vec,), name = self._embed([text])
        _, dup_sim = self._limits(name)
        with self.lock:
            same = [m for m in self._items.values() if m["emb_name"] == name and m["vec"] is not None]
            if same:
                sims = np.stack([m["vec"] for m in same]) @ vec
                i = int(np.argmax(sims))
                if float(sims[i]) >= dup_sim:
                    m = same[i]
                    m["imp"], m["used"] = max(m["imp"], importance), time.time()
                    self.db.execute("UPDATE memories SET importance=?, last_used=? WHERE id=?", (m["imp"], m["used"], m["id"]))
                    self.db.commit()
                    return False
            now = time.time()
            cur = self.db.execute(
                "INSERT OR IGNORE INTO memories(kind,text,importance,created,last_used,uses,emb_name,emb) VALUES(?,?,?,?,?,0,?,?)",
                (kind, text, int(importance), now, now, name, _dump(vec)))
            self.db.commit()
            if cur.rowcount == 0:
                return False
            self._items[cur.lastrowid] = dict(id=cur.lastrowid, kind=kind, text=text, imp=int(importance), created=now,
                                              used=now, uses=0, emb_name=name, vec=vec)
            self._prune()
            return True

    def _prune(self):
        extra = len(self._items) - self.max_items
        if extra <= 0:
            return
        now = time.time()
        def keep(m):
            return m["imp"] / 5 + 0.5 * math.exp(-(now - m["created"]) / 86400 / 90) + min(m["uses"], 10) / 20
        for m in sorted(self._items.values(), key=keep)[:extra]:
            self.db.execute("DELETE FROM memories WHERE id=?", (m["id"],))
            self._items.pop(m["id"], None)
        self.db.commit()

    def forget(self, needle=None):
        """needle=None apaga tudo; senão apaga as memórias que contêm a palavra (sem acento/maiúscula)."""
        with self.lock:
            if needle is None:
                ids = list(self._items)
            else:
                n = fold(needle)
                ids = [i for i, m in self._items.items() if n and n in fold(m["text"])]
            self.db.executemany("DELETE FROM memories WHERE id=?", [(i,) for i in ids])
            self.db.commit()
            for i in ids:
                self._items.pop(i, None)
        return len(ids)

    # ----- leitura
    def recall(self, query, k=5, pinned=3):
        """Memórias mais úteis para a pergunta: semelhança + importância + recência, mais os fatos de perfil fixados."""
        with self.lock:
            items = list(self._items.values())
        if not items:
            return []
        pins = sorted((m for m in items if m["kind"] == "perfil" and m["imp"] >= 4),
                      key=lambda m: (-m["imp"], -m["uses"]))[:pinned]
        chosen = []
        if query and query.strip():
            (qv,), name = self._embed([query])
            cand = [m for m in items if m["emb_name"] == name and m["vec"] is not None]
            if cand:
                sims = np.stack([m["vec"] for m in cand]) @ qv
                min_sim, _ = self._limits(name)
                floor = max(min_sim, float(sims.max()) - 0.2)
                now, scored = time.time(), []
                for m, s in zip(cand, sims):
                    if float(s) < floor:
                        continue
                    rec = math.exp(-(now - m["created"]) / 86400 / 45)
                    scored.append((0.65 * float(s) + 0.15 * rec + 0.20 * m["imp"] / 5, m))
                chosen = [m for _, m in sorted(scored, key=lambda x: -x[0])[:k]]
        out, seen = [], set()
        for m in chosen + pins:
            if m["id"] not in seen:
                seen.add(m["id"])
                out.append(m)
        self._touch(out)
        return [{"kind": m["kind"], "text": m["text"]} for m in out]

    def _touch(self, items):
        if not items:
            return
        now = time.time()
        with self.lock:
            for m in items:
                m["used"], m["uses"] = now, m["uses"] + 1
            self.db.executemany("UPDATE memories SET last_used=?, uses=uses+1 WHERE id=?", [(now, m["id"]) for m in items])
            self.db.commit()

    def recent(self, limit=15):
        with self.lock:
            items = sorted(self._items.values(), key=lambda m: -m["created"])[:limit]
        return [(m["kind"], m["imp"], m["text"]) for m in items]

    def count(self):
        return len(self._items)

    # ----- estado simples (resumo, histórico, humor...)
    def get_state(self, key, default=""):
        with self.lock:
            r = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    def set_state(self, key, value):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO state(key,value) VALUES(?,?)", (key, value))
            self.db.commit()
