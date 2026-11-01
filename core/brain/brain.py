"""Cérebro do Bututu: decide o que lembrar, como se sentir, quando agir e como aprender com você.
O modelo de linguagem (Ollama) só gera a fala; todo o resto passa por aqui:

  sentidos (texto, tela, voz) -> CÉREBRO -> corpo (fala, sprite, sons, ações)
     antes da resposta : recupera memórias, lê o humor e a situação (hora, disco...) e monta o prompt
     depois da resposta: atualiza humor e histórico, guarda o exemplo para treino
     em segundo plano  : extrai memórias, resume o passado, puxa assunto sozinho (se você deixar)
"""
import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..persona import build_system_prompt, parse_reply
from . import awareness
from .memory import Memory, fold, make_embedder
from .mood import Mood

SCREEN_WORDS = re.compile(r"\b(olha|olhe|v[eê]|veja|repara)\b.{0,25}\b(tela|monitor)\b|\b(na|da) minha tela\b|"
                          r"o que (est[aá]|tem) (na|aparecendo)", re.I)
EMO_EVENT = {"happy": "feliz", "angry": "raiva", "sad": "triste", "scared": "medo"}
KINDS = ("perfil", "preferencia", "episodio")

EXTRACT_SYSTEM = """Você é um extrator de memórias. Leia a conversa entre um usuário e seu assistente e liste SOMENTE o que vale lembrar em conversas futuras: fatos duráveis sobre o usuário (nome, gostos, projetos, rotina), preferências sobre como o assistente deve agir, ou acontecimentos marcantes.
Regras:
- Uma linha por item, no formato: tipo|importância|texto
- tipo = perfil, preferencia ou episodio. importância = número de 1 a 5.
- Escreva o texto em terceira pessoa ("O usuário ..."), curto e específico.
- NUNCA inclua senhas, chaves, números de documentos ou cartões, endereços exatos nem dados de outras pessoas.
- Não invente nada que não esteja na conversa. Piadas e brincadeiras do assistente não são fatos.
- Se não houver nada que valha lembrar, responda apenas: NADA"""

SUMMARY_SYSTEM = """Você mantém o resumo da relação entre um usuário e seu assistente. Atualize o resumo anterior com as novas conversas, em no máximo 5 frases curtas, em terceira pessoa, guardando só o que ajuda em conversas futuras (assuntos recorrentes, combinados, andamento de projetos). Não invente nada. Responda só com o resumo atualizado."""


@dataclass
class Turn:
    system: str
    history: list
    user_text: str          # o que vai ao modelo (pode incluir a descrição da tela)
    raw_user: str           # o que o usuário realmente escreveu
    mode: str = "chat"      # chat | auto (observação da tela) | spont (puxou assunto)
    with_screen: bool = False
    extra: dict = field(default_factory=dict)


def parse_memories(raw):
    """Lê linhas 'tipo|importância|texto' (formato simples, que modelos pequenos erram menos que JSON)."""
    out = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.strip().lstrip("-*• ").split("|", 2)]
        if len(parts) != 3:
            continue
        kind, imp, text = fold(parts[0]), parts[1], parts[2]
        if kind not in KINDS or not 8 <= len(text) <= 240:
            continue
        m = re.search(r"\d", imp)
        out.append((kind, max(1, min(5, int(m.group()))) if m else 3, text))
    return out[:6]


class Brain:
    def __init__(self, cfg, llm, sounds_fn=lambda: {}, apps_fn=lambda: [], data_dir=None):
        self.cfg, self.llm, self.sounds_fn, self.apps_fn = cfg, llm, sounds_fn, apps_fn
        self.bc = cfg["brain"]
        self.data = Path(data_dir or Path(__file__).resolve().parents[2] / "data")
        self.data.mkdir(parents=True, exist_ok=True)
        self.home = Path(cfg["home"])
        self.home.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        emb = make_embedder(cfg["llm"]["base_url"], self.bc["embed_model"], self.bc["embedder"])
        self.mem = Memory(self.home / "brain.db", emb, self.bc["max_memories"])
        self.mood = Mood.from_dict(self._load_json("mood", {}))
        self.history = self._load_json("history", [])
        self.summary = self.mem.get_state("summary", "")
        self.last_chat = float(self.mem.get_state("last_chat", "0") or 0)
        self.mood.last = time.time()          # abrir o programa conta como interação
        self.pending = int(self.mem.get_state("pending", "0") or 0)  # trocas ainda não viradas memória (sobrevive ao reinício)
        self.last_turn = None                 # (usuário, resposta) para /bom e /corrigir
        self.spontaneous = bool(self.bc["spontaneous"])
        self._spont_at = time.time()
        self._consolidating, self._retry_at, self.last_error = False, 0.0, ""

    # ---------- persistência
    def _load_json(self, key, default):
        try:
            return json.loads(self.mem.get_state(key, "") or "null") or default
        except ValueError:
            return default

    def _save_state(self):
        self.mem.set_state("history", json.dumps(self.history, ensure_ascii=False))
        self.mem.set_state("mood", json.dumps(self.mood.to_dict()))
        self.mem.set_state("summary", self.summary)
        self.mem.set_state("last_chat", str(self.last_chat))
        self.mem.set_state("pending", str(self.pending))

    def _append(self, name, row):
        with open(self.data / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self):
        with self.lock:
            self._save_state()

    # ---------- planejamento
    def wants_screen(self, text):
        return bool(SCREEN_WORDS.search(text))

    # ---------- antes da resposta
    def build_turn(self, text, with_screen=False, mode="chat", screen_desc=None):
        now = time.time()
        with self.lock:
            if mode == "chat":
                self.mood.read_text(text, now)
            mood, summary = self.mood.describe(now), self.summary
            hist = list(self.history[-self.bc["working_messages"]:])
        while hist and hist[0]["role"] != "user":
            hist = hist[1:]
        mems = self.mem.recall(text if mode == "chat" else (screen_desc or ""), self.bc["memory_top_k"])
        parts = [build_system_prompt(self.cfg, list(self.sounds_fn()), list(self.apps_fn())),
                 f"Estado interno agora (deixe influenciar levemente o tom, sem anunciar): {mood}.",
                 "Situação agora: " + awareness.snapshot(self.home, self.last_chat, now)]
        if summary:
            parts.append("Resumo do que já aconteceu entre vocês:\n" + summary)
        if mems:
            parts.append("Coisas que você sabe sobre o usuário (use só se for relevante, sem forçar nem recitar):\n"
                         + "\n".join("- " + m["text"] for m in mems))
        user_text = f"[o que aparece na tela agora: {screen_desc}]\n{text}" if screen_desc else text
        return Turn("\n\n".join(parts), hist, user_text, text, mode, with_screen, {"memories": mems})

    # ---------- depois da resposta
    def after_reply(self, turn, reply):
        now = time.time()
        _, emo, _, _ = parse_reply(reply)
        with self.lock:
            if emo:
                self.mood.event(EMO_EVENT[emo], now)  # o que ele sente na fala continua um pouco nas próximas
            if turn.mode == "chat":
                marker = ("(captura de tela) " if turn.with_screen else "") + turn.raw_user
                self.last_chat, self.last_turn = now, (turn.raw_user, reply)
                self.pending += 1
            elif turn.mode == "auto":
                marker = "(observação automática da tela)"
            else:
                marker, self._spont_at = "(você puxou assunto sozinho)", now
            self.history += [{"role": "user", "content": marker}, {"role": "assistant", "content": reply}]
            self.history = self.history[-max(40, self.bc["working_messages"] * 3):]
            self._save_state()
        if turn.mode == "chat" and self.cfg["training"]["log_chats"]:
            self._append("chatlog.jsonl", {"messages": [{"role": "user", "content": turn.raw_user},
                                                        {"role": "assistant", "content": reply}]})

    # ---------- segundo plano
    def tick(self, busy=False):
        """Chamado a cada poucos segundos. Dispara a consolidação e, se ligado, devolve um assunto espontâneo."""
        now = time.time()
        with self.lock:
            self.mood.tick(now)
            due = (self.pending >= self.bc["consolidate_every"] and now - self.last_chat > 20
                   and not self._consolidating and now >= self._retry_at)
        if due:
            threading.Thread(target=self.consolidate, daemon=True).start()
        if busy or not self.spontaneous:
            return None
        return self._maybe_spontaneous(now)

    def _maybe_spontaneous(self, now):
        gap = self.bc["spontaneous_minutes"] * 60
        h, (q0, q1) = datetime.fromtimestamp(now).hour, self.bc["quiet_hours"]
        quiet = (q0 <= h < q1) if q0 <= q1 else (h >= q0 or h < q1)
        with self.lock:
            if (quiet or now - self._spont_at < gap or now - self.last_chat < gap / 2
                    or self.mood.boredom(now) < 0.5):
                return None
            self._spont_at = now
        topics = ["Puxe assunto sozinho com UMA frase curta e boba, no seu estilo.",
                  "Faça uma pergunta curiosa e inesperada ao usuário (UMA frase).",
                  "Solte uma referência rápida a FNAF ou DOOM, em UMA frase."]
        info = awareness.disk(self.home)
        if info and info[0] >= 80:
            topics.append("Reclame dramaticamente da falta de espaço em disco, em UMA frase.")
        return "(Você está entediado. " + random.choice(topics) + ")"

    def _transcript(self, msgs):
        name, lines, pending_user = self.cfg["name"], [], None
        for m in msgs:
            if m["role"] == "user":
                pending_user = None if m["content"].startswith(("(observação", "(você puxou")) else \
                    m["content"].replace("(captura de tela) ", "")
            elif pending_user:
                lines += [f"Usuário: {pending_user}", f"{name}: {parse_reply(m['content'])[0]}"]
                pending_user = None
        return "\n".join(lines)

    def consolidate(self):
        """Extrai memórias das últimas trocas e, se o histórico cresceu, resume o passado (tudo local)."""
        with self.lock:
            if self._consolidating or self.pending == 0:
                return
            self._consolidating = True
            recent = list(self.history[-self.pending * 2:])
        helper = self.bc.get("helper_model") or None
        try:
            convo = self._transcript(recent)
            if convo:
                raw = self.llm.chat(EXTRACT_SYSTEM, [], convo, temperature=0.2, max_tokens=250, model=helper)
                for kind, imp, text in parse_memories(raw):
                    self.mem.add(kind, text, imp)
            self._roll_summary(helper)
            with self.lock:
                self.pending = 0
                self._save_state()
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            self._retry_at = time.time() + 120  # não insiste a cada tick se o modelo estiver fora do ar
        finally:
            with self.lock:
                self._consolidating = False

    def _roll_summary(self, helper):
        keep = self.bc["working_messages"]
        with self.lock:
            if len(self.history) <= keep + 8:
                return
            old, prev = list(self.history[:-keep]), self.summary
        new = self.llm.chat(SUMMARY_SYSTEM, [],
                            f"Resumo anterior:\n{prev or '(nenhum)'}\n\nNovas conversas:\n{self._transcript(old)}",
                            temperature=0.2, max_tokens=220, model=helper).strip()
        if new:
            with self.lock:
                self.summary = new[:900]
                self.history = self.history[len(old):]
                self._save_state()

    # ---------- aprendizado com o usuário
    def feedback(self, kind, correction=""):
        """/bom guarda a resposta como exemplo; /corrigir guarda a resposta certa (e a errada, para comparação)."""
        with self.lock:
            last = self.last_turn
        if not last:
            return "Ainda não tenho nenhuma resposta para avaliar."
        user, reply = last
        msg = lambda a: {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": a}]}  # noqa: E731
        if kind == "bom":
            self._append("exemplos.jsonl", msg(reply))
            with self.lock:
                self.mood.event("bom")
            return "Salvei como bom exemplo para o treino."
        correction = correction.strip()
        if not correction:
            return "Use: /corrigir <como a resposta deveria ter sido> (pode começar com [emoção:feliz], por exemplo)"
        self._append("exemplos.jsonl", msg(correction))
        self._append("preferencias.jsonl", {"prompt": user, "chosen": correction, "rejected": reply})
        with self.lock:
            self.mood.event("ruim")
            for m in reversed(self.history):
                if m["role"] == "assistant":
                    m["content"] = correction  # a conversa segue como se tivesse sido assim
                    break
            self.last_turn = (user, correction)
            self._save_state()
        return "Anotado! Vou usar a correção no treino."

    # ---------- comandos de inspeção
    def memories_text(self, limit=12):
        rows = self.mem.recent(limit)
        head = f"{self.mem.count()} memória(s), busca: {self.mem.embedder.name}"
        return head + ("\n" + "\n".join(f"- [{k} {i}] {t}" for k, i, t in rows) if rows else "")

    def mood_text(self):
        with self.lock:
            d = self.mood.describe()
            return f"Humor agora: {d} (humor {self.mood.valence:+.2f}, energia {self.mood.energy:.2f}, " \
                   f"tédio {self.mood.boredom():.2f})"

    def clear_recent(self):
        """Apaga só a conversa recente (como o /limpar do original). Memórias e resumo ficam."""
        with self.lock:
            self.history, self.pending, self.last_turn = [], 0, None
            self._save_state()
        return "(conversa recente limpa; memórias e resumo continuam)"

    def forget(self, what):
        what = what.strip()
        if what.lower() in ("tudo", "todas"):
            self.mem.forget(None)
            with self.lock:
                self.summary, self.history, self.pending, self.last_turn = "", [], 0, None
                self._save_state()
            return "Esqueci tudo: memórias, resumo e conversa recente."
        return f"Esqueci {self.mem.forget(what)} memória(s) com '{what}'."
