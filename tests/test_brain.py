"""Testes do cérebro sem Ollama: usa um modelo de mentira e o modo de memória offline (hash).
Rode na pasta do projeto:  python -m unittest discover -s tests -v"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.brain import Brain  # noqa: E402
from core.brain.brain import parse_memories  # noqa: E402
from core.brain.mood import Mood  # noqa: E402


class FakeLLM:
    def __init__(self, replies=()):
        self.replies, self.calls = list(replies), []

    def chat(self, system, history, user_text, image_b64=None, **kw):
        self.calls.append((system, user_text, kw))
        return self.replies.pop(0) if self.replies else ""


def make_brain(tmp, llm=None, **brain_over):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["home"] = str(Path(tmp) / "home")
    cfg["brain"].update(embedder="hash", **brain_over)
    return Brain(cfg, llm or FakeLLM(), data_dir=Path(tmp) / "data")


class TestMemory(unittest.TestCase):
    def test_recall_dedup_and_secrets(self):
        with tempfile.TemporaryDirectory() as d:
            m = make_brain(d).mem
            self.assertTrue(m.add("perfil", "O usuário gosta muito de DOOM e FNAF", 3))
            self.assertTrue(m.add("perfil", "O usuário mora em Curitiba", 3))
            self.assertTrue(m.add("episodio", "O usuário tem um gato chamado Miau", 3))
            self.assertTrue(m.add("perfil", "O usuário se chama Ana", 5))  # importância alta = fixada
            self.assertFalse(m.add("perfil", "O usuário gosta muito de DOOM e FNAF", 3))  # duplicada
            self.assertFalse(m.add("perfil", "A senha do wifi do usuário é 12345678", 3))  # segredo
            texts = [r["text"] for r in m.recall("me fala de doom")]
            self.assertEqual(texts[0], "O usuário gosta muito de DOOM e FNAF")
            self.assertIn("O usuário se chama Ana", texts)
            self.assertNotIn("O usuário mora em Curitiba", texts)
            self.assertEqual(m.forget("curitiba"), 1)
            self.assertEqual(m.count(), 3)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            b1 = make_brain(d)
            b1.mem.add("perfil", "O usuário adora FNAF desde criança", 4)
            b1.history = [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "e aí"}]
            b1.close()
            b2 = make_brain(d)
            self.assertEqual(b2.mem.count(), 1)
            self.assertEqual(b2.history[1]["content"], "e aí")


class TestMood(unittest.TestCase):
    def test_events_decay_and_boredom(self):
        now = time.time()
        m = Mood(t=now, last=now)
        v0 = m.valence
        m.read_text("valeu, mandou bem!", now)
        self.assertGreater(m.valence, v0)
        m.read_text("você é um idiota", now)
        self.assertLess(m.valence, v0 + 0.25)
        m.tick(now + 5 * 3600)  # horas depois, volta ao normal
        self.assertAlmostEqual(m.valence, Mood.BASE_V, delta=0.02)
        self.assertIn("entediado", m.describe(now + 5 * 3600))
        self.assertIsNone(m.emotion())


class TestBrain(unittest.TestCase):
    def test_build_turn_has_context(self):
        with tempfile.TemporaryDirectory() as d:
            b = make_brain(d)
            b.mem.add("perfil", "O usuário gosta muito de DOOM e FNAF", 3)
            b.summary = "Eles já falaram sobre speedrun de DOOM."
            b.history = [{"role": "assistant", "content": "órfã"}, {"role": "user", "content": "oi"},
                         {"role": "assistant", "content": "e aí"}]
            t = b.build_turn("me fala de doom")
            for piece in ("DOOM e FNAF", "Estado interno", "Situação agora", "speedrun"):
                self.assertIn(piece, t.system)
            self.assertEqual(t.history[0]["role"], "user")  # nunca começa com resposta solta
            t2 = b.build_turn("olha isso", with_screen=True, screen_desc="um terminal aberto")
            self.assertTrue(t2.user_text.startswith("[o que aparece na tela agora: um terminal aberto]"))
            self.assertTrue(b.wants_screen("olha a minha tela"))
            self.assertFalse(b.wants_screen("oi tudo bem"))

    def test_parse_memories_ignores_garbage(self):
        raw = "- perfil|5|O usuário se chama Ana\nlixo sem pipes\nsegredo|3|tipo inválido aqui\nepisodio|x|Ana terminou um jogo hoje"
        self.assertEqual(parse_memories(raw), [("perfil", 5, "O usuário se chama Ana"),
                                               ("episodio", 3, "Ana terminou um jogo hoje")])
        self.assertEqual(parse_memories("NADA"), [])

    def test_consolidate_extracts_and_summarizes(self):
        with tempfile.TemporaryDirectory() as d:
            llm = FakeLLM(["perfil|5|O usuário se chama Ana e adora FNAF\nepisodio|2|Ana terminou um jogo hoje cedo",
                           "Ana e o Bututu conversaram sobre jogos."])
            b = make_brain(d, llm, working_messages=4)
            b.history = sum(([{"role": "user", "content": f"pergunta {i}"},
                              {"role": "assistant", "content": f"[emoção:feliz] resposta {i}"}] for i in range(7)), [])
            b.pending = 3
            b.consolidate()
            self.assertEqual(b.mem.count(), 2)
            self.assertEqual(b.summary, "Ana e o Bututu conversaram sobre jogos.")
            self.assertEqual(len(b.history), 4)
            self.assertEqual(b.pending, 0)
            self.assertNotIn("[emoção", llm.calls[0][1])  # etiquetas não vão para o extrator
            self.assertEqual(llm.calls[0][2]["temperature"], 0.2)

    def test_consolidate_failure_backs_off(self):
        with tempfile.TemporaryDirectory() as d:
            class Broken(FakeLLM):
                def chat(self, *a, **k):
                    raise RuntimeError("ollama fora do ar")
            b = make_brain(d, Broken())
            b.history = [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "e aí"}]
            b.pending = 6
            b.consolidate()
            self.assertEqual(b.pending, 6)  # tenta de novo depois
            self.assertGreater(b._retry_at, time.time() + 60)
            self.assertIn("fora do ar", b.last_error)

    def test_feedback_files_and_dataset_filter(self):
        with tempfile.TemporaryDirectory() as d:
            b = make_brain(d)
            self.assertIn("Ainda não", b.feedback("bom"))
            t = b.build_turn("oi")
            b.after_reply(t, "[emoção:feliz] resposta ruim")
            self.assertEqual(b.last_turn, ("oi", "[emoção:feliz] resposta ruim"))
            b.feedback("bom")
            b.feedback("corrigir", "[emoção:triste] resposta certa")
            data = Path(d) / "data"
            ex = [json.loads(x) for x in (data / "exemplos.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(ex), 2)
            pref = json.loads((data / "preferencias.jsonl").read_text(encoding="utf-8"))
            self.assertEqual((pref["rejected"], pref["chosen"]), ("[emoção:feliz] resposta ruim", "[emoção:triste] resposta certa"))
            self.assertEqual(b.history[-1]["content"], "[emoção:triste] resposta certa")
            self.assertTrue((data / "chatlog.jsonl").exists())

    def test_spontaneous_rules(self):
        with tempfile.TemporaryDirectory() as d:
            b = make_brain(d, spontaneous=True, quiet_hours=[0, 0])
            self.assertIsNone(b.tick())  # acabou de abrir: sem tédio
            old = time.time() - 7200
            b.last_chat = b.mood.last = b._spont_at = old
            prompt = b.tick()
            self.assertTrue(prompt and prompt.startswith("(Você está entediado"))
            self.assertIsNone(b.tick())  # limite de frequência
            b.last_chat = b.mood.last = b._spont_at = old
            self.assertIsNone(b.tick(busy=True))  # ocupado: não interrompe
            b.bc["quiet_hours"] = [0, 24]
            self.assertIsNone(b.tick())  # horário de silêncio
            b.spontaneous = False
            self.assertIsNone(b.tick())

    def test_pending_survives_restart_and_clear_recent(self):
        with tempfile.TemporaryDirectory() as d:
            b1 = make_brain(d)
            b1.after_reply(b1.build_turn("oi, eu me chamo Ana"), "e aí, Ana!")
            self.assertEqual(b1.pending, 1)
            b2 = make_brain(d)
            self.assertEqual((b2.pending, len(b2.history)), (1, 2))  # sessão curta não perde a vez de virar memória
            b2.mem.add("perfil", "O usuário se chama Ana", 5)
            b2.summary = "resumo"
            self.assertIn("limpa", b2.clear_recent())
            self.assertEqual((b2.history, b2.pending, b2.mem.count(), b2.summary), ([], 0, 1, "resumo"))

    def test_forget_all(self):
        with tempfile.TemporaryDirectory() as d:
            b = make_brain(d)
            b.mem.add("perfil", "O usuário gosta de DOOM", 4)
            b.history, b.summary = [{"role": "user", "content": "oi"}], "resumo"
            b.forget("tudo")
            self.assertEqual((b.mem.count(), b.history, b.summary), (0, [], ""))


if __name__ == "__main__":
    unittest.main()
