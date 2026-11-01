"""Testa a camada que conversa com o Ollama usando um servidor de mentira (mesmo formato de /api/chat, /api/tags, /api/embed)."""
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.brain import Brain  # noqa: E402
from core.brain.memory import OllamaEmbedder, make_embedder  # noqa: E402
from core.llm import LocalLLM  # noqa: E402

LAST = {}
MODE = {"error": False}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json({"models": [{"name": "gemma3:4b"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/embed":
            vecs = [[float(t.lower().count(c)) + 0.01 for c in "abcdefghijklmnopqrstuvwxyz"] for t in req["input"]]
            return self._json({"embeddings": vecs})
        LAST.update(req)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        lines = [{"error": "model not found"}] if MODE["error"] else [
            {"message": {"role": "assistant", "content": "Oi, "}, "done": False},
            {"message": {"role": "assistant", "content": "MEU DEUS!"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True}]
        for ln in lines:
            self.wfile.write((json.dumps(ln) + "\n").encode())


class TestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        cls.cfg["llm"]["base_url"] = cls.url

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_stream_and_options(self):
        MODE["error"] = False
        llm = LocalLLM(self.cfg)
        self.assertTrue(llm.alive())
        self.assertEqual("".join(llm.stream("sistema", [{"role": "user", "content": "a"}], "olá", image_b64="IMG")), "Oi, MEU DEUS!")
        self.assertEqual(LAST["messages"][0], {"role": "system", "content": "sistema"})
        self.assertEqual(LAST["messages"][-1]["images"], ["IMG"])
        self.assertEqual(LAST["options"]["num_ctx"], 4096)
        self.assertEqual(llm.chat("s", [], "x", temperature=0.2, max_tokens=50, model="auxiliar"), "Oi, MEU DEUS!")
        self.assertEqual((LAST["model"], LAST["options"]["temperature"], LAST["options"]["num_predict"]), ("auxiliar", 0.2, 50))

    def test_error_from_server(self):
        MODE["error"] = True
        try:
            with self.assertRaises(RuntimeError):
                LocalLLM(self.cfg).chat("s", [], "x")
        finally:
            MODE["error"] = False

    def test_brain_uses_ollama_embeddings_then_falls_back(self):
        emb = make_embedder(self.url, "nomic-embed-text", "auto")
        self.assertIsInstance(emb, OllamaEmbedder)
        with tempfile.TemporaryDirectory() as d:
            cfg = yaml.safe_load(yaml.safe_dump(self.cfg))
            cfg["home"] = str(Path(d) / "home")
            b = Brain(cfg, LocalLLM(cfg), data_dir=Path(d) / "data")
            self.assertEqual(b.mem.embedder.name, "ollama:nomic-embed-text")
            self.assertTrue(b.mem.add("perfil", "O usuário gosta muito de DOOM e FNAF", 3))
            self.assertTrue(b.mem.recall("doom e fnaf"))
            b.mem.embedder.url = "http://127.0.0.1:9"  # servidor 'caiu': a busca não pode quebrar
            self.assertIsInstance(b.mem.recall("doom"), list)
            self.assertTrue(b.mem.add("perfil", "O usuário joga FNAF toda sexta", 3))  # guarda com o fallback


if __name__ == "__main__":
    unittest.main()
