"""Cliente do modelo local (Ollama). Tudo roda na sua máquina, sem chave e sem internet."""
import json
import requests


class LocalLLM:
    def __init__(self, cfg):
        c = cfg["llm"]
        self.url = c["base_url"].rstrip("/")
        self.model = c["model"]
        self.opts = {"temperature": c["temperature"], "num_predict": c["max_tokens"],
                     "num_ctx": c.get("num_ctx", 4096)}  # janela de contexto: prompt + memórias + histórico
        self.keep_alive = c.get("keep_alive", "30m")
        self.timeout = c.get("timeout", 180)

    def alive(self):
        try:
            return requests.get(self.url + "/api/tags", timeout=1.5).ok
        except requests.RequestException:
            return False

    def stream(self, system, history, user_text, image_b64=None, temperature=None, max_tokens=None, model=None):
        """Gera pedaços de texto conforme o modelo escreve (permite começar a falar antes do fim)."""
        opts = dict(self.opts)
        if temperature is not None:
            opts["temperature"] = temperature
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        user = {"role": "user", "content": user_text}
        if image_b64:
            user["images"] = [image_b64]
        body = {"model": model or self.model, "stream": True, "options": opts, "keep_alive": self.keep_alive,
                "messages": [{"role": "system", "content": system}] + history + [user]}
        with requests.post(self.url + "/api/chat", json=body, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break

    def chat(self, system, history, user_text, image_b64=None, **kw):
        return "".join(self.stream(system, history, user_text, image_b64, **kw)).strip()

    def describe(self, image_b64, question, model):
        """Usa um modelo de VISÃO separado só para descrever a tela em texto.
        Permite que o modelo de personalidade (treinado por você) seja pequeno e só texto."""
        body = {"model": model, "stream": False, "options": {"temperature": 0.2, "num_predict": 300},
                "keep_alive": self.keep_alive,
                "messages": [{"role": "user", "images": [image_b64],
                              "content": "Descreva objetivamente o que aparece nesta captura de tela "
                                         "(apps, textos importantes, o que o usuário parece fazer). "
                                         f"Pergunta do usuário, se houver: {question}"}]}
        r = requests.post(self.url + "/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
