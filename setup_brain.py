"""Prepara o cérebro: confere o Ollama, baixa os modelos e cria o modelo 'bututu' (personalidade embutida).
Uso:  python setup_brain.py            (faz tudo)
      python setup_brain.py --check    (só diagnostica, sem baixar nada)
      python setup_brain.py --no-create
O modelo 'bututu' serve para usar a personalidade fora do app (ollama run bututu) e é o nome onde você vai colocar
o modelo treinado depois (training/finetune_lora.py). O app sempre envia o próprio prompt, com memórias e humor."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ok(msg):
    print(f"[ok] {msg}")


def bad(msg):
    print(f"[x]  {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="só diagnostica")
    ap.add_argument("--no-create", action="store_true", help="não cria o modelo 'bututu'")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    url, base, embed = cfg["llm"]["base_url"].rstrip("/"), cfg["llm"]["model"], cfg["brain"]["embed_model"]

    exe = shutil.which("ollama")
    if not exe:
        bad("Ollama não encontrado no PATH. Instale em https://ollama.com e rode de novo.")
        return 1
    ok(f"Ollama instalado: {exe}")
    try:
        have = {m["name"] for m in requests.get(url + "/api/tags", timeout=3).json().get("models", [])}
    except requests.RequestException:
        bad(f"O servidor do Ollama não responde em {url}. Abra o app do Ollama (ou rode: ollama serve).")
        return 1
    ok(f"Servidor do Ollama respondendo ({len(have)} modelo(s) instalado(s))")

    def has(name):
        return name in have or f"{name}:latest" in have

    for name, why in ((base, "modelo de conversa/visão"), (embed, "memória (busca por significado)")):
        if has(name):
            ok(f"{name} instalado ({why})")
        elif args.check:
            bad(f"{name} falta ({why}). Rode: ollama pull {name}")
        else:
            print(f"Baixando {name} ({why})...")
            if subprocess.run([exe, "pull", name]).returncode != 0:
                bad(f"não consegui baixar {name}")
                return 1
            ok(f"{name} instalado")
    try:
        requests.post(url + "/api/embed", json={"model": embed, "input": ["teste"]}, timeout=30).raise_for_status()
        ok("busca por significado funcionando")
    except requests.RequestException:
        bad("embeddings indisponíveis: a memória usará o modo offline simples (funciona, mas entende menos)")

    if args.check or args.no_create:
        return 0
    persona = (ROOT / "persona" / "persona.md").read_text(encoding="utf-8").replace("{name}", cfg["name"])
    out = ROOT / "brain"
    out.mkdir(exist_ok=True)
    (out / "Modelfile").write_text(
        f"FROM {base}\nPARAMETER temperature {cfg['llm']['temperature']}\nPARAMETER num_ctx {cfg['llm'].get('num_ctx', 4096)}\n"
        f'SYSTEM """{persona}"""\n'
        "# Depois do treino (training/finetune_lora.py), aponte o FROM para o .gguf gerado, ou aplique o adaptador:\n"
        "# ADAPTER ./caminho/do/adaptador\n", encoding="utf-8")
    if subprocess.run([exe, "create", "bututu", "-f", str(out / "Modelfile")]).returncode != 0:
        bad("não consegui criar o modelo 'bututu'")
        return 1
    ok("modelo 'bututu' criado (teste: ollama run bututu)")
    print("\nPronto. Para o app usar o 'bututu', troque em config.yaml:  llm.model: bututu\nDepois: python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
