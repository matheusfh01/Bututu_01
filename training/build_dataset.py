"""Transforma suas conversas reais (data/chatlog.jsonl) + exemplos escritos à mão (data/exemplos.jsonl)
em data/train.jsonl / data/val.jsonl no formato de chat, já com a persona como mensagem de sistema.

Formato de data/exemplos.jsonl (uma linha por exemplo):
  {"messages": [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "[emoção:feliz] MEU DEUS DO CÉU, apareceu!"}]}

Dica: 150 a 500 exemplos BONS valem mais que 5.000 medianos. Revise o chatlog e apague/edite o que não é a cara do personagem.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def read(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def ok(row):
    msgs = row.get("messages", [])
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
        return False
    reply = msgs[-1]["content"].strip()
    return 2 < len(reply) < 700 and not reply.startswith("[falha") and "[silencio]" not in reply.lower()


def collect():
    """Exemplos válidos, sem duplicatas e sem as respostas que você corrigiu com /corrigir (já com a persona no início)."""
    persona = (ROOT / "persona" / "persona.md").read_text(encoding="utf-8").replace("{name}", "Bututu")
    # respostas que você corrigiu nunca entram no treino (só a versão certa, que está em exemplos.jsonl)
    rejected = {(r["prompt"], r["rejected"]) for r in read(DATA / "preferencias.jsonl")}
    rows = [r for r in read(DATA / "exemplos.jsonl") + read(DATA / "chatlog.jsonl")
            if ok(r) and (r["messages"][-2]["content"], r["messages"][-1]["content"]) not in rejected]
    seen, out = set(), []
    for r in rows:  # remove duplicatas
        key = json.dumps(r["messages"], ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append({"messages": [{"role": "system", "content": persona}] + r["messages"]})
    return out


def main():
    out = collect()
    random.seed(7)
    random.shuffle(out)
    cut = max(1, int(len(out) * 0.95)) if len(out) > 20 else len(out)
    for name, part in (("train", out[:cut]), ("val", out[cut:])):
        with open(DATA / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} exemplos -> train: {cut}, val: {len(out) - cut}")
    if len(out) < 100:
        print("Aviso: poucos exemplos. Converse mais com o Bututu e/ou escreva exemplos em data/exemplos.jsonl.")


if __name__ == "__main__":
    main()
