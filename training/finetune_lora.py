"""Fine-tuning LoRA (QLoRA) da PERSONALIDADE com Unsloth.
    python training/finetune_lora.py [--base unsloth/Qwen2.5-3B-Instruct] [--epochs 3]
Requer GPU e, no ambiente de treino:  pip install unsloth trl peft datasets   (ou use o Colab / o Unsloth Studio)

Conferido em 31/10/2026 contra o guia de fine-tuning atual do Unsloth (SFTTrainer + SFTConfig com dataset_text_field e
max_seq_length). Eu NÃO consegui executá-lo em GPU, então se algo falhar, compare com o guia oficial; o plano B sem
código é o Unsloth Studio (README, "Caminho A").

Use um modelo SÓ TEXTO pequeno (Instruct). A visão fica com outro modelo (config.yaml > vision.describer_model).
"""
import argparse
from pathlib import Path

from unsloth import FastLanguageModel, is_bfloat16_supported  # unsloth sempre ANTES de trl/transformers
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training" / "out"
# marcações de "início da fala do usuário / do assistente" por família de modelo (para treinar só nas respostas)
MARKERS = {
    "qwen": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "llama": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "gemma": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen2.5-3B-Instruct")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2, help="se faltar memória da GPU (OOM), use 1")
    a = ap.parse_args()

    files = {"train": str(ROOT / "data" / "train.jsonl")}
    val = ROOT / "data" / "val.jsonl"
    has_val = val.exists() and val.stat().st_size > 0  # arquivo vazio quebraria o load_dataset
    if has_val:
        files["val"] = str(val)
    ds = load_dataset("json", data_files=files)

    model, tok = FastLanguageModel.from_pretrained(a.base, max_seq_length=a.max_seq, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth")
    ds = ds.map(lambda b: {"text": [tok.apply_chat_template(m, tokenize=False) for m in b["messages"]]},
                batched=True, remove_columns=["messages"])

    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds["train"], eval_dataset=ds["val"] if has_val else None,
        args=SFTConfig(dataset_text_field="text", max_seq_length=a.max_seq, per_device_train_batch_size=a.batch,
                       gradient_accumulation_steps=8 // a.batch, num_train_epochs=a.epochs, learning_rate=2e-4,
                       warmup_ratio=0.05, lr_scheduler_type="cosine", optim="adamw_8bit", weight_decay=0.01,
                       fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=5,
                       eval_strategy="epoch" if has_val else "no", output_dir=str(OUT / "ckpt"),
                       report_to="none", seed=3407))
    markers = next((m for k, m in MARKERS.items() if k in a.base.lower()), None)
    if markers:  # aprende a RESPONDER como o personagem, não a repetir o que o usuário disse
        trainer = train_on_responses_only(trainer, instruction_part=markers[0], response_part=markers[1])
    else:
        print("Aviso: família do modelo desconhecida; treinando no texto inteiro (funciona, mas é menos eficiente).")
    trainer.train()

    model.save_pretrained(str(OUT / "lora"))  # adaptador pequeno: permite reexportar sem treinar de novo
    tok.save_pretrained(str(OUT / "lora"))
    try:
        model.save_pretrained_gguf(str(OUT / "gguf"), tok, quantization_method="q4_k_m")
    except Exception as exc:  # noqa: BLE001
        print(f"\nA exportação GGUF falhou ({exc}).\nO treino está salvo em {OUT / 'lora'}; reexporte pelo guia "
              "'Saving to GGUF' do Unsloth ou pelo Unsloth Studio.")
        raise SystemExit(2)
    print(f"\nTreino concluído. GGUF em {OUT}. Falta registrar no Ollama (o treinar.py faz isso).")


if __name__ == "__main__":
    main()
