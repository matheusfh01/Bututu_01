# Bututu Local

Assistente de desktop **100% local** (sem chave de API e sem internet depois da instalação), reorganizado a partir do
`bututu_legacy.py`: personalidade, visão de tela, voz, audição, emoções, soundboard, ações e agora um **cérebro**.

```
main.py              junta tudo (comandos: /tela /limpar /som /voz /calar /emocao /ver /sair + os do cérebro)
setup_brain.py       diagnostica o Ollama, baixa os modelos e cria o modelo "bututu"
treinar.py           treina a personalidade em um comando (dataset -> LoRA -> GGUF -> Ollama)
config.yaml          todas as configurações
persona/persona.md   personalidade (extraída do código antigo; edite à vontade)
core/brain/          O CÉREBRO (memória, humor, situação, decisões, aprendizado com você)
core/llm.py          conversa com o modelo local via Ollama (streaming + imagem)
core/vision.py       captura contínua da tela (~2 fps), detecção de mudança, quadro mais recente sob demanda
core/tts.py          voz local (Piper / Kokoro / XTTS) + soundboard, numa fila só
core/stt.py          microfone -> texto (faster-whisper), opcional
core/actions.py      ações permitidas (lista fechada, sem shell, com confirmação)
core/ui.py           sprite sempre no topo, balão e chat (usa ~/.Bututu/sprites do v2)
training/            dataset + LoRA da personalidade + guia da voz
tests/               14 testes automáticos do cérebro e do protocolo com o Ollama
```

## Instalar
1. Instale o [Ollama](https://ollama.com) (versão recente) e abra o app.
2. `pip install -r requirements.txt` (Linux: `sudo apt install python3-tk espeak-ng`; use sessão X11).
3. `python setup_brain.py` baixa `gemma3:4b` (conversa + visão) e `nomic-embed-text` (memória), e cria o modelo `bututu`.
   Para só diagnosticar: `python setup_brain.py --check`.
4. Baixe uma voz Piper pt-BR (`.onnx` + `.onnx.json`) para `~/.Bututu/voices/piper/` e ajuste `voice.piper_model`.
5. `python main.py`. Sprites (opcional): `idle_0.png`, `thinking_0.png`, `speaking_0.png`, `happy_0.png`... em
   `~/.Bututu/sprites/`. Sons: `.wav/.mp3` em `~/.Bututu/sounds/` (a IA toca com `[som:nome]`).

## O cérebro (`core/brain/`)
O modelo de linguagem só gera a fala. Tudo o que torna o Bututu "alguém" passa pelo cérebro:

```
sentidos (texto, tela, voz) ─▶ CÉREBRO ─▶ corpo (fala, sprite, sons, ações)
  ANTES da resposta : busca memórias relevantes, lê o humor e a situação (hora, disco cheio...), monta o prompt
  DEPOIS            : atualiza humor e histórico, guarda a troca para treino
  EM SEGUNDO PLANO  : extrai memórias novas, resume o passado, (opcional) puxa assunto sozinho
```
- **Memória de longo prazo** (`~/.Bututu/brain.db`, SQLite): fatos sobre você, preferências e episódios, com busca por
  significado (embeddings do Ollama; sem eles, cai num modo offline mais simples). Nunca guarda senhas, chaves, CPF
  ou cartões. A cada 6 trocas, quando você para de falar, o próprio modelo extrai o que vale lembrar e resume o passado.
- **Humor**: dois eixos (humor e energia) que reagem a elogios, ofensas e à emoção da própria fala, e voltam ao normal
  com o tempo. O tédio vem do tempo sem conversa.
- **Situação real**: dia, hora, tempo desde a última conversa e disco do PC quando passa de 80% (o drama de espaço em
  disco do personagem agora tem motivo de verdade).
- **Espontâneo** (desligado por padrão, `/espontaneo` liga): se estiver entediado, puxa assunto no máximo a cada 25 min,
  nunca entre 00h e 07h e nunca enquanto você está conversando.
- **Aprende com você**: `/bom` guarda a última resposta como bom exemplo; `/corrigir <resposta certa>` guarda a
  correção (e a resposta errada, para ser excluída do treino). Isso alimenta `data/exemplos.jsonl`.

Comandos do cérebro: `/memoria` (lista), `/esquecer tudo` ou `/esquecer palavra`, `/humor`, `/espontaneo`, `/bom`,
`/corrigir`, `/limpar` (só a conversa recente). Tudo fica no seu PC.

## Como deixar a IA treinando
**A IA não treina sozinha enquanto você a usa.** Memória e humor aprendem na hora; o *modelo* só muda quando você roda
um treino. O ciclo é: usar -> juntar exemplos -> treinar -> trocar o modelo. Sempre treine a partir do modelo base com
**todos** os exemplos acumulados (não por cima do modelo já treinado).

| Parte | Precisa treinar? | Como |
|---|---|---|
| Visão de tela | **Não.** Usa um modelo de visão já treinado | `llm.model` ou `vision.describer_model` |
| Memória e humor | **Não.** Aprendem sozinhos usando o app | `/memoria`, `/esquecer` |
| Personalidade | Funciona **sem treino** (prompt). Treino deixa mais fiel e consistente | abaixo |
| Voz própria | Opcional | `training/VOZ.md` |

### 1. Junte exemplos (o que mais importa)
Meta: **150+ exemplos bons** (poucos e bons valem mais que muitos medianos). Fontes: as conversas (`data/chatlog.jsonl`),
`/bom` e `/corrigir` no app, e exemplos escritos à mão em `data/exemplos.jsonl` (formato no topo de
`training/build_dataset.py`). Revise: o que você corrige com `/corrigir` nunca entra no treino na versão errada.
Confira com: `python treinar.py --check`.

### 2. Escolha o caminho
**A) Unsloth Studio (sem código, recomendado com GPU NVIDIA).** Folga sugerida: 8 GB de VRAM para um modelo de 3B
(o guia do Unsloth lista 3,5 GB como mínimo absoluto). A documentação atual também cita AMD, Intel e Mac; confira a sua
placa em unsloth.ai/docs. O Studio está em beta, então nomes de botões podem mudar.
1. Instale. Windows (PowerShell): `irm https://unsloth.ai/install.ps1 | iex`. Mac/Linux/WSL: `curl -fsSL https://unsloth.ai/install.sh | sh`
2. Abra: `unsloth studio -H 127.0.0.1 -p 8888` e acesse http://localhost:8888 (não use `0.0.0.0`: expõe na sua rede).
3. `python training/build_dataset.py` gera `data/train.jsonl`. No Studio: **Train** -> modelo base (um *Instruct* pequeno,
   de 3B a 4B) -> dataset `data/train.jsonl` (formato `messages`, ele detecta sozinho) -> QLoRA -> **Start Training**.
4. **Export** -> o modelo treinado -> **GGUF** -> `Q4_K_M` -> salvar localmente.
5. Registre no Ollama: crie um arquivo `Modelfile` com `FROM ./nome-do-arquivo.gguf` e rode `ollama create bututu -f Modelfile`.

**B) `python treinar.py` (um comando, em ambiente com GPU).** Feche o Bututu antes. Instale no ambiente de treino:
`pip install unsloth trl peft datasets`. Ele confere dados e GPU, libera a GPU do Ollama, monta o dataset, treina,
exporta e cria o modelo `bututu` (o anterior vira `bututu-anterior`, para você poder voltar). Se faltar memória:
`python treinar.py --batch 1`. **Não consegui executar o treino em GPU aqui**; se quebrar, use o caminho A.

**C) Sem GPU: Google Colab (ou aluguel de GPU).** Abra um notebook com GPU, envie a pasta do projeto, rode
`pip install unsloth trl peft datasets` e `python treinar.py --force`, e baixe o `.gguf` de `training/out/`. A GPU gratuita
tem limite de tempo e a sessão pode cair. **Atenção: seus dados de conversa sobem para a nuvem.** Revise o `chatlog`
antes e apague o que for pessoal.

### 3. Depois de treinar
O modelo treinado é **só texto**. Em `config.yaml`:
```yaml
llm:
  model: bututu
vision:
  describer_model: gemma3:4b   # sem isso, /tela deixa de funcionar
brain:
  helper_model: gemma3:4b      # extrai memórias com um modelo neutro
```
Teste conversando. Se ficou pior, volte com `llm.model: bututu-anterior` (ou o modelo base) e junte mais exemplos.
Deixe o PC sem suspensão durante o treino e feche jogos e o Bututu para liberar a GPU.

## Limites honestos
- **"Tempo real" = quadros, não vídeo.** A tela é capturada ~2 vezes por segundo, mas o modelo só analisa o quadro mais
  recente quando você pede ou quando a tela muda (modo automático, desligado por padrão). A resposta leva de 1 a vários
  segundos, dependendo da GPU.
- Modelos locais pequenos erram mais que os da nuvem: leem texto miúdo pior e às vezes extraem memórias erradas.
  Confira com `/memoria` e apague o que estiver errado com `/esquecer`.
- **O que foi testado:** 14 testes automáticos (`python -m unittest discover -s tests`) cobrindo memória, humor, prompt,
  consolidação, feedback, espontâneo e o protocolo com o Ollama, este último contra um servidor de mentira.
  **Não testado aqui:** Ollama de verdade, Tk com monitor, microfone, vozes e GPU. Espere ajustes na primeira execução
  (caminhos das vozes, dispositivo de áudio etc.).
- `finetune_lora.py` é ponto de partida: as bibliotecas de treino mudam rápido.

## Privacidade e segurança
- Capturas de tela ficam na memória e só vão ao modelo **local**; nada é gravado em disco. O `chatlog.jsonl` e o
  `brain.db` guardam só texto, no seu PC.
- Ações só abrem apps listados em `actions.apps`, sem shell e com confirmação (`actions.confirm`).
- Clone de voz: só a sua ou com permissão de quem fala.
