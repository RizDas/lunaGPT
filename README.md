# lunaGPT

A lightweight experimental repository for learning and prototyping language-model and voice-assistant components in Python.

This project is **not** a packaged library yet; it is a collection of standalone scripts that cover:

- Character-level GPT training experiments (`gpt/`)
- A GPT-2 architecture reimplementation + pretrained-weight loading demo (`gpt2/`)
- Simple speech input/output building blocks (`head/`)
- Placeholder memory files (`memory/`)

---

## Repository Structure

```text
lunaGPT/
├── gpt/
│   ├── input.txt
│   ├── bigram.py
│   ├── v2.py
│   ├── v2.2_noDropout.py
│   ├── v2.5.py
│   ├── v3.py
│   └── check.py
├── gpt2/
│   ├── train_gpt2.py
│   └── (test)dummy.ipynb
├── head/
│   ├── ear.py
│   └── mouth.py
└── memory/
    ├── cache.txt
    ├── learning.txt
    └── saved.txt
```

---

## What Each Part Does

## 1) `gpt/` — Character-level model progression

This folder contains a step-by-step progression from a bigram baseline to deeper transformer models, trained on the included Tiny Shakespeare corpus (`gpt/input.txt`).

### `gpt/bigram.py`
- Minimal bigram language model (`nn.Embedding(vocab_size, vocab_size)`) that predicts next character from current character.
- Uses fixed context length (`block_size=8`) and samples text after training.
- Good as a baseline and sanity check.

> Note: it loads `input.txt` using a relative path without the `gpt/` prefix, so run it from inside `gpt/` (or adjust the path).

### `gpt/v2.py`
- Adds a full decoder-only transformer stack:
  - token + positional embeddings
  - masked self-attention with multiple heads
  - feed-forward layers
  - residual + layernorm
  - dropout
- Trains and then generates characters.

### `gpt/v2.2_noDropout.py`
- Similar to `v2.py` but removes dropout layers.
- Also increases context and batch size versus `v2.py`.

### `gpt/v2.5.py`
- A larger variant of `v2.py`:
  - larger `block_size`
  - larger embedding dimension
  - larger batch size
  - higher learning rate

### `gpt/v3.py`
- Biggest local character model in this repo:
  - much larger context (`block_size=256`)
  - larger embedding (`n_embd=192`)
  - more heads (`n_head=6`)
  - lower learning rate for stability

### `gpt/check.py`
- Tiny CUDA availability check script.

---

## 2) `gpt2/train_gpt2.py` — GPT-2 architecture and generation demo

This script defines a GPT-2-like model from scratch (`CausalSelfAttention`, `MLP`, `Block`, `GPTConfig`, `GPT`) and then attempts to load Hugging Face GPT-2 weights via `from_pretrained`.

It finally runs top-k sampling generation from a fixed prompt.

### Important behavior notes
- Requires external packages: `transformers` and `tiktoken` in addition to `torch`.
- First run may download model weights from Hugging Face (internet required).
- The weight-loading loop currently only copies a subset of parameters (transposed list), so output quality may differ from a fully-copied checkpoint.

---

## 3) `head/` — Voice I/O primitives

### `head/ear.py`
- Captures microphone input via `speech_recognition`.
- Uses Google speech recognition API through `recognize_google`.
- Includes configurable thresholds for ambient noise and pause handling.

### `head/mouth.py`
- Converts text to speech using `edge-tts`.
- Saves temporary audio and plays it back via `pygame`.
- Cleans up generated file afterwards.

### Runtime considerations for voice scripts
- You need a microphone/audio device.
- You may need OS-level audio dependencies for `pygame` and microphone backends.
- `speech_recognition` often needs `PyAudio` installed.
- `recognize_google` and `edge-tts` require internet access.

---

## 4) `memory/` — Placeholder memory store

The three files are currently empty and appear intended as simple text-based persistence stubs:

- `memory/cache.txt`
- `memory/learning.txt`
- `memory/saved.txt`

---

## Setup

## Prerequisites
- Python 3.10+
- `pip`
- (Optional but recommended) CUDA-capable GPU for training scripts

## Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers tiktoken speechrecognition colorama edge-tts pygame pyaudio
```

> `pyaudio` may require platform-specific system packages.

---

## How to Run

From repository root:

```bash
# Transformer character model variants
python gpt/v2.py
python gpt/v2.2_noDropout.py
python gpt/v2.5.py
python gpt/v3.py

# CUDA check
python gpt/check.py

# GPT-2 architecture + sampling demo
python gpt2/train_gpt2.py

# Voice modules
python head/ear.py
python head/mouth.py
```

For the bigram script:

```bash
cd gpt
python bigram.py
```

---

## Known Issues / Cleanup Opportunities

- `gpt/check.py` uses `if torch.cuda.is_available:` (function object) instead of `if torch.cuda.is_available():`.
- Script style is currently monolithic; extracting reusable modules/config files would improve maintainability.
- There is no dependency lockfile (`requirements.txt`, `pyproject.toml`, etc.) yet.
- No automated tests are included.
- Path handling is inconsistent (`bigram.py` path behavior differs from other GPT scripts).

---

## Suggested Next Steps

1. Add `requirements.txt` or `pyproject.toml`.
2. Refactor model code into importable modules.
3. Add CLI arguments for hyperparameters and checkpoint paths.
4. Add periodic checkpoint save/load support.
5. Add unit tests for tokenization, masking, and generation shape checks.
6. Add a top-level launcher script integrating `head/` + model inference.

---

## License

No license file is currently present in this repository. Add one before distributing.
