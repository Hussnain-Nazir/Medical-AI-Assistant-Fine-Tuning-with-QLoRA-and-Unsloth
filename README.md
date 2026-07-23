# Medical AI Assistant Fine-Tuning with QLoRA and Unsloth

> Fine-tune a medical AI assistant on a free Google Colab T4 GPU using
> 4-bit Quantisation (QLoRA) and Low-Rank Adaptation (LoRA) via Unsloth.

---

## Project Overview

| Item | Detail |
|---|---|
| **Notebook** | `Medical_Fine_Tuning.ipynb` |
| **Base model** | `unsloth/Llama-3.2-3B-Instruct` |
| **Method** | QLoRA — 4-bit NF4 quantisation + LoRA adapters |
| **Dataset** | `medalpaca/medical_meadow_medqa` (USMLE-style Q&A) |
| **Framework** | Unsloth + HuggingFace TRL (`SFTTrainer` / `SFTConfig`) |
| **Hardware** | Google Colab free tier — T4 GPU (15 GB VRAM) |
| **Training time** | ~20 min (500 samples, 1 epoch) |

---

## What's in This Project

```
Medical-AI-Assistant-Fine-Tuning/
├── Medical-AI-Assistant-Fine-Tuning.ipynb   ← Notebook
├── outputs/                                  ← Created by the notebook at runtime
│   ├── medical_lora_adapter/                 ← Trained adapter weights (download this)
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── training_loss_curve.png               ← Loss plot
│   └── train_results.json                    ← Training metrics
├── requirements.txt                         
└── README.md                                 
```

> **The `src/` folder, `.txt` guides, and diagram files from earlier versions
> are not part of this project.** Everything is self-contained inside the notebook.

---

## Quick Start

### Step 1 — Open in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook** → select `Medical_AI_Assistant_Fine_Tuning.ipynb`
3. Click **Runtime → Change runtime type → T4 GPU → Save**

### Step 2 — Run the Notebook

Click **Runtime → Run all** — or run each cell top to bottom.

> After Cell 2 finishes installing, Colab may show a **Restart Runtime** button.
> Click it, then re-run from Cell 1.

### Step 3 — Download Your Results

Once training completes, go to the **Files panel** (folder icon, left sidebar):
- Download `/content/medical_lora_adapter/` — your trained model
- Download `/content/training_loss_curve.png` — your loss chart

---

## Notebook Cells — What Each One Does

| Cell | Purpose |
|---|---|
| **Cell 1** | GPU check — verifies T4 is connected before anything else |
| **Cell 2** | Install all dependencies (Unsloth, unsloth_zoo, TRL, HuggingFace stack) |
| **Cell 3** | Imports + global config (`CFG` dictionary with all hyperparameters) |
| **Cell 4** | `gpu_stats()` and `free_memory()` helper functions |
| **Cell 5** | Load `Llama-3.2-3B-Instruct` in 4-bit NF4 quantisation |
| **Cell 6** | Attach LoRA adapters to 7 attention/MLP weight matrices |
| **Cell 7** | Download and preview `medalpaca/medical_meadow_medqa` dataset |
| **Cell 8** | Format dataset into Llama-3 chat template, filter, train/val split |
| **Cell 9** | Token length distribution check + histogram plot |
| **Cell 10** | Configure `SFTTrainer` with `SFTConfig` |
| **Cell 11** | Run training loop |
| **Cell 12** | Plot training + validation loss curve, save PNG |
| **Cell 13** | Save LoRA adapter weights + tokenizer to disk |
| **Cell 14** | Switch model to inference mode, run 3 test medical queries |
| **Cell 15** | Final VRAM summary and list of saved artefacts |
| **Cell 16** | *(Optional)* Export to GGUF for Ollama / LM Studio |
| **Cell 17** | *(Optional)* Push adapter to HuggingFace Hub |

---

## Key Concepts

### What is QLoRA?

QLoRA = **Q**uantisation + **Lo**w-**R**ank **A**daptation.

1. The base model weights are loaded in **4-bit NF4** format (frozen) — reducing
   Llama-3.2-3B from ~6 GB to ~1.7 GB VRAM.
2. Small trainable **LoRA adapter matrices** (A and B) are added beside the frozen
   weights. Only these are updated during training — about **0.8% of all parameters**.
3. Combined effect: fine-tuning quality close to full training at a fraction of the
   memory cost.

| Method | VRAM needed | Fits free T4? |
|---|---|---|
| Full fp16 fine-tune | ~12 GB | Borderline |
| LoRA (fp16 base) | ~8 GB | Borderline |
| **QLoRA (4-bit + LoRA)** | **~5 GB** | **Yes** |

### Why Llama-3.2-3B-Instruct?

- Small enough for the free T4 even without quantisation
- Already instruction-tuned — strong medical Q&A baseline
- 128k token vocabulary handles medical terminology well
- Unsloth ships hand-optimised Triton kernels for Llama-3 → ~2× training speed
- Open weights — no API key required

### Why medalpaca/medical_meadow_medqa?

- USMLE Step 1-3 style clinical questions
- Already in `{instruction, input, output}` format — minimal preprocessing
- ~11k curated pairs, Apache 2.0 licence
- Covers pharmacology, anatomy, pathology, clinical diagnosis

---

## Configuration (CFG Dictionary — Cell 3)

```python
CFG = {
    'model_name'            : 'unsloth/Llama-3.2-3B-Instruct',
    'max_seq_length'        : 2048,
    'load_in_4bit'          : True,
    'lora_r'                : 16,      # LoRA rank
    'lora_alpha'            : 32,      # scaling = alpha/rank = 2.0
    'lora_dropout'          : 0.05,
    'dataset_name'          : 'medalpaca/medical_meadow_medqa',
    'max_samples'           : 500,     # ← increase for better quality
    'num_train_epochs'      : 1,       # ← increase for better quality
    'per_device_batch_size' : 2,
    'grad_accum_steps'      : 4,       # effective batch = 2×4 = 8
    'learning_rate'         : 2e-4,
    'warmup_steps'          : 50,
}
```

**To run a full-quality training:** set `max_samples=2000` and `num_train_epochs=2`.
Estimated time: ~60-90 min on free T4.

---

## Dependencies

All installed automatically by **Cell 2** of the notebook. No manual pip installs needed.

| Package | Version | Purpose |
|---|---|---|
| `unsloth` | latest (git) | Fast QLoRA training framework |
| `unsloth_zoo` | latest | Required sub-dependency of Unsloth |
| `transformers` | ≥4.45.0 | Model loading and tokenisation |
| `trl` | ≥0.12.0 | `SFTTrainer` + `SFTConfig` |
| `peft` | ≥0.13.0 | LoRA adapter implementation |
| `bitsandbytes` | ≥0.44.0 | 4-bit quantisation + 8-bit AdamW |
| `datasets` | ≥2.18.0 | Dataset download and processing |
| `accelerate` | ≥0.34.0 | Training acceleration |
| `xformers` | latest | Memory-efficient attention |
| `sentencepiece` | ≥0.2.0 | Tokeniser support |

> **Note on `unsloth_zoo`:** This is a required internal dependency of Unsloth
> that is not auto-installed when using `--no-deps`. Cell 2 installs it explicitly.
> If you ever see `ModuleNotFoundError: No module named 'unsloth_zoo'`, run
> `!pip install unsloth_zoo` in a new cell before Cell 3.

---

## Optimisation Tips (Limited GPU)

| Goal | Change |
|---|---|
| Fastest demo (~15 min) | `max_samples=200`, `num_train_epochs=1` |
| Reduce VRAM further | `max_seq_length=1024`, `lora_r=8` |
| Better model quality | `max_samples=2000`, `num_train_epochs=2`, `lora_r=32` |
| Faster throughput | Set `packing=True` in `SFTConfig` (Cell 10) |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'unsloth_zoo'`
Cell 2 now installs this automatically. If you still see it, run `!pip install unsloth_zoo` in a new cell then continue from Cell 3.

### `CUDA out of memory`
1. Runtime → Disconnect and delete runtime → reconnect
2. Lower `max_seq_length` to `1024`
3. Lower `lora_r` to `8`
4. Lower `per_device_batch_size` to `1`

### `Restart Runtime` button appears after Cell 2
This is normal. Click it, then re-run from Cell 1.

### Training loss is NaN
Lower `learning_rate` to `1e-4`. Check dataset has no empty outputs.

### `evaluation_strategy` deprecation warning
The notebook uses `eval_strategy` (the updated parameter name in TRL ≥0.12). This warning can be safely ignored if it appears.

### Colab session disconnects
Set `save_steps=100` in Cell 10 to checkpoint more often. Resume with `trainer.train(resume_from_checkpoint=True)`.

---

## Outputs

After the notebook runs successfully you will have:

```
/content/medical_lora_adapter/
    adapter_config.json          ← LoRA hyperparameters
    adapter_model.safetensors    ← Trained adapter weights (~75 MB)
    tokenizer.json               ← Tokenizer
    tokenizer_config.json
    special_tokens_map.json

/content/training_loss_curve.png ← Loss curve chart
/content/medical_outputs/train_results.json ← Final metrics
```

Download these from the **Files panel** (left sidebar folder icon in Colab).

---

## License

**Base Model:** [Meta Llama 3.2 Community License](https://llama.meta.com/llama3_2/)  
**Dataset:** Apache 2.0 (MedAlpaca / medalpaca/medical_meadow_medqa)  
**Project:** MIT License
