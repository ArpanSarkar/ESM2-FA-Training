# ESM2-FA: FlashAttention Pretraining for ESM-2

This repository provides **a complete workflow for training ESM-2 models** at various sizes using **FlashAttention** via [FAESM](https://github.com/pengzhangzhi/faesm) (from the [FAPLM](https://github.com/pengzhangzhi/faplm/tree/main) ecosystem). It includes:
- **FASTA file preprocessing**: Converts raw protein sequences into tokenized, batched, and padded datasets.
- **Integration with Hugging Face's `transformers` and `accelerate`** for efficient distributed training.

---
## Setting Up the Environment

To ensure compatibility, create a **Conda environment** and install dependencies:

```bash

# Installing Torch
conda create --name faesm_training python=3.12
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge transformers datasets accelerate safetensors einops
conda install ipykernel
conda install pip
python -m ipykernel install --user --name=faesm_training

# For optional tracking
pip install wandb
wandb -login [API key]

# Installing FlashAttention
pip install flash-attn --no-build-isolation
pip install faesm[flash_attn]

# Set up `accelerate` for multi-GPU training (configure when prompted)
accelerate config
```

---
## Configuring Model & Training Hyperparameters

Before preprocessing, review and modify `src/config.yaml` as needed.

This repo follows a training setup heavily inspired by Cheng et al. (2024, bioRxiv), *Training Compute-Optimal Protein Language Models*.

### **Key Parameters to Check:**

#### **Model Parameters (`model:` in `src/config.yaml`)**

- `model_name`: the HF/FAESM checkpoint name used to load the tokenizer and `AutoConfig` (**architecture source of truth**).
- `max_position_embeddings`: maximum sequence length **including** special tokens. `src/data_processing.py` checks this matches the underlying checkpoint config.
- `max_batch_size`: token budget for preprocessing (max tokens per pre-batched batch, before padding).
- `hidden_dropout_prob`, `attention_probs_dropout_prob` (optional): if present, these override dropout values in the checkpoint config.

#### **Training Parameters (`training:` in `src/config.yaml`)**
- `learning_rate`: **4e-4**
- `gradient_accumulation_steps`: **64**
- `mlm_probability`: e.g. **0.2**
- `schedule_steps`: cosine decay horizon (LR goes from 1.0× to 0.1× by this step)
- `max_steps`: hard stop for training; LR stays flat at 0.1× after `schedule_steps`
- `warmup_steps`: fixed warmup; LR ramps 0 → 1 over this span (no auto-derive)
- `mixed_precision`: **bf16** (preferred on A100/H100) or **fp16**
- `gradient_clipping`: used as `TrainingArguments.max_grad_norm`
- `early_stopping`: enabled by default; patience 5 evals, threshold 0.0, monitors `eval_loss`
- `load_best_model_at_end` / `metric_for_best_model` / `greater_is_better`: defaults set for `eval_loss`
- `optimizer.fused` (optional): if omitted and supported by your PyTorch/CUDA build, fused AdamW is used; set `false` to force unfused

`src/config.py` is just a small helper that loads `src/config.yaml` and exposes `get_model_config()`, `get_training_config()`, and `get_data_config()`.

### How model architecture is chosen

`src/custom_model.py` loads:

- `tokenizer = AutoTokenizer.from_pretrained(model_name)`
- `config = AutoConfig.from_pretrained(model_name)`

and then instantiates `FAEsmForMaskedLM(config)`. There is **no hardcoding** of `hidden_size`, `num_layers`, etc. in YAML.

---
## Preprocessing the Data

Before training, raw **FASTA files** must be converted into **batched, tokenized, and padded datasets**.

```bash
python src/data_processing.py \
  --input_fasta "<path_to_raw_fasta>" \
  --output_dir "<path_to_processed_dataset>" \
  --chunk_size 1000000 \
  --shard_size 25000 \
  --max_aa_length 1024
```

### Preprocessing outputs

- Writes `shard-XXXXX/` directories under `--output_dir` (Hugging Face datasets saved with `Dataset.save_to_disk()`).
- Writes `<input_name>_refined.fasta` next to the input FASTA:
  - sequences longer than `max_aa_length` are split into balanced chunks
  - chunk IDs are suffixed with ranges like `/1-1022`, `/1023-end`
  - only the first chunk gets BOS/CLS; only the last chunk gets EOS; middle chunks get neither

### What’s inside a shard

Each row/example contains:

- `input_ids`
- `attention_mask`
- `labels` (present, but the MLM collator overwrites labels each step)
- `sequence_length`
- `batch_id`
- `id` (FASTA header line, including `>` and any range suffix after chunking)

Examples are written in **batch-major order** (all examples of one `batch_id` contiguous), so the streaming dataloader can reconstruct batches without `list(shard)` in RAM. Batches are formed by:
- length-sorting sequences within a window (`chunk_size`)
- packing greedily under a token budget (`model.max_batch_size`)
- padding to the longest sequence in the batch; storing `batch_id`

---
## Training ESM-2

After preprocessing, run the training script to fine-tune **ESM-2 models** with **FlashAttention**.

```bash
accelerate launch --gpu_ids all src/train.py \
  --train_dir <train_dir> \
  --val_dir <val_dir>
```

### Notes for distributed training

- `--train_dir` must contain many `shard-XXXXX/` directories.
- Streaming dataloader partitions shards across `(rank, dataloader_worker)` slots; require `num_shards >= world_size * dataloader_num_workers` (guarded).
- Training data is treated as an infinite stream:
  - shard ownership fixed per `(rank, worker)` (no redundant reads)
  - each slot reshuffles its shard order each cycle
  - batches reconstructed via contiguous `batch_id`
- Training stops by `training.max_steps` / `TrainingArguments.max_steps`.

### LR schedule & stopping (design choices)
- Warmup: linear 0 → 1 over `warmup_steps`.
- Decay: cosine from 1.0 → 0.1 over `schedule_steps`.
- Plateau: LR held at 0.1× after `schedule_steps` until `max_steps` (hard cap).
- Early stopping: enabled by default; patience 5 evals on `eval_loss` (lower is better); `load_best_model_at_end` uses the same metric.

---
## Inspiration and Upstream Work

This codebase is inspired by:
- Cheng et al. (2024): training compute-optimal protein language models (LR/warmup/cosine design).
- Zhangzhi et al. (FAESM) and FlashAttention implementations for efficient ESM training.

### BibTeX for this repository
```

```