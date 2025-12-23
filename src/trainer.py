import math
import random
import inspect
from collections import defaultdict
from typing import List, Dict, Iterable

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from datasets import load_from_disk
from transformers import Trainer, set_seed


class ShardBatchDataset(IterableDataset):
    """
    IterableDataset that:
      - shuffles the full shard list once per pass
      - partitions shards across (rank, worker) slots
      - streams pre-batched examples from its local shard subset only

    This avoids redundant I/O: each shard is read by exactly one worker
    across all ranks, instead of every rank reading every shard.
    """

    def __init__(
        self,
        shard_paths: List[str],
        seed: int,
        rank: int,
        world_size: int,
        args,
    ):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.seed = seed
        # TrainingArguments.local_rank is -1 in non-distributed runs.
        # Normalize to avoid negative slot indices.
        self.rank = 0 if rank is None or rank < 0 else rank
        self.world_size = 1 if world_size is None or world_size < 1 else world_size
        self.args = args

    @staticmethod
    def _stream_batches_from_shard(shard_path: str) -> Iterable[List[Dict]]:
        """
        Stream pre-batched examples from a shard by grouping contiguous examples
        that share the same batch_id.

        Assumes preprocessing wrote examples in batch-major order:
          [batch 0 ex0, batch 0 ex1, ..., batch 1 ex0, batch 1 ex1, ...]
        """
        ds = load_from_disk(shard_path)

        current_batch_id = None
        current_batch: List[Dict] = []

        for ex in ds:
            bid = ex["batch_id"]
            if current_batch_id is None:
                current_batch_id = bid

            if bid != current_batch_id:
                # finish previous batch
                if current_batch:
                    yield current_batch
                current_batch = [ex]
                current_batch_id = bid
            else:
                current_batch.append(ex)

        if current_batch:
            yield current_batch

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Total "slots" = (rank, worker) pairs. With slot partitioning, we require
        # enough shards so that every slot gets at least one shard; otherwise some
        # ranks may see no data and distributed training can hang.
        num_slots = max(1, self.world_size * num_workers)
        slot_index = self.rank * num_workers + worker_id
        if slot_index < 0 or slot_index >= num_slots:
            # Shouldn't happen after rank/world_size normalization, but be safe.
            return

        # Shuffle the *global* shard list once per pass (per __iter__) in a way that is
        # deterministic across ranks/workers. We then take this slot's strided subset.
        #
        # IMPORTANT: We do NOT reshuffle the global list inside the infinite loop below.
        # Worker/rank throughput can differ slightly, and if each slot reshuffled the
        # *global* list independently per cycle, shard ownership would drift and shards
        # could be read redundantly. Instead, we keep shard ownership fixed per pass,
        # and reshuffle only the *order* of this slot's local shards each cycle.
        shards = self.shard_paths[:]
        rng0 = random.Random(self.seed)
        rng0.shuffle(shards)

        num_shards = len(shards)
        if num_shards < num_slots:
            raise ValueError(
                f"Need num_shards >= world_size * num_workers for slot partitioning, "
                f"but got num_shards={num_shards}, world_size={self.world_size}, "
                f"num_workers={num_workers} (num_slots={num_slots}). "
                "Increase shard count, reduce dataloader_num_workers, or reduce world_size."
            )

        local_shards_base = shards[slot_index:num_shards:num_slots]
        if not local_shards_base:
            raise ValueError(
                f"No shards assigned to rank={self.rank} worker_id={worker_id} "
                f"(slot_index={slot_index}, num_slots={num_slots}, num_shards={num_shards})."
            )

        if self.rank == 0 and worker_id == 0:
            print(
                f"[Rank {self.rank}] Starting shard stream: "
                f"{num_shards} total shards, {num_slots} slots, "
                f"{len(local_shards_base)} shards for this slot."
            )

        # Robust reshuffling pattern for max-steps pretraining:
        # iterate forever and reshuffle this slot's local shard ORDER each cycle.
        cycle_idx = 0
        while True:
            cycle_idx += 1
            rng = random.Random(self.seed + int(cycle_idx))
            local_shards = local_shards_base[:]
            rng.shuffle(local_shards)

            # Stream batches from local shards only
            for shard_idx, shard_path in enumerate(local_shards):
                if self.rank == 0 and worker_id == 0:
                    print(
                        f"[Rank {self.rank}] Cycle {cycle_idx}: "
                        f"slot 0 reading shard {shard_idx + 1}/{len(local_shards)}: {shard_path}"
                    )
                for batch in self._stream_batches_from_shard(shard_path):
                    yield batch


class ProteinLMTrainer(Trainer):
    """
    Trainer for large-scale protein MLM pretraining with:

      - ShardBatchDataset streaming over pre-batched shards
      - AdamW optimizer following Cheng et al. (2024, bioRxiv: Training Compute-Optimal Protein Language Models)
      - Warmup + cosine LR schedule following Cheng et al. (2024, bioRxiv: Training Compute-Optimal Protein Language Models)
      - MLM pseudo-perplexity evaluation
    """

    def __init__(
        self,
        train_dataset,
        eval_dataset,
        data_collator,
        training_config: dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            *args,
            **kwargs,
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.training_config = training_config or {}
        self.true_global_step = 0

    # ---------- grouping / collate ----------

    def group_by_batch(self, dataset):
        """
        Groups a list of examples into lumps keyed by 'batch_id'.
        Each 'batch_id' becomes one list (lump).

        Only used for evaluation, where eval_dataset is small.
        """
        grouped_data = defaultdict(list)
        for example in dataset:
            grouped_data[example["batch_id"]].append(example)
        return list(grouped_data.values())

    @staticmethod
    def create_collate_fn(base_collator, keys_to_remove=None):
        """
        Returns a collate function that removes unwanted keys from each example,
        then calls 'base_collator' on the cleaned examples.

        For training:
          - batch_list is [lump], where lump = list[examples] for one batch_id
        """
        if keys_to_remove is None:
            keys_to_remove = []

        def custom_collate_fn(batch_list):
            # batch_size=1 -> this is our pre-batched lump
            lumps = batch_list[0]
            for ex in lumps:
                for key in keys_to_remove:
                    ex.pop(key, None)
            return base_collator(lumps)

        return custom_collate_fn

    # ---------- optimizer & scheduler ----------

    def create_optimizer(self):
        """
        AdamW defaults following Cheng et al. (2024, bioRxiv: Training Compute-Optimal Protein Language Models):

          beta1 = 0.9
          beta2 = 0.95
          eps   = 1e-8
          wd    = 0.01

        LR comes from training_config["learning_rate"] if set,
        otherwise from self.args.learning_rate.
        """
        if self.optimizer is not None:
            return self.optimizer

        opt_cfg = self.training_config.get("optimizer", {})

        lr_raw = self.training_config.get("learning_rate", None)
        if lr_raw is None:
            lr_raw = self.args.learning_rate
        lr = float(lr_raw)

        beta1 = float(opt_cfg.get("beta_1", 0.9))
        beta2 = float(opt_cfg.get("beta_2", 0.95))
        eps = float(opt_cfg.get("epsilon", 1e-8))
        weight_decay = float(opt_cfg.get("weight_decay", 0.01))
        fused_flag = opt_cfg.get("fused", None)
        supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters

        # Respect explicit config; otherwise prefer fused when supported.
        if fused_flag is None and supports_fused:
            fused_flag = True
        elif fused_flag and not supports_fused:
            raise ValueError(
                "Requested fused AdamW but this PyTorch build does not support it. "
                "Upgrade to a CUDA build with fused AdamW."
            )

        adamw_kwargs = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": eps,
            "weight_decay": weight_decay,
        }
        if supports_fused and fused_flag is not None:
            adamw_kwargs["fused"] = bool(fused_flag)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **adamw_kwargs,
        )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Warmup -> cosine decay to 0.1 by `schedule_steps` -> flat at 0.1 until `max_steps`.

        - `max_steps` is the hard cap (HF TrainingArguments + config)
        - `schedule_steps` is the decay horizon (<= max_steps)
        - `warmup_steps` is taken as-is from config (no auto-derivation)
        """
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        optimizer = optimizer if optimizer is not None else self.optimizer
        cfg = self.training_config

        max_steps_hard = cfg.get("max_steps", self.args.max_steps)
        schedule_steps = cfg.get("schedule_steps", max_steps_hard)
        warmup_steps = cfg.get("warmup_steps", None)
        if warmup_steps is None:
            # Should be set in config; fall back to args to avoid crash.
            warmup_steps = self.args.warmup_steps

        def lr_lambda(step: int):
            # Phase 0: warmup
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            # Phase 1: cosine decay from 1.0 -> 0.1 between warmup_steps and schedule_steps
            if step <= schedule_steps:
                progress = float(step - warmup_steps) / float(
                    max(1, schedule_steps - warmup_steps)
                )
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return 0.1 + 0.9 * cosine

            # Phase 2: flat at 0.1 beyond schedule_steps (until max_steps cap)
            return 0.1

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return self.lr_scheduler

    # ---------- training ----------

    # IMPORTANT:
    #  - Do NOT rescale loss here; Trainer already handles gradient_accumulation_steps.
    #  - Do NOT override optimizer_step; let Trainer/Accelerate handle fp16/bf16,
    #    gradient scaling, clipping (via TrainingArguments.max_grad_norm), etc.

    def get_train_dataloader(self):
        """
        DataLoader over ShardBatchDataset:

          - streams batches from shard-XXXXX directories
          - no list(shard), no giant in-RAM structures
          - partitions shards across ranks and workers
        """
        # If train_dataset is a real Dataset, let HF handle it
        if not isinstance(self.train_dataset, list):
            return super().get_train_dataloader()

        shard_iterable = ShardBatchDataset(
            shard_paths=self.train_dataset,
            seed=self.args.seed,
            rank=self.args.local_rank,
            world_size=self.args.world_size,
            args=self.args,
        )

        collate_fn = self.create_collate_fn(
            base_collator=self.data_collator,
            keys_to_remove=["batch_id", "id", "sequence_length"],
        )

        return DataLoader(
            shard_iterable,
            batch_size=1,  # each item is already a full batch (lump)
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
            persistent_workers=self.args.dataloader_num_workers > 0,
        )

    # ---------- evaluation ----------

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Evaluation data loader.

        For now we assume eval_dataset is a single HF Dataset (not sharded),
        so it's safe to materialize it in memory and group-by-batch.
        """
        eval_dataset = eval_dataset or self.eval_dataset
        all_ex = list(eval_dataset)
        grouped = self.group_by_batch(all_ex)

        collate_fn = self.create_collate_fn(
            base_collator=self.data_collator,
            keys_to_remove=["batch_id", "id", "sequence_length"],
        )

        return DataLoader(
            grouped,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss and log one-time masking stats for a sanity check.
        """
        # Get base loss
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        # One-time sanity print: masking stats and raw loss
        if not hasattr(self, "_mask_debug_logged"):
            try:
                labels = inputs.get("labels")
                attn = inputs.get("attention_mask")
                if labels is not None and attn is not None:
                    masked = (labels != -100)
                    total_tokens = attn.sum().item()
                    masked_tokens = (masked & attn.bool()).sum().item()
                    masked_frac = masked_tokens / max(1, total_tokens)
                    max_id = inputs["input_ids"].max().item()
                    if self.args.local_rank in [-1, 0]:
                        # loss may be a tuple if return_outputs is True
                        loss_val = loss[0] if isinstance(loss, tuple) else loss
                        loss_scalar = loss_val.detach().float().mean().item()
                        print(
                            f"[Mask debug] masked_frac={masked_frac:.4f}, "
                            f"total_tokens={total_tokens}, masked_tokens={masked_tokens}, "
                            f"max_input_id={max_id}, "
                            f"raw_loss={loss_scalar:.4f}, "
                            f"raw_loss/accum={loss_scalar / max(1, self.args.gradient_accumulation_steps):.4f}"
                        )
                self._mask_debug_logged = True
            except Exception:
                self._mask_debug_logged = True

        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Compute MLM-style pseudo-perplexity:

          - We use the model's masked-LM loss (cross-entropy over masked tokens only).
          - We aggregate loss *per masked token* across the eval set:
                avg_loss = sum(loss_i * masked_tokens_i) / sum(masked_tokens_i)
          - Report perplexity = exp(avg_loss).

        This matches the standard MLM evaluation used by Cheng et al. (2024, bioRxiv:
        Training Compute-Optimal Protein Language Models):
          - NOT expensive "mask one token at a time" pppl,
          - but standard masked-LM per-token cross-entropy on fresh random masks.
        """
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            return {}

        # Make eval masking repeatable without perturbing training RNG:
        # fork torch RNG (CPU + all CUDA devices), seed inside the context,
        # and auto-restore on exit.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else ()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.args.seed)

            eval_dataloader = self.get_eval_dataloader(eval_dataset)

            model = self.model
            device = self.args.device
            model.eval()
            model.to(device)

            total_loss = 0.0
            total_masked_tokens = 0

            with torch.no_grad():
                for batch in eval_dataloader:
                    for k, v in batch.items():
                        batch[k] = v.to(device)

                    outputs = model(**batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

                    masked_tokens = (batch["labels"] != -100).sum().item()
                    total_loss += loss.item() * masked_tokens
                    total_masked_tokens += masked_tokens

            metrics = {}
            if total_masked_tokens > 0:
                avg_loss = total_loss / total_masked_tokens
                ppl = math.exp(avg_loss)
                metrics[f"{metric_key_prefix}_perplexity"] = ppl
                metrics[f"{metric_key_prefix}_loss"] = avg_loss
            else:
                metrics[f"{metric_key_prefix}_perplexity"] = float("nan")
                metrics[f"{metric_key_prefix}_loss"] = float("nan")

            self.log(metrics)
            return metrics

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Override logging to report loss averaged over gradient accumulation steps.
        Accept extra args/kwargs to match Trainer.log signature.
        """
        if self.args.gradient_accumulation_steps > 1:
            try:
                logs = dict(logs)
                if "loss" in logs:
                    logs["loss"] = logs["loss"] / float(self.args.gradient_accumulation_steps)
                if "grad_norm" in logs:
                    logs["grad_norm"] = logs["grad_norm"] / float(self.args.gradient_accumulation_steps)
            except Exception:
                pass
        return super().log(logs, *args, **kwargs)