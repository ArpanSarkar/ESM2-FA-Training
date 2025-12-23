# train.py

import glob
import os
import argparse
import torch.distributed as dist
from datasets import load_from_disk
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from config import get_training_config
from model import initialize_model_and_tokenizer
from trainer import ProteinLMTrainer


def main():
    parser = argparse.ArgumentParser(description="Train ESM2-style FAESM model on preprocessed datasets.")
    parser.add_argument("--train_dir", type=str, help="Path to the processed training dataset shards (shard-XXXXX).")
    parser.add_argument("--val_dir", type=str, help="Path to the processed validation dataset (HF dataset dir).")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Run evaluation only (no training); requires --val_dir.",
    )
    args = parser.parse_args()

    # Override global config if a custom path is provided.
    from config import set_config
    set_config(args.config)

    # Check if distributed training is initialized
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    # 1) Load training configuration
    training_config = get_training_config()
    eval_only = args.eval_only or training_config.get("eval_only", False)
    training_config["eval_only"] = eval_only
    max_steps = training_config.get("max_steps", 0) or 0

    # 2) Gather shard directories for the training set
    train_dir = args.train_dir
    shard_paths = None
    if not eval_only:
        if not train_dir:
            raise ValueError("--train_dir is required unless running with --eval_only.")
        shard_paths = sorted(glob.glob(os.path.join(train_dir, "shard-*")))
        if len(shard_paths) == 0:
            raise ValueError(f"No shards found in {train_dir}!")
        if rank == 0:
            print(f"Found {len(shard_paths)} shards for training in {train_dir}.")

    # 3) Load the validation dataset (single HF dataset folder)
    def resolve_val_dir(val_root: str) -> str:
        """
        Allow pointing --val_dir either at a single HF dataset folder (shard-XXXXX)
        or at a parent containing shard-* subfolders. Enforce exactly one shard if
        a parent is given.
        """
        if not os.path.exists(val_root):
            raise ValueError(f"Validation path does not exist: {val_root}")

        dataset_info = os.path.join(val_root, "dataset_info.json")
        if os.path.isfile(dataset_info):
            return val_root

        shard_dirs = sorted(glob.glob(os.path.join(val_root, "shard-*")))
        if len(shard_dirs) == 1:
            return shard_dirs[0]
        if len(shard_dirs) == 0:
            raise ValueError(
                f"{val_root} is not a HF dataset dir and contains no shard-XXXXX subfolders."
            )
        raise ValueError(
            f"{val_root} contains multiple shard-XXXXX subfolders. "
            "Point --val_dir to a single shard directory."
        )

    val_dataset_dir = resolve_val_dir(args.val_dir)
    if rank == 0:
        print(f"Loading validation dataset from {val_dataset_dir}")
    val_dataset = load_from_disk(val_dataset_dir)

    # 4) Initialize model and tokenizer
    if rank == 0:
        print("Initializing model and tokenizer...")
    model, tokenizer = initialize_model_and_tokenizer(eval_only=eval_only)

    # Print model size only on rank 0
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {num_params} trainable parameters.")

    # 5) Resolve last checkpoint (for resume)
    output_dir = training_config["output_dir"]
    last_checkpoint = None
    if (not eval_only) and os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint and rank == 0:
            print(f"Detected checkpoint, will resume from: {last_checkpoint}")

    # 6) Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=training_config["mlm_probability"],
    )

    # 7) TrainingArguments
    # Handle HF version differences: some releases use evaluation_strategy, others eval_strategy.
    eval_strategy = training_config.get("evaluation_strategy", training_config.get("eval_strategy", "steps"))
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        # HF version differences: use eval_strategy (older versions) instead of evaluation_strategy
        eval_strategy=eval_strategy,
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config.get("save_total_limit", None),
        learning_rate=training_config["learning_rate"],
        max_grad_norm=training_config.get("gradient_clipping", 0.0),
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        max_steps=max_steps,
        warmup_steps=training_config["warmup_steps"],
        logging_dir="./logs",
        logging_steps=training_config["logging_steps"],
        fp16=training_config["mixed_precision"] == "fp16",
        bf16=training_config["mixed_precision"] == "bf16",
        report_to=["wandb"],
        lr_scheduler_type=training_config["lr_scheduler"],
        dataloader_num_workers=training_config["dataloader_num_workers"],
        dataloader_pin_memory=training_config["dataloader_pin_memory"],
        dataloader_prefetch_factor=training_config["dataloader_prefetch_factor"],
        seed=training_config["seed"],
        load_best_model_at_end=training_config.get("load_best_model_at_end", False),
        metric_for_best_model=training_config.get("metric_for_best_model", None),
        greater_is_better=training_config.get("greater_is_better", None),
    )

    # 8) Early stopping callback (optional)
    callbacks = []
    early_stop_cfg = training_config.get("early_stopping", {})
    if early_stop_cfg.get("enabled", False):
        patience = early_stop_cfg.get("patience", 5)
        threshold = early_stop_cfg.get("threshold", 0.0)
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=threshold,
            )
        )

    # 9) Initialize Trainer
    trainer = ProteinLMTrainer(
        train_dataset=shard_paths,   # list of shard-XXXXX directories (None for eval_only)
        eval_dataset=val_dataset,    # HF Dataset
        data_collator=data_collator,
        training_config=training_config,
        model=model,
        args=training_args,
        callbacks=callbacks,
    )

    if eval_only:
        if rank == 0:
            print("Running evaluation only...")
        metrics = trainer.evaluate()
        if rank == 0:
            print(f"Eval metrics: {metrics}")
        return

    if rank == 0:
        print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)


if __name__ == "__main__":
    main()

