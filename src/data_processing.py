import os
import random
import argparse
from collections import defaultdict  # may still be useful elsewhere
import math
from typing import Iterator, List, Tuple, Dict, Optional

from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

from config import get_model_config, get_data_config


def pad_batch(batch, max_length, pad_token_id):
    """
    Pads all fields in a batch (input_ids, attention_mask, labels) to the same length.
    """
    for example in batch:
        padding_length = max_length - len(example["input_ids"])
        # Pad input_ids
        example["input_ids"] += [pad_token_id] * padding_length
        # Pad attention_mask
        example["attention_mask"] += [0] * padding_length
        # Pad labels with -100 (ignored during loss computation)
        example["labels"] += [-100] * padding_length

        # Add checks to ensure padding consistency
        assert len(example["input_ids"]) == max_length, (
            f"Padding error in input_ids: {len(example['input_ids'])} != {max_length}"
        )
        assert len(example["attention_mask"]) == max_length, (
            f"Padding error in attention_mask: {len(example['attention_mask'])} != {max_length}"
        )
        assert len(example["labels"]) == max_length, (
            f"Padding error in labels: {len(example['labels'])} != {max_length}"
        )
    return batch


def iter_fasta_sequences(path: str) -> Iterator[Tuple[str, str]]:
    """
    Streaming FASTA reader.
    Yields (header_line, sequence_string) pairs without loading all sequences into RAM.
    """
    # We open in binary mode so we can track byte-accurate progress (useful for tqdm).
    # FASTA is ASCII; we decode with replacement to avoid hard crashes on stray bytes.
    with open(path, "rb") as f:
        current_id = None
        current_seq: List[str] = []
        for raw_line in f:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None and current_seq:
                    yield current_id, "".join(current_seq)
                current_id = line
                current_seq = []
            else:
                current_seq.append(line)
        # Last sequence
        if current_id is not None and current_seq:
            yield current_id, "".join(current_seq)


def iter_fasta_sequences_with_byte_progress(
    path: str,
) -> Iterator[Tuple[str, str]]:
    """
    Like iter_fasta_sequences(), but also shows an overall tqdm progress bar for
    bytes-read through an uncompressed FASTA file.
    """
    total_bytes = os.path.getsize(path)
    # Throttle tqdm refresh to roughly every 100MB
    with tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc="Reading FASTA (bytes)",
        miniters=100_000_000,
    ) as pbar:
        with open(path, "rb") as f:
            current_id = None
            current_seq: List[str] = []
            for raw_line in f:
                pbar.update(len(raw_line))
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_id is not None and current_seq:
                        yield current_id, "".join(current_seq)
                    current_id = line
                    current_seq = []
                else:
                    current_seq.append(line)
            # Last sequence
            if current_id is not None and current_seq:
                yield current_id, "".join(current_seq)


def split_sequence_balanced(seq: str, max_chunk_len: int) -> List[Tuple[int, int, str]]:
    """
    Split a sequence into chunks of length <= max_chunk_len, trying to keep chunk lengths
    as equal as possible (sizes differ by at most 1).

    Returns (start_1based, end_1based, chunk_seq) tuples.
    """
    if max_chunk_len <= 0:
        raise ValueError(f"max_chunk_len must be > 0, got {max_chunk_len}")

    L = len(seq)
    if L == 0:
        return []
    if L <= max_chunk_len:
        return [(1, L, seq)]

    n_chunks = int(math.ceil(L / max_chunk_len))
    base = L // n_chunks
    rem = L % n_chunks

    out: List[Tuple[int, int, str]] = []
    pos0 = 0
    for i in range(n_chunks):
        this_len = base + (1 if i < rem else 0)
        chunk = seq[pos0 : pos0 + this_len]
        start_1 = pos0 + 1
        end_1 = pos0 + this_len
        out.append((start_1, end_1, chunk))
        pos0 += this_len

    assert pos0 == L, "Chunking bug: did not consume full sequence"
    return out


def make_chunk_header(header_line: str, start_1based: int, end_1based: int, is_last: bool) -> str:
    """
    Append a range suffix to the first token of a FASTA header, preserving any description.

    Example:
      >seqA some desc + (1,100,last=False)  -> >seqA/1-100 some desc
      >seqA some desc + (101,250,last=True) -> >seqA/101-end some desc
    """
    if not header_line.startswith(">"):
        raise ValueError(f"Expected FASTA header starting with '>', got: {header_line!r}")

    body = header_line[1:]
    parts = body.split(None, 1)
    seq_id = parts[0] if parts else ""
    desc = parts[1] if len(parts) == 2 else ""

    end_str = "end" if is_last else str(end_1based)
    new_id = f"{seq_id}/{start_1based}-{end_str}"
    if desc:
        return f">{new_id} {desc}"
    return f">{new_id}"


def refine_fasta(
    input_fasta: str,
    output_fasta: str,
    max_aa_length: int,
    stats: Optional[Dict[str, int]] = None,
) -> Iterator[Tuple[str, str, bool, bool]]:
    """
    Streaming FASTA chunking + rewriting:
      - Reads input_fasta
      - Splits sequences longer than max_aa_length into balanced chunks (<= max_aa_length)
      - Writes the (possibly chunked) sequences to output_fasta
      - Yields (header, seq, add_bos, add_eos) for each output record

    Chunking semantics:
      - First chunk: add BOS/CLS only
      - Last chunk: add EOS only
      - Middle chunks: neither
      - Single-chunk sequence: both

    NOTE: Unlike the old implementation, this does NOT build a dict of all
    sequences in RAM. It streams them, which is crucial for huge FASTAs.
    """

    print("Reading FASTA file (streaming), chunking, and writing refined FASTA...")
    record_count = 0  # input FASTA records
    chunk_record_count = 0  # output records after chunking
    split_seq_count = 0

    with open(output_fasta, "w") as out_f:
        for header, seq in iter_fasta_sequences_with_byte_progress(input_fasta):
            record_count += 1
            chunks = split_sequence_balanced(seq, max_aa_length)
            if len(chunks) > 1:
                split_seq_count += 1

            for i, (start_1, end_1, chunk_seq) in enumerate(chunks):
                is_first = i == 0
                is_last = i == (len(chunks) - 1)

                # Determine BOS/EOS flags for tokenization.
                add_bos = is_first
                add_eos = is_last
                if len(chunks) == 1:
                    add_bos = True
                    add_eos = True
                    out_header = header
                else:
                    out_header = make_chunk_header(
                        header_line=header,
                        start_1based=start_1,
                        end_1based=end_1,
                        is_last=is_last,
                    )

                out_f.write(f"{out_header}\n")
                out_f.write(f"{chunk_seq}\n")
                chunk_record_count += 1
                yield out_header, chunk_seq, add_bos, add_eos

    if stats is not None:
        stats["input_records"] = record_count
        stats["output_records"] = chunk_record_count
        stats["split_sequences"] = split_seq_count

    print(f"Read {record_count} FASTA records.")
    print(f"Split {split_seq_count} sequences into chunks.")
    print(f"Refined FASTA contains {chunk_record_count} output records (after chunking).")
    print(f"Refined FASTA written to {output_fasta}.")


def preprocess_fasta(
    file_path: str,
    tokenizer,
    max_aa_length: int,
    max_tokens_per_batch: int,
    chunk_size: int = 1_000_000,
    shard_size: int = 25_000,
    output_dir: str = "data/processed",
    seed: int = 100,
):
    """
    Preprocess a FASTA file to create pre-batched, padded datasets with batch_id,
    based on a max tokens-per-batch budget. Works in a streaming fashion:

      1. Streams the FASTA, chunking sequences longer than max_aa_length, and writes a
         <name>_refined.fasta alongside the original.
      2. Buffers up to `chunk_size` sequences at a time (window), length-sorts
         that window, and forms batches under `max_tokens_per_batch`.
      3. Accumulates `shard_size` batches in memory and writes them as a single
         Hugging Face Dataset shard to `output_dir/shard-XXXXX`.

    This avoids:
      - Holding all sequences in memory
      - Loading/saving intermediate "chunk_*.json" datasets
      - Global merging and shuffling (we shuffle within each shard instead)
    """

    rng = random.Random(seed)

    raw_dir = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name_no_ext = os.path.splitext(base_name)[0]
    refined_fasta_path = os.path.join(raw_dir, f"{name_no_ext}_refined.fasta")

    os.makedirs(output_dir, exist_ok=True)

    cls_id = getattr(tokenizer, "cls_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if cls_id is None or eos_id is None:
        raise ValueError(
            "Tokenizer must define cls_token_id and eos_token_id for manual BOS/EOS handling."
        )

    def tokenize_chunk(sequence: str, add_bos: bool, add_eos: bool) -> Dict:
        # Tokenize AA-only. We add special tokens manually to match desired semantics.
        tokens = tokenizer(
            sequence,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        input_ids: List[int] = list(tokens["input_ids"])
        if add_bos:
            input_ids = [cls_id] + input_ids
        if add_eos:
            input_ids = input_ids + [eos_id]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.copy(),
            "length": len(input_ids),
        }

    shard_batches: List[List[Dict]] = []
    shard_index = 0
    batch_id = 0

    def flush_shard():
        """
        Shuffle batches within the shard and write to disk as a Dataset.
        """
        nonlocal shard_batches, shard_index
        if not shard_batches:
            return

        rng.shuffle(shard_batches)
        shard_examples = [ex for batch in shard_batches for ex in batch]
        shard_dir = os.path.join(output_dir, f"shard-{shard_index:05d}")
        ds = Dataset.from_list(shard_examples)
        ds.save_to_disk(shard_dir)
        print(
            f"Saved shard {shard_index} with {len(shard_batches)} batches "
            f"and {len(shard_examples)} examples to {shard_dir}"
        )
        shard_index += 1
        shard_batches = []

    def process_window(window_seqs: List[Tuple[str, Tuple[str, bool, bool]]]):
        """
        Given a list of (id, seq) pairs, length-sort them, form batches
        under the token budget, and add those batches into shard_batches.
        """
        nonlocal batch_id, shard_batches

        if not window_seqs:
            return

        # Length-sort within this window
        window_seqs.sort(key=lambda x: len(x[1][0]))

        current_batch: List[Dict] = []
        max_padded_length = 0

        for seq_id, payload in window_seqs:
            seq, add_bos, add_eos = payload
            tokens = tokenize_chunk(seq, add_bos=add_bos, add_eos=add_eos)
            sequence_length = len(tokens["input_ids"])

            # Check whether adding this sequence would exceed the token budget.
            # IMPORTANT: We must NOT update max_padded_length until we've decided
            # whether this sequence belongs in the current batch; otherwise we'd
            # pad/finalize the previous batch to a length it doesn't contain.
            prospective_max_len = max(max_padded_length, sequence_length)
            prospective_tokens = prospective_max_len * (len(current_batch) + 1)

            # If adding this sequence would exceed the token budget, finalize current batch
            # (pad to the current batch's max_padded_length), then start a new batch.
            if current_batch and prospective_tokens > max_tokens_per_batch:
                pad_batch(current_batch, max_padded_length, tokenizer.pad_token_id)
                for ex in current_batch:
                    ex["batch_id"] = batch_id
                shard_batches.append(current_batch)
                batch_id += 1

                # Flush shard if we've accumulated enough batches
                if len(shard_batches) >= shard_size:
                    flush_shard()

                # Start a new batch
                current_batch = []
                max_padded_length = 0

                # Recompute for a fresh batch (this sequence alone)
                prospective_max_len = sequence_length
                prospective_tokens = prospective_max_len * 1

            # If even a single sequence cannot fit, fail fast with a clear error.
            if prospective_tokens > max_tokens_per_batch:
                raise ValueError(
                    f"Single sequence tokenized length {sequence_length} exceeds "
                    f"max_tokens_per_batch={max_tokens_per_batch}. "
                    "Increase max_tokens_per_batch or reduce max_aa_length."
                )

            current_batch.append(
                {
                    "id": seq_id,
                    "input_ids": tokens["input_ids"],
                    "attention_mask": tokens["attention_mask"],
                    "labels": tokens["labels"],
                    "sequence_length": sequence_length,
                }
            )
            max_padded_length = max(max_padded_length, sequence_length)

        # Last batch in this window
        if current_batch:
            pad_batch(current_batch, max_padded_length, tokenizer.pad_token_id)
            for ex in current_batch:
                ex["batch_id"] = batch_id
            shard_batches.append(current_batch)
            batch_id += 1

            if len(shard_batches) >= shard_size:
                flush_shard()

    # ---- Main streaming loop ----
    print("Streaming FASTA, chunking, and building shards...")

    window: List[Tuple[str, Tuple[str, bool, bool]]] = []
    output_record_count = 0
    refine_stats: Dict[str, int] = {}

    # refine_fasta writes refined_fasta_path and yields (id, seq, add_bos, add_eos) tuples
    stream = refine_fasta(
        file_path, refined_fasta_path, max_aa_length=max_aa_length, stats=refine_stats
    )
    for seq_id, seq, add_bos, add_eos in stream:
        output_record_count += 1
        window.append((seq_id, (seq, add_bos, add_eos)))

        # When window is full, process it and reset
        if len(window) >= chunk_size:
            process_window(window)
            window = []

    # leftover window
    if window:
        process_window(window)
        window = []

    # any remaining batches in the last shard
    flush_shard()

    input_record_count = refine_stats.get("input_records", -1)
    print(
        f"Finished preprocessing {output_record_count} output records "
        f"from {input_record_count} input FASTA records."
    )
    print(f"Final shards are in {output_dir} as shard-XXXXX directories.")


def main():
    parser = argparse.ArgumentParser(description="Process FASTA files for ESM-2 training")

    # Required arguments
    parser.add_argument(
        "--input_fasta",
        type=str,
        required=True,
        help="Path to the raw FASTA file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the processed dataset shards (shard-XXXXX)",
    )

    # Optional arguments with defaults (keep CLI similar)
    data_config = get_data_config()
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=data_config["chunk_size"],
        help=(
            "Number of sequences per length-sorting window "
            "(previously 'examples per intermediate chunk')."
        ),
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=data_config["default_shard_size"],
        help="Number of pre-batched batches per dataset shard.",
    )
    parser.add_argument(
        "--max_aa_length",
        type=int,
        default=None,
        help=(
            "Maximum amino acids per sequence/chunk, excluding BOS/CLS and EOS. "
            "Long sequences are split into balanced chunks of length <= max_aa_length, "
            "and chunk IDs are suffixed with ranges (e.g., /1-1022, /1023-end). "
            "Default: model max_position_embeddings - 2."
        ),
    )

    args = parser.parse_args()

    # Load tokenizer and configuration
    config = get_model_config()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Safety check: ensure YAML max_position_embeddings matches the underlying checkpoint.
    hf_cfg = AutoConfig.from_pretrained(config.model_name)
    hf_mpe = getattr(hf_cfg, "max_position_embeddings", None)
    if hf_mpe is not None and int(hf_mpe) != int(config.max_position_embeddings):
        raise ValueError(
            f"config.yaml model.max_position_embeddings={config.max_position_embeddings} does not match "
            f"checkpoint max_position_embeddings={hf_mpe} for model_name={config.model_name!r}. "
            "Make them consistent to avoid preprocessing/model length mismatch."
        )

    # Infer max_length from max_position_embeddings
    max_aa_length = args.max_aa_length or (config.max_position_embeddings - 2)
    if max_aa_length + 2 > config.max_position_embeddings:
        raise ValueError(
            f"max_aa_length={max_aa_length} is too large for "
            f"max_position_embeddings={config.max_position_embeddings}. "
            "Need max_aa_length + 2 <= max_position_embeddings."
        )
    max_tokens_per_batch = config.max_batch_size

    os.makedirs(args.output_dir, exist_ok=True)

    preprocess_fasta(
        file_path=args.input_fasta,
        tokenizer=tokenizer,
        max_aa_length=max_aa_length,
        max_tokens_per_batch=max_tokens_per_batch,
        chunk_size=args.chunk_size,
        shard_size=args.shard_size,
        output_dir=args.output_dir,
        seed=100,
    )


if __name__ == "__main__":
    main()