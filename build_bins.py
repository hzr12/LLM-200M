"""Encode corpus jsonl into uint16 memmap bins (nanoGPT style), streamed.

Incremental batch reading:
  - files are processed one at a time (alphabetical + seeded shuffle of the
    file list for cross-language mixing)
  - each file is read in fixed-size CHUNK batches of lines; every chunk is
    shuffled in-memory, tokenized and appended to the memmap, then released
  - peak memory = one CHUNK of docs + a few MB of state, INDEPENDENT of total
    corpus size (14GB of jsonl needs only ~100MB RAM)

Splitting:
  - the first `--val-tokens` tokens of the stream go to val.bin
  - the rest (up to `--train-tokens`) go to train.bin

Writes data/meta.json (vocab size, counts, special ids).

Usage:
    python build_bins.py --sp-model tokenizer/spm.model \
        --train-tokens 3000000000 --val-tokens 2000000
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import sentencepiece as spm

SPECIALS = ["<|im_start|>", "<|im_end|>", "<|tool_call|>", "<|tool_result|>", "<pad>"]
CHUNK = 5000          # docs per in-memory batch
PROGRESS_EVERY = 20_000_000  # tokens between progress lines


def iter_chunks(files, seed):
    """Yield (file_name, list_of_texts) chunks; file order shuffled, chunks of CHUNK."""
    rng = random.Random(seed)
    order = list(files)
    rng.shuffle(order)
    for f in order:
        chunk = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = rec.get("text")
                if text:
                    chunk.append(text)
                if len(chunk) >= CHUNK:
                    rng.shuffle(chunk)
                    yield f.name, chunk
                    chunk = []
        if chunk:
            rng.shuffle(chunk)
            yield f.name, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="data/corpus")
    ap.add_argument("--sp-model", default="tokenizer/spm.model")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--train-tokens", type=int, default=3_000_000_000)
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.sp_model)
    specials = {t: sp.piece_to_id(t) for t in SPECIALS if sp.piece_to_id(t) != sp.unk_id()}
    print("special ids:", specials)

    files = sorted(Path(args.corpus_dir).glob("*.jsonl"))
    print(f"corpus files ({len(files)}): {[f.name for f in files]}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_mmap = np.memmap(out_dir / "val.bin", dtype=np.uint16, mode="w+",
                         shape=(args.val_tokens,))
    train_mmap = np.memmap(out_dir / "train.bin", dtype=np.uint16, mode="w+",
                           shape=(args.train_tokens,))
    val_pos = train_pos = 0
    t_total = 0
    next_progress = PROGRESS_EVERY

    for fname, docs in iter_chunks(files, args.seed):
        for text in docs:
            ids = sp.encode(text, out_type=int)
            n = len(ids)
            t_total += n
            if val_pos < args.val_tokens:
                take = min(n, args.val_tokens - val_pos)
                val_mmap[val_pos:val_pos + take] = ids[:take]
                val_pos += take
            if train_pos < args.train_tokens:
                take = min(n, args.train_tokens - train_pos)
                train_mmap[train_pos:train_pos + take] = ids[:take]
                train_pos += take
            if train_pos >= args.train_tokens and val_pos >= args.val_tokens:
                break
        if train_pos >= args.train_tokens and val_pos >= args.val_tokens:
            print(f"target reached: val {val_pos:,} train {train_pos:,}", flush=True)
            break
        if t_total >= next_progress:
            print(f"  tokenized {t_total/1e6:.0f}M tokens (from {fname}) | "
                  f"val {val_pos/1e6:.1f}M train {train_pos/1e6:.1f}M", flush=True)
            next_progress += PROGRESS_EVERY

    val_mmap.flush()
    train_mmap.flush()
    print(f"val.bin: {val_pos:,} tokens | train.bin: {train_pos:,} tokens", flush=True)

    meta = {
        "vocab_size": sp.vocab_size(),
        "train_tokens": int(train_pos),
        "val_tokens": int(val_pos),
        "special_ids": specials,
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("meta.json written")


if __name__ == "__main__":
    main()
