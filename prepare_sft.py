"""SFT data pipeline: download (resumable, skip-if-done) + pack to bins.

Phase A - download (per-spec, resumable):
  - ultrachat_200k (en instructions)   target 800K tokens
  - alpaca-zh (zh instructions)        target 1.2M tokens
  - tool dialogues (local, already in data/sft_tool.jsonl)

  State lives in data/cache/:
    <name>.jsonl   collected dialogues appended incrementally
    <name>.done    written when the token target is reached
  Re-running skips sources whose .done exists (or re-uses partial cache).

Phase B - pack:
  - renders ChatML, loss-masks only assistant turns
  - writes data/sft_data.bin + sft_mask.bin (+ val bins + sft_meta.json)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

try:  # use Windows system CA store (fixes CERTIFICATE_VERIFY_FAILED)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

CHAT_SPECIALS = ["<|im_start|>", "<|im_end|>", "<|tool_call|>", "<|tool_result|>", "<pad>"]
CHATML_START = "<|im_start|>"
CHATML_END = "<|im_end|>"

SOURCES = {
    "ultrachat": {"repo": "HuggingFaceH4/ultrachat_200k", "split": "train_sft",
                  "tokens": 5_000_000},
    "alpaca_zh": {"repo": "shibing624/alpaca-zh", "split": "train",
                  "tokens": 3_500_000},
}


def render_messages(messages, encode):
    ids, mask = [], []
    for m in messages:
        role, content = m["role"], m["content"]
        if role in ("tool", "tool_result"):
            # wrap tool results with the reserved <|tool_result|> token so the
            # tokenizer treats it as a single unit (aligned with chat_template)
            content = f"<|tool_result|>{content}<|tool_result|>"
        seg = f"{CHATML_START}{role}\n{content}{CHATML_END}\n"
        seg_ids = encode(seg)
        seg_mask = [1 if role == "assistant" else 0] * len(seg_ids)
        ids += seg_ids
        mask += seg_mask
    return ids, mask


def _load_cache(cache_dir, name):
    """Return (lines, token_count, hashes)."""
    path = cache_dir / f"{name}.jsonl"
    lines = []
    n = 0
    hashes = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                lines.append(rec)
                hashes.add(rec["key"])
                n += rec["n_tokens"]
    return lines, n, hashes


def download_ultrachat(sp, cache_dir, target_tokens):
    name = "ultrachat"
    done = cache_dir / f"{name}.done"
    if done.exists():
        print(f"[skip] {name}: already done ({json.loads(done.read_text())['tokens']} tokens)")
        return
    lines, n_tokens, hashes = _load_cache(cache_dir, name)
    out_path = cache_dir / f"{name}.jsonl"
    fout = open(out_path, "a", encoding="utf-8")
    reconnects = 0
    try:
        while n_tokens < target_tokens and reconnects < 20:
            try:
                from datasets import load_dataset
                ds = load_dataset(SOURCES[name]["repo"], split=SOURCES[name]["split"], streaming=True)
                for row in ds:
                    msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
                    ids, mask = render_messages(msgs, sp.encode)
                    key = str(ids[:64])
                    if key in hashes:
                        continue
                    hashes.add(key)
                    rec = {"key": key, "n_tokens": len(ids), "messages": msgs}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    lines.append(rec)
                    n_tokens += len(ids)
                    if n_tokens % 100_000 < 200:
                        print(f"  [{name}] {n_tokens/1000:.0f}K tokens", flush=True)
                    if n_tokens >= target_tokens:
                        break
                reconnects = 99  # clean exit
            except Exception as e:
                reconnects += 1
                print(f"  [{name}] stream error ({str(e)[:80]}); reconnect {reconnects}/20", flush=True)
    finally:
        fout.close()
    if n_tokens >= target_tokens:
        done.write_text(json.dumps({"tokens": n_tokens, "dialogues": len(lines)}), encoding="utf-8")
        print(f"[done] {name}: {n_tokens} tokens, {len(lines)} dialogues")
    else:
        print(f"[warn] {name}: only {n_tokens} tokens collected (reconnect budget exhausted)")


def download_alpaca(sp, cache_dir, target_tokens):
    name = "alpaca_zh"
    done = cache_dir / f"{name}.done"
    if done.exists():
        print(f"[skip] {name}: already done ({json.loads(done.read_text())['tokens']} tokens)")
        return
    lines, n_tokens, hashes = _load_cache(cache_dir, name)
    out_path = cache_dir / f"{name}.jsonl"
    fout = open(out_path, "a", encoding="utf-8")
    reconnects = 0
    try:
        while n_tokens < target_tokens and reconnects < 20:
            try:
                from datasets import load_dataset
                ds = load_dataset(SOURCES[name]["repo"], split=SOURCES[name]["split"], streaming=True)
                for row in ds:
                    instruction = (row.get("instruction") or "").strip()
                    inp = (row.get("input") or "").strip()
                    output = (row.get("output") or "").strip()
                    if not instruction or not output:
                        continue
                    user_q = instruction if not inp else f"{instruction}\n{inp}"
                    msgs = [{"role": "user", "content": user_q},
                            {"role": "assistant", "content": output}]
                    ids, mask = render_messages(msgs, sp.encode)
                    key = str(ids[:64])
                    if key in hashes:
                        continue
                    hashes.add(key)
                    rec = {"key": key, "n_tokens": len(ids), "messages": msgs}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    lines.append(rec)
                    n_tokens += len(ids)
                    if n_tokens % 100_000 < 200:
                        print(f"  [{name}] {n_tokens/1000:.0f}K tokens", flush=True)
                    if n_tokens >= target_tokens:
                        break
                reconnects = 99
            except Exception as e:
                reconnects += 1
                print(f"  [{name}] stream error ({str(e)[:80]}); reconnect {reconnects}/20", flush=True)
    finally:
        fout.close()
    if n_tokens >= target_tokens:
        done.write_text(json.dumps({"tokens": n_tokens, "dialogues": len(lines)}), encoding="utf-8")
        print(f"[done] {name}: {n_tokens} tokens, {len(lines)} dialogues")
    else:
        print(f"[warn] {name}: only {n_tokens} tokens collected (reconnect budget exhausted)")


def load_tool_dialogues():
    out = []
    path = Path("data/sft_tool.jsonl")
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out.append(rec["messages"])
    return out


def pack(sp, cache_dir, val_dialogues, seed):
    all_d = [(m, 1) for m in load_tool_dialogues()]
    for name in SOURCES:
        path = cache_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                all_d.append((json.loads(line)["messages"], 0))
    print(f"packing {len(all_d)} dialogues")
    rng = random.Random(seed)
    rng.shuffle(all_d)
    val = all_d[:val_dialogues]
    train = all_d[val_dialogues:]

    def enc(msgs):
        return render_messages(msgs, sp.encode)

    def pack_to(dialogues, ids_path, mask_path):
        ids_all, mask_all = [], []
        for msgs, _ in dialogues:
            ids, mask = enc(msgs)
            ids_all += ids
            mask_all += mask
        np.array(ids_all, dtype=np.uint16).tofile(ids_path)
        np.array(mask_all, dtype=np.uint8).tofile(mask_path)
        return len(ids_all)

    out = Path("data")
    n_train = pack_to(train, out / "sft_data.bin", out / "sft_mask.bin")
    n_val = pack_to(val, out / "sft_val_data.bin", out / "sft_val_mask.bin")
    meta = {"train_tokens": n_train, "val_tokens": n_val,
            "n_train_dialogues": len(train), "n_val_dialogues": len(val),
            "mask_frac": sum(m for _, m in all_d) / len(all_d)}
    (out / "sft_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("sft meta:", meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp-model", default="tokenizer/spm.model")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--download", action="store_true", help="download missing SFT sources")
    ap.add_argument("--pack", action="store_true", help="pack cache into bins (requires downloads done)")
    ap.add_argument("--val-dialogues", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.sp_model)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_ultrachat(sp, cache_dir, SOURCES["ultrachat"]["tokens"])
        download_alpaca(sp, cache_dir, SOURCES["alpaca_zh"]["tokens"])

    if args.pack:
        pack(sp, cache_dir, args.val_dialogues, args.seed)


if __name__ == "__main__":
    main()