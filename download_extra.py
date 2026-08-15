"""Download ADDITIONAL pretrain corpus to reach 70M total tokens (49M existing + 21M new).

Writes NEW files with a `_2` suffix so the existing data/corpus/*.jsonl are
NOT overwritten — build_bins.py globs every *.jsonl in the dir and merges them,
so the final train.bin is the union of old + new.

Char budgets (~4.2 en chars/token, ~1.7 zh chars/token):
  - en: +~15M tokens  -> 63M chars
  - zh: +~6M tokens   -> 10M chars
This overshoots slightly; build_bins caps at exactly 70M train tokens.
"""
import argparse
import json
import sys
import time
from pathlib import Path

try:  # Windows system CA store
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

ZH_DATASET = "0xDing/wikipedia-cn-20230720-filtered"
EN_DATASET = "HuggingFaceFW/fineweb-edu"
EN_CONFIG = "sample-100BT"


def pick_text(row):
    for k in ("text", "completion", "content"):
        if k in row and isinstance(row[k], str) and len(row[k]) > 0:
            return row[k]
    if "conversations" in row and isinstance(row["conversations"], list):
        parts = [m.get("value", "") for m in row["conversations"] if isinstance(m, dict)]
        if parts:
            return "\n".join(parts)
    return None


def stream(name, config, budget, lang, out_path, retries=3):
    from datasets import load_dataset
    print(f"[{lang}] loading {name} (config={config}) ...", flush=True)
    ds = None
    for attempt in range(retries):
        try:
            ds = load_dataset(name, config, split="train", streaming=True)
            break
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == retries - 2 and config is not None:
                config = None
            time.sleep(2)
    if ds is None:
        raise RuntimeError(f"cannot load {name}")
    f = open(out_path, "w", encoding="utf-8")
    chars = 0
    n = 0
    skipped = 0
    t0 = time.time()
    try:
        for row in ds:
            text = pick_text(row)
            if not text or len(text.strip()) < 100:
                skipped += 1
                continue
            text = text.strip()
            f.write(json.dumps({"text": text, "lang": lang}, ensure_ascii=False) + "\n")
            chars += len(text)
            n += 1
            if n % 500 == 0:
                print(f"  [{lang}] docs={n} chars={chars/1e6:.1f}M elapsed={time.time()-t0:.0f}s",
                      flush=True)
            if chars >= budget:
                break
    finally:
        f.close()
    print(f"[{lang}] done: {n} docs, {chars/1e6:.1f}M chars, skipped {skipped}, "
          f"{time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-chars", type=int, default=63_000_000)
    ap.add_argument("--zh-chars", type=int, default=10_000_000)
    ap.add_argument("--out-dir", default="data/corpus")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stream(ZH_DATASET, None, args.zh_chars, "zh", out / "train_zh2.jsonl")
    stream(EN_DATASET, EN_CONFIG, args.en_chars, "en", out / "train_en2.jsonl")
    print("extra corpus written; build_bins.py will merge all *.jsonl -> 70M train.bin")


if __name__ == "__main__":
    main()
