"""Download pre-cleaned corpora and sample to char-based budgets.

Sources (already cleaned, no extra filtering needed):
  - zh: 0xDing/wikipedia-cn-20230720-filtered   (quality-filtered Chinese wiki)
  - en: HuggingFaceFW/fineweb-edu               (educational web, dedup + PII removed)

Outputs (JSONL, one doc per line: {"text": ..., "lang": ...}):
  - data/corpus/train_zh.jsonl
  - data/corpus/train_en.jsonl
  - data/corpus/val.jsonl          (held-out docs)

Char budgets are rough proxies (~1 token per 1.7 zh chars / 4 en chars);
tokenize.py applies the exact 49M / 1M token split later.
"""
import argparse
import json
import time
from pathlib import Path

try:  # use Windows system CA store (fixes CERTIFICATE_VERIFY_FAILED)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

ZH_DATASET = "0xDing/wikipedia-cn-20230720-filtered"
EN_DATASET = "HuggingFaceFW/fineweb-edu"
EN_CONFIG = "sample-100BT"  # faster subset; fall back to default if unavailable

ZH_CHARS = 62_000_000    # ~15M tokens
EN_CHARS = 128_000_000   # ~30M tokens (tokenizer gives ~4.2 chars/token)
VAL_CHARS = 4_000_000    # held out across sources


def pick_text(row):
    for k in ("text", "completion", "content"):
        if k in row and isinstance(row[k], str) and len(row[k]) > 0:
            return row[k]
    if "conversations" in row and isinstance(row["conversations"], list):
        parts = [m.get("value", "") for m in row["conversations"] if isinstance(m, dict)]
        if parts:
            return "\n".join(parts)
    return None


def stream_sample(name, config, budget, lang, out_path, val_path, val_budget, retries=3):
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
                print(f"  retrying without config {config}", flush=True)
                config = None
            time.sleep(2)
    if ds is None:
        raise RuntimeError(f"cannot load {name}")

    f_train = open(out_path, "w", encoding="utf-8")
    f_val = open(val_path, "w", encoding="utf-8")
    train_chars = 0
    val_chars = 0
    n_docs = 0
    skipped = 0
    t0 = time.time()
    try:
        for row in ds:
            text = pick_text(row)
            if text is None:
                skipped += 1
                continue
            text = text.strip()
            if len(text) < 100:
                skipped += 1
                continue
            if val_chars < val_budget:
                f_val.write(json.dumps({"text": text, "lang": lang}, ensure_ascii=False) + "\n")
                val_chars += len(text)
            elif train_chars < budget:
                f_train.write(json.dumps({"text": text, "lang": lang}, ensure_ascii=False) + "\n")
                train_chars += len(text)
            else:
                break
            n_docs += 1
            if n_docs % 500 == 0:
                el = time.time() - t0
                print(f"  [{lang}] docs={n_docs} train_chars={train_chars/1e6:.1f}M "
                      f"val_chars={val_chars/1e6:.1f}M elapsed={el:.0f}s", flush=True)
    finally:
        f_train.close()
        f_val.close()
    print(f"[{lang}] done: {n_docs} docs, train {train_chars/1e6:.1f}M chars, "
          f"val {val_chars/1e6:.1f}M chars, skipped {skipped}, {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh-chars", type=int, default=ZH_CHARS)
    ap.add_argument("--en-chars", type=int, default=EN_CHARS)
    ap.add_argument("--val-chars", type=int, default=VAL_CHARS)
    ap.add_argument("--out-dir", default="data/corpus")
    ap.add_argument("--no-tool", action="store_true", help="skip generating tool dialogues")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    val_budget = args.val_chars // 3

    stream_sample(ZH_DATASET, None, args.zh_chars, "zh",
                  out / "train_zh.jsonl", out / "val.jsonl", val_budget)
    stream_sample(EN_DATASET, EN_CONFIG, args.en_chars, "en",
                  out / "train_en.jsonl", out / "val.jsonl", val_budget)

    if not args.no_tool:
        import subprocess
        subprocess.run([sys.executable, "make_tool_data.py",
                        "--out-corpus", str(out / "train_tool.jsonl")], check=True)

    print("corpus ready in", out)


if __name__ == "__main__":
    import sys
    main()