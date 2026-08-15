"""Download Chinese (SkyPile-150B) + English (fineweb-edu) text corpora.

Targets:
  zh: Skywork/SkyPile-150B (gated; 150B-token Chinese web)  1.40B chars -> ~0.8B tokens
  en: HuggingFaceFW/fineweb-edu sample-100BT                2.10B chars -> ~0.5B tokens

Writes data/corpus/train_zh3.jsonl / train_en3.jsonl (new suffix so existing
corpus files are untouched). build_bins.py merges all *.jsonl.

Note on SkyPile-150B:
  - gated dataset; pass a read token via --token or $env:HF_TOKEN
  - field is "text"; streaming is supported
  - there is also a small filtered wiki source 0xDing/wikipedia-cn (191M chars,
    already drained) kept in data/corpus/train_zh.jsonl

Usage:
    set HF_ENDPOINT=https://hf-mirror.com   (China mirror, optional)
    $env:HF_TOKEN = "hf_xxx"
    python download_text.py
    python download_text.py --zh-chars 100000000 --en-chars 200000000
"""
import argparse
import json
import time
from pathlib import Path

try:  # Windows system CA store
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

ZH_DATASET = "Skywork/SkyPile-150B"
EN_DATASET = "HuggingFaceFW/fineweb-edu"
EN_CONFIG = "sample-100BT"

ZH_CHARS = 1_400_000_000
EN_CHARS = 2_100_000_000


def pick_text(row):
    for k in ("text", "completion", "content"):
        if k in row and isinstance(row[k], str) and len(row[k]) > 0:
            return row[k]
    return None


def stream(name, config, budget, lang, out_path, token=None, retries=5):
    from datasets import load_dataset
    print(f"[{lang}] loading {name} (config={config}) ...", flush=True)
    ds = None
    for attempt in range(retries):
        try:
            kwargs = {"token": token} if token else {}
            ds = load_dataset(name, config, split="train", streaming=True, **kwargs)
            break
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {str(e)[:120]}", flush=True)
            if "gated" in str(e).lower() and token is None:
                print("  gated dataset: pass --token hf_xxx or set $env:HF_TOKEN", flush=True)
                raise SystemExit(1)
            if attempt == retries - 2 and config is not None:
                config = None
            time.sleep(3)
    if ds is None:
        raise RuntimeError(f"cannot load {name}")

    f = open(out_path, "w", encoding="utf-8")
    chars, n, skipped = 0, 0, 0
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
                print(f"  [{lang}] docs={n} chars={chars/1e6:.0f}M "
                      f"({chars/1e9:.2f}B) elapsed={time.time()-t0:.0f}s", flush=True)
            if chars >= budget:
                break
    finally:
        f.close()
    print(f"[{lang}] done: {n} docs, {chars/1e6:.0f}M chars, skipped {skipped} "
          f"-> {out_path} ({time.time()-t0:.0f}s)", flush=True)


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh-chars", type=int, default=ZH_CHARS)
    ap.add_argument("--en-chars", type=int, default=EN_CHARS)
    ap.add_argument("--out-dir", default="data/corpus")
    ap.add_argument("--token", default=None,
                    help="HF read token for gated datasets (or set $env:HF_TOKEN)")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stream(ZH_DATASET, None, args.zh_chars, "zh", out / "train_zh3.jsonl", token)
    stream(EN_DATASET, EN_CONFIG, args.en_chars, "en", out / "train_en3.jsonl", token)
    print("text corpus download complete.")


if __name__ == "__main__":
    main()
