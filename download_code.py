"""Download code corpora (streaming, char-budget capped).

Sources:
  - bigcode/starcoderdata  (GATED - needs HF account + terms + token; multi-language)
  - codeparrot/codeparrot-clean (NOT gated; Python only) as fallback

Targets (starcoderdata, ~6B chars -> ~1.5B tokens at ~4 chars/token):
  python 2.4B chars | javascript 1.2B | html 1.2B | css 0.6B | java 0.6B

Writes one JSONL per language into data/corpus/ (same format as existing
corpus files: {"text": ..., "lang": ...}) so build_bins.py picks them up.

--- GATED DATASET SETUP (required for starcoderdata) ---
1. Create/login an account at https://huggingface.co
2. Open https://huggingface.co/datasets/bigcode/starcoderdata and click
   "Agree and send request to access repo" (usually auto-approved)
3. Create a read token at https://huggingface.co/settings/tokens
4. Expose it when running:
       $env:HF_TOKEN = "hf_xxx"          # PowerShell
   or  python download_code.py --token hf_xxx

Usage:
    python download_code.py                       # all 5 languages (starcoder)
    python download_code.py --langs python java   # subset
    python download_code.py --source codeparrot   # no-auth fallback (Python only)
"""
import argparse
import json
import os
import time
from pathlib import Path

try:  # Windows system CA store
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# language -> target chars (about 1.5B tokens total at 4 chars/token)
BUDGETS = {
    "python": 2_400_000_000,
    "javascript": 1_200_000_000,
    "html": 1_200_000_000,
    "css": 600_000_000,
    "java": 600_000_000,
}

STARCODER = "bigcode/starcoderdata"
CODEPARROT = "codeparrot/codeparrot-clean"

GATED_HELP = (
    "bigcode/starcoderdata is a GATED dataset. You must:\n"
    "  1) log in at https://huggingface.co\n"
    "  2) open https://huggingface.co/datasets/bigcode/starcoderdata and click "
    "'Agree and send request to access repo'\n"
    "  3) create a READ token at https://huggingface.co/settings/tokens\n"
    "  4) run with --token hf_xxx or set $env:HF_TOKEN\n"
    "Or switch to the non-gated Python-only source: --source codeparrot"
)


def stream_lang(source: str, lang: str, target_chars: int, out_path: Path,
                token: str | None, retries=5):
    from datasets import load_dataset

    ds = None
    for attempt in range(retries):
        try:
            print(f"[{lang}] loading {source} (data_dir={lang}) ...", flush=True)
            kwargs = {"token": token} if token else {}
            if source == CODEPARROT:
                # codeparrot-clean: single python config, no data_dir
                ds = load_dataset(source, split="train", streaming=True, **kwargs)
            else:
                # starcoderdata: language is selected via data_dir (single default config)
                ds = load_dataset(source, data_dir=lang, split="train",
                                  streaming=True, **kwargs)
            break
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {str(e)[:150]}", flush=True)
            if "gated" in str(e).lower():
                print("\n" + GATED_HELP + "\n", flush=True)
                raise SystemExit(1)
            time.sleep(3)
    if ds is None:
        raise RuntimeError(f"cannot load {source}/{lang}")

    f = open(out_path, "w", encoding="utf-8")
    chars, n, skipped = 0, 0, 0
    t0 = time.time()
    try:
        for row in ds:
            text = row.get("content") or row.get("text") or ""
            if not text or len(text) < 50:
                skipped += 1
                continue
            f.write(json.dumps({"text": text, "lang": f"code_{lang}"},
                               ensure_ascii=False) + "\n")
            chars += len(text)
            n += 1
            if n % 500 == 0:
                print(f"  [{lang}] docs={n} chars={chars/1e6:.0f}M "
                      f"({chars/1e9:.2f}B) elapsed={time.time()-t0:.0f}s", flush=True)
            if chars >= target_chars:
                break
    finally:
        f.close()
    print(f"[{lang}] done: {n} docs, {chars/1e6:.0f}M chars, skipped {skipped} "
          f"-> {out_path} ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="starcoder",
                    choices=["starcoder", "codeparrot"],
                    help="starcoder = multi-language (GATED); codeparrot = Python-only (no auth)")
    ap.add_argument("--langs", nargs="+",
                    default=["python", "javascript", "html", "css", "java"])
    ap.add_argument("--target-chars", type=int, default=0,
                    help="override all budgets (0 = use per-language defaults)")
    ap.add_argument("--token", default=None,
                    help="HF read token for gated datasets (or set $env:HF_TOKEN)")
    ap.add_argument("--out-dir", default="data/corpus")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if args.source == "starcoder":
        source = STARCODER
        # starcoder has js under config "javascript"
        langs = args.langs
    else:
        source = CODEPARROT
        # codeparrot-clean only has a python config; single pass with a big budget
        langs = ["python"]
        print("codeparrot-clean is Python-only; overriding --langs to [python]")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        budget = args.target_chars or BUDGETS.get(lang, 600_000_000)
        stream_lang(source, lang, budget, out / f"code_{lang}.jsonl", token)
    print("code corpus download complete.")


if __name__ == "__main__":
    main()
