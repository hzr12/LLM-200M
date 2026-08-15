"""Train a sentencepiece BPE tokenizer (vocab 16K) on the prepared corpus.

Special symbols are registered via user_defined_symbols so they are kept as
single tokens by encode() (control_symbols would split them into pieces):
  <|im_start|>, <|im_end|>, <|tool_call|>, <|tool_result|>, <pad>
"""
import argparse
from pathlib import Path

import sentencepiece as spm

CONTROL_SYMBOLS = ["<|im_start|>", "<|im_end|>", "<|tool_call|>", "<|tool_result|>", "<pad>"]


def make_corpus_files(corpus_dir: Path, sample_chars: int, out_path: Path):
    """Concatenate corpus jsonl into a plain-text file for SP training.

    Samples evenly across all corpus files (alphabetical order would let
    one file eat the whole budget and starve the others).
    """
    files = [f for f in sorted(corpus_dir.glob("*.jsonl")) if "val" not in f.name]
    per_file = sample_chars // max(1, len(files))
    lines = []
    for f in files:
        total = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    text = __import__("json").loads(line)["text"]
                except Exception:
                    continue
                lines.append(text)
                total += len(text)
                if total >= per_file:
                    break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"tokenizer corpus: {len(lines)} docs, {sum(len(l) for l in lines)/1e6:.1f}M chars -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="data/corpus")
    ap.add_argument("--out-dir", default="tokenizer")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--sample-chars", type=int, default=10_000_000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text_path = out / "spm_input.txt"
    make_corpus_files(Path(args.corpus_dir), args.sample_chars, text_path)

    spm.SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix=str(out / "spm"),
        vocab_size=args.vocab_size - len(CONTROL_SYMBOLS),
        model_type="bpe",
        character_coverage=0.9995,
        bos_id=-1,
        eos_id=-1,
        unk_id=0,
        pad_id=-1,
        user_defined_symbols=CONTROL_SYMBOLS,
        num_threads=8,
        shuffle_input_sentence=True,
    )
    sp = spm.SentencePieceProcessor(model_file=str(out / "spm.model"))
    if sp.vocab_size() != args.vocab_size:
        print(f"note: SP produced vocab {sp.vocab_size()} (target {args.vocab_size}); "
              f"meta.json will carry the actual size")
    ids = {t: sp.piece_to_id(t) for t in CONTROL_SYMBOLS}
    print("special token ids:", ids)
    print("tokenizer ready:", out / "spm.model")


if __name__ == "__main__":
    main()