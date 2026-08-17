"""Chat inference + tool-calling loop for the fine-tuned MoE (MindSpore).

Usage:
    python chat.py --checkpoint runs/moe-200m-sft/sft_best_model.ckpt
    python chat.py --checkpoint ... --prompt "帮我算一下 3+5*2"   # single-shot
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import mindspore as ms

sys.path.insert(0, str(Path(__file__).parent))
import device as device_lib  # noqa: E402
from model import Config, MoETransformer  # noqa: E402


def build_prompt(messages: list[dict]) -> str:
    """ChatML renderer, mirrors prepare_sft.py."""
    out = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role in ("tool", "tool_result"):
            content = f"<|tool_result|>{content}<|tool_result|>"
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(out)


def parse_tool_call(text: str):
    """Extract a single JSON tool call after <|tool_call|>."""
    marker = "<|tool_call|>"
    if marker not in text:
        return None
    seg = text.split(marker, 1)[1].strip()
    start, end = seg.find("{"), seg.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(seg[start:end + 1])
    except Exception:
        return None


class ToolExecutor:
    def run(self, name: str, arguments: dict) -> str:
        if name == "calculator":
            expr = str(arguments.get("expression", "")).replace("^", "**")
            try:
                import re
                expr2 = re.sub(r"[^0-9+\-*/%()., eE ]", "", expr)
                val = eval(expr2, {"__builtins__": {}}, {})
                return f"{float(val):.10g}"
            except Exception:
                return "ERROR: invalid expression"
        if name == "get_datetime":
            import datetime
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if name == "get_weather":
            return f"{arguments.get('city', '')}: 晴，气温 22°C"
        if name == "web_search":
            return (f"关于“{arguments.get('query', '')}”的搜索结果："
                    f"该话题的相关资料显示其定义、历史沿革与当前现状。")
        return "ERROR: unknown tool"


def run_conversation(model, tok, args, messages: list[dict]) -> str:
    """One full turn: model may emit tool call(s), executor runs them, loop."""
    executor = ToolExecutor()
    max_iters = 4
    for _ in range(max_iters):
        prompt = build_prompt(messages)
        ids = tok.encode(prompt, out_type=int)
        idx = ms.Tensor(np.asarray([ids], np.int32), ms.int32)
        generated = model.generate(
            idx, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, eos_id=None)
        new_tokens = generated.asnumpy()[0, len(ids):].tolist()
        im_end_id = tok.piece_to_id("<|im_end|>")
        text_ids = []
        for t in new_tokens:
            if t == im_end_id:
                break
            text_ids.append(t)
        reply = tok.decode(text_ids).strip()

        tool_call = parse_tool_call(reply)
        if tool_call is None:
            messages.append({"role": "assistant", "content": reply})
            return reply

        name, args_ = tool_call.get("name"), tool_call.get("arguments", {})
        result = executor.run(name, args_ if isinstance(args_, dict) else {})
        messages.append({"role": "assistant",
                         "content": f"<|tool_call|>{json.dumps(tool_call, ensure_ascii=False)}"})
        messages.append({"role": "tool", "content": result})
        print(f"  [tool] {name}({json.dumps(args_, ensure_ascii=False)}) -> {result[:80]}", flush=True)
    return "(tool loop limit reached)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/moe-200m-sft/sft_best_model.ckpt")
    ap.add_argument("--sp-model", default="tokenizer/spm.model")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    import sentencepiece as spm
    tok = spm.SentencePieceProcessor(model_file=args.sp_model)

    device_lib.init_ms(mode="pynative")  # chat runs in PYNATIVE mode
    ckpt_path = str(args.checkpoint)
    if ckpt_path.endswith(".pt"):
        sys.exit("torch .pt checkpoint: 先运行 convert_ckpt.py 转换后再指定 .ckpt/.npz")
    if ckpt_path.endswith(".npz"):
        from train import load_npz_into_model
        model = MoETransformer(Config())
        load_npz_into_model(model, ckpt_path)
    else:
        meta_path = Path(ckpt_path).parent / \
            (Path(ckpt_path).stem.replace("_model", "") + "_meta.json")
        ckpt_cfg = (json.loads(meta_path.read_text(encoding="utf-8")).get("cfg", {})
                    if meta_path.exists() else {})
        cfg = Config(**{k: v for k, v in ckpt_cfg.items() if hasattr(Config, k)})
        model = MoETransformer(cfg)
        ms.load_param_into_net(model, ms.load_checkpoint(ckpt_path))
    nparams = sum(p.size for p in model.trainable_params())
    print(f"loaded {args.checkpoint} | params {nparams/1e6:.1f}M | "
          f"target={ms.get_context('device_target')}")

    if args.prompt:
        messages = [{"role": "system", "content": "你是一个有帮助的助手。当需要计算、查询时间、天气或搜索信息时，请调用相应的工具。"},
                    {"role": "user", "content": args.prompt}]
        ans = run_conversation(model, tok, args, messages)
        print(f"\n模型: {ans}")
        return

    print("输入 'exit' 退出。")
    messages = [{"role": "system", "content": "你是一个有帮助的助手。当需要计算、查询时间、天气或搜索信息时，请调用相应的工具。"}]
    while True:
        user = input("\n你: ").strip()
        if user.lower() in ("exit", "quit", "退出"):
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        ans = run_conversation(model, tok, args, messages)
        print(f"模型: {ans}")


if __name__ == "__main__":
    main()