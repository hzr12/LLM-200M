"""Chat inference + tool-calling loop for the fine-tuned MoE.

Usage:
    python chat.py --checkpoint runs/moe-200m-sft/sft_best.pt
    python chat.py --checkpoint ... --prompt "帮我算一下 3+5*2"   # single-shot

The model uses ChatML with reserved tokens:
    <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>
Tool calls are emitted as:
    <|im_start|>assistant\n<|tool_call|>{"name": "...", "arguments": {...}}<|im_end|>
Tool results are fed back wrapped with <|tool_result|>.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import device as device_lib
from model import Config, MoETransformer


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
    # JSON is usually the last {...} in the segment
    start, end = seg.find("{"), seg.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(seg[start:end + 1])
    except Exception:
        return None


class ToolExecutor:
    def __init__(self):
        pass

    def run(self, name: str, arguments: dict) -> str:
        if name == "calculator":
            expr = str(arguments.get("expression", ""))
            expr = expr.replace("^", "**")
            try:
                # safe eval: only numbers/operators
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


@torch.no_grad()
def run_conversation(model, tok, args, messages: list[dict]) -> str:
    """One full turn: model may emit tool call(s), executor runs them, loop until final answer."""
    executor = ToolExecutor()
    max_iters = 4
    for _ in range(max_iters):
        prompt = build_prompt(messages)
        ids = tok.encode(prompt, out_type=int)
        idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
        generated = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=None,  # rely on <|im_end|> stop below
        )
        new_tokens = generated[0, len(ids):].tolist()
        # stop at first <|im_end|>
        im_end_id = tok.piece_to_id("<|im_end|>")
        text_ids = []
        for t in new_tokens:
            if t == im_end_id:
                break
            text_ids.append(t)
        reply = tok.decode(text_ids).strip()

        tool_call = parse_tool_call(reply)
        if tool_call is None:
            # final answer
            messages.append({"role": "assistant", "content": reply})
            return reply

        # run tool, push result back
        name, args_ = tool_call.get("name"), tool_call.get("arguments", {})
        result = executor.run(name, args_ if isinstance(args_, dict) else {})
        messages.append({"role": "assistant", "content": f"<|tool_call|>{json.dumps(tool_call, ensure_ascii=False)}"})
        messages.append({"role": "tool", "content": result})
        print(f"  [tool] {name}({json.dumps(args_, ensure_ascii=False)}) -> {result[:80]}", flush=True)
    return "(tool loop limit reached)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/moe-200m-sft/sft_best.pt")
    ap.add_argument("--sp-model", default="tokenizer/spm.model")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    import sentencepiece as spm
    tok = spm.SentencePieceProcessor(model_file=args.sp_model)

    device = device_lib.get_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["cfg"].items() if k in Config().__dict__})
    model = MoETransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} | params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

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
