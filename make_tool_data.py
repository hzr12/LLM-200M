"""Generate synthetic bilingual (zh/en) tool-calling dialogues.

Outputs:
  - data/corpus/train_tool.jsonl  : one JSON object per dialogue
      {"text": "<chatml text>", "lang": "zh", "messages": [...]}
  - data/sft_tool.jsonl           : same dialogues, structured for SFT

The chatml text uses reserved special tokens so the tokenizer learns them
as single units:
  <|im_start|>, <|im_end|>, <|tool_call|>, <|tool_result|>
"""
import argparse
import datetime as _dt
import json
import math
import random
import re
import sys
from pathlib import Path

TOOLS = {
    "calculator": {
        "desc": "Evaluate a mathematical expression.",
        "args": {"expression": "str"},
    },
    "get_datetime": {
        "desc": "Get the current date and time in a timezone.",
        "args": {"timezone": "str"},
    },
    "get_weather": {
        "desc": "Get the weather for a city.",
        "args": {"city": "str"},
    },
    "web_search": {
        "desc": "Search the web for information.",
        "args": {"query": "str"},
    },
}

CITIES = ["北京", "上海", "广州", "深圳", "成都", "杭州", "New York", "London", "Tokyo", "Paris"]
WEATHER = ["晴", "多云", "小雨", "阴", "晴朗", "sunny", "cloudy", "rainy", "overcast", "clear"]
WEEKDAYS_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _safe_eval(expr: str):
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/").replace("，", ",")
    expr = re.sub(r"[^0-9+\-*/%()., eE ]", "", expr)
    if not expr or any(ch in expr for ch in "abcdefghijklmnopqrstuvwxyz_"):
        return None
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        return float(val)
    except Exception:
        return None


def run_tool(name: str, args: dict) -> str:
    if name == "calculator":
        r = _safe_eval(args.get("expression", ""))
        if r is None:
            return "ERROR: invalid expression"
        return f"{r:.10g}"
    if name == "get_datetime":
        tz = args.get("timezone", "UTC")
        try:
            from zoneinfo import ZoneInfo
            if tz in ("CST", "UTC+8", "北京时间", "Asia/Shanghai"):
                tzi = ZoneInfo("Asia/Shanghai")
            elif tz and tz != "UTC":
                try:
                    tzi = ZoneInfo(tz)
                except Exception:
                    tzi = ZoneInfo("UTC")
            else:
                tzi = ZoneInfo("UTC")
            now = _dt.datetime.now(tzi)
            return now.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "ERROR: unknown timezone"
    if name == "get_weather":
        city = args.get("city", "")
        return f"{city}: {random.choice(WEATHER)}，气温 {random.randint(5, 35)}°C"
    if name == "web_search":
        q = args.get("query", "")
        return (f"搜索结果：关于“{q}”的百科页面显示，这是一个通用话题，"
                f"主要信息包括定义、历史沿革与现状。") if any("\u4e00" <= c <= "\u9fff" for c in q) else \
            (f"Search results for '{q}': Wikipedia-style summary: {q} is a general topic; "
             f"key information covers definition, history and current status.")
    return "ERROR: unknown tool"


def _tc(name, args):
    return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)


def _sample_expr():
    a, b = random.randint(1, 1000), random.randint(1, 1000)
    op = random.choice(["+", "-", "*", "/", "%"])
    return f"{a}{op}{b}"


def make_zh_dialogue(rng):
    kind = rng.random()
    expr = _sample_expr()
    city = rng.choice(CITIES[:6])
    now = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(hours=8)
    today_zh = WEEKDAYS_ZH[now.weekday()]
    date_zh = now.strftime("%Y年%m月%d日")
    a, b = rng.randint(1, 100), rng.randint(1, 100)
    weather = rng.choice(WEATHER)

    variants = [
        # single calculator call
        {
            "user": [
                f"帮我算一下 {expr} 等于多少？",
                f"计算 {expr} 的结果。",
                f"{expr} 是多少？",
                f"请帮我算一下{expr}。",
            ],
            "assistant": [
                "结果是 {r1}。",
                "计算结果为 {r1}。",
                "{expr} 等于 {r1}。",
                "这个算式的答案是 {r1}。",
            ],
            "steps": [("calculator", {"expression": expr})],
            "special": {"expr": expr},
        },
        # datetime call
        {
            "user": [
                "现在几点了？",
                "今天是什么日子？",
                "现在北京时间是多少？",
                "告诉我现在的日期和时间。",
            ],
            "assistant": [
                "现在是 {date} {time}。",
                "当前时间为 {time}，日期为 {date}。",
                "今天是{date}，现在是{time}。",
            ],
            "steps": [("get_datetime", {"timezone": "Asia/Shanghai"})],
            "special": {"date": date_zh, "time": f"{now.hour:02d}:{now.minute:02d}"},
        },
        # weather call
        {
            "user": [
                f"{city}今天天气怎么样？",
                f"查一下{city}的天气。",
                f"今天{city}适合出门吗？",
            ],
            "assistant": [
                "{city}今天的天气是{w}。",
                "根据查询结果，{city}今日{w}。",
                "{city}今天{w}，注意天气变化。",
            ],
            "steps": [("get_weather", {"city": city})],
            "special": {"city": city, "w": weather},
        },
        # multi-step: calc twice
        {
            "user": [
                f"帮我算 {a}+{b} 再乘以 2 是多少？",
                f"{a}+{b} 的结果乘以2等于几？",
            ],
            "assistant": [
                "先计算加法：{r1}，再乘以 2 得 {r2}。",
                "第一步 {a}+{b}={r1}，第二步 {r1}×2={r2}。",
            ],
            "steps": [("calculator", {"expression": f"{a}+{b}"}),
                      ("calculator", {"expression": f"({a}+{b})*2"})],
            "special": {"a": a, "b": b},
        },
        # web search
        {
            "user": [
                "帮我搜索一下神经网络的历史。",
                "搜索一下“变压器”是什么。",
                "请帮我查一下深度学习的发展历程。",
            ],
            "assistant": [
                "根据搜索结果：{r1}",
                "我搜索了一下，得到的结论是{r1}",
            ],
            "steps": [("web_search", {"query": "神经网络历史"})],
        },
    ]
    v = rng.choice(variants)
    q = rng.choice(v["user"])
    msgs = [{"role": "system", "content": "你是一个有帮助的助手。当需要计算、查询时间、天气或搜索信息时，请调用相应的工具。"}]
    msgs.append({"role": "user", "content": q})
    results = {}
    for i, (name, args) in enumerate(v["steps"]):
        res = run_tool(name, args)
        results[f"r{i+1}"] = res
        msgs.append({"role": "assistant", "content": f"<|tool_call|>{_tc(name, args)}"})
        msgs.append({"role": "tool", "content": res})
    if kind < 0.7:
        template = rng.choice(v["assistant"])
        final = template.format(**results, **v.get("special", {}))
        msgs.append({"role": "assistant", "content": final})
    else:
        msgs.append({"role": "assistant", "content": f"还需要我继续帮您处理其他问题吗？"})
    return msgs


def make_en_dialogue(rng):
    expr = _sample_expr()
    city = rng.choice(CITIES[6:])
    now = _dt.datetime.now(_dt.timezone.utc)
    today_en = WEEKDAYS_EN[now.weekday()]
    date_en = now.strftime("%B %d, %Y")
    a, b = rng.randint(1, 100), rng.randint(1, 100)

    variants = [
        {
            "user": [
                f"Can you calculate {expr}?",
                f"What is {expr}?",
                f"Please compute {expr}.",
                f"Could you work out {expr} for me?",
            ],
            "assistant": [
                "The result is {r1}.",
                "It equals {r1}.",
                "{expr} is {r1}.",
            ],
            "steps": [("calculator", {"expression": expr})],
            "special": {"expr": expr},
        },
        {
            "user": [
                "What time is it now?",
                "What is today's date?",
                "What is the current date and time?",
            ],
            "assistant": [
                "It is {time} UTC, on {date}.",
                "The current time is {time} UTC; today is {date}.",
            ],
            "steps": [("get_datetime", {"timezone": "UTC"})],
            "special": {"time": now.strftime("%H:%M:%S"), "date": f"{today_en}, {date_en}"},
        },
        {
            "user": [
                f"What is the weather in {city}?",
                f"Check the weather for {city}.",
                f"How is the weather in {city} today?",
            ],
            "assistant": [
                "The weather in {city} is {r1}.",
                "According to the data, {city} has {r1}.",
            ],
            "steps": [("get_weather", {"city": city})],
            "special": {"city": city},
        },
        {
            "user": [
                f"Calculate {a}+{b} and then multiply by 2.",
                f"What is ({a}+{b}) times 2?",
            ],
            "assistant": [
                "First {a}+{b}={r1}, then times 2 gives {r2}.",
                "Step one: {a}+{b}={r1}. Step two: multiply by 2 → {r2}.",
            ],
            "steps": [("calculator", {"expression": f"{a}+{b}"}),
                      ("calculator", {"expression": f"({a}+{b})*2"})],
            "special": {"a": a, "b": b},
        },
        {
            "user": [
                "Search the web for the history of neural networks.",
                "Could you look up what transformers are?",
            ],
            "assistant": [
                "Here is what I found: {r1}",
                "Based on my search: {r1}",
            ],
            "steps": [("web_search", {"query": "history of neural networks"})],
        },
    ]
    v = rng.choice(variants)
    q = rng.choice(v["user"])
    msgs = [{"role": "system", "content": "You are a helpful assistant. Use tools when calculation, time, weather or search is needed."}]
    msgs.append({"role": "user", "content": q})
    results = {}
    for i, (name, args) in enumerate(v["steps"]):
        res = run_tool(name, args)
        results[f"r{i+1}"] = res
        msgs.append({"role": "assistant", "content": f"<|tool_call|>{_tc(name, args)}"})
        msgs.append({"role": "tool", "content": res})
    if rng.random() < 0.7:
        msgs.append({"role": "assistant", "content": rng.choice(v["assistant"]).format(**results, **v.get("special", {}))})
    else:
        msgs.append({"role": "assistant", "content": "Is there anything else I can help you with?"})
    return msgs


def chatml(msgs):
    out = []
    for m in msgs:
        role = m["role"]
        content = m["content"]
        if role in ("tool", "tool_result"):
            # wrap tool results with the reserved <|tool_result|> token so the
            # tokenizer treats it as a single unit (aligned with chat_template)
            content = f"<|tool_result|>{content}<|tool_result|>"
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(out)


def gen(n_zh, n_en, seed, out_corpus, out_sft):
    rng = random.Random(seed)
    Path(out_corpus).parent.mkdir(parents=True, exist_ok=True)
    with open(out_corpus, "w", encoding="utf-8") as fc, open(out_sft, "w", encoding="utf-8") as fs:
        n = 0
        for lang, count, fn in (("zh", n_zh, make_zh_dialogue), ("en", n_en, make_en_dialogue)):
            for _ in range(count):
                msgs = fn(rng)
                text = chatml(msgs)
                rec = {"text": text, "lang": lang, "messages": msgs}
                fc.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fs.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 2000 == 0:
                    print(f"  generated {n} dialogues", flush=True)
    print(f"done: {n} dialogues -> {out_corpus}, {out_sft}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-zh", type=int, default=8000)
    ap.add_argument("--n-en", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-corpus", default="data/corpus/train_tool.jsonl")
    ap.add_argument("--out-sft", default="data/sft_tool.jsonl")
    args = ap.parse_args()
    gen(args.n_zh, args.n_en, args.seed, args.out_corpus, args.out_sft)
