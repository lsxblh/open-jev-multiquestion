"""零新增参数对照：用冻结底座的 LM head 在选项字母 token 上取 logprob。

不加载决策头、不做任何训练，与 V3 用同一份 test.parquet、同一份 4-bit 底座。
输出逐题 accuracy / 分题型 accuracy，用于对比判别式改造的收益。

用法：
    python baseline_logprob.py --limit 24                 # 测速
    python baseline_logprob.py --out results/baseline_logprob.json   # 全量
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "models/Qwen3.5-4B"
TEST = "data/hybrid_v3/test.parquet"
LETTERS = "ABCDEFG"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=BASE)
    p.add_argument("--test", default=TEST)
    p.add_argument("--max-seq", type=int, default=4500)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0 为全部")
    p.add_argument("--out", default="results/baseline_logprob.json")
    return p.parse_args()


ARGS = parse_args()


def options_of(q: dict) -> list[str]:
    """把三种题型统一成选项文本列表。"""
    kind = q.get("type")
    if kind == "choice":
        return [f"{k}: {v}" for k, v in q["criteria"].items()]
    if kind == "score":
        return [f"{i}: {v}" for i, v in enumerate(q["criteria"])]
    return ["false", "true"]


def gold_label(g: dict) -> str:
    """从 gold 取硬标签。"""
    return str(g.get("label"))


def label_to_index(kind: str, label: str, q: dict) -> int | None:
    """把硬标签映射成选项下标。"""
    if kind == "choice":
        keys = list(q["criteria"].keys())
        return keys.index(label) if label in keys else None
    if kind == "score":
        return int(label) if label.isdigit() else None
    return 0 if label.lower() == "false" else 1


def build_prompt(state: str, q: dict) -> str:
    opts = options_of(q)
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
    return (f"{state}\n\n问题：{q.get('instructions','')}\n"
            f"选项：\n{lines}\n\n答案：")


def main():
    os.makedirs(os.path.dirname(ARGS.out), exist_ok=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(ARGS.base)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        ARGS.base, quantization_config=quant, device_map={"": 0},
    )
    model.eval()
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

    # 候选字母的 token id：兼容带/不带前导空格
    cand_ids = {}
    for c in LETTERS:
        for s in (c, " " + c):
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in cand_ids:
                cand_ids[ids[0]] = c
                break
    print(f"[tokens] 字母 token 映射: {cand_ids}", flush=True)

    rows = pq.read_table(ARGS.test).to_pylist()
    tasks = []
    for r in rows:
        gold = json.loads(r["gold"])
        for name, q in json.loads(r["questions"]).items():
            if name not in gold:
                continue
            tasks.append((r["source"], r["state"], name, q, gold[name]))
    if ARGS.limit:
        tasks = tasks[:ARGS.limit]
    print(f"[data] 待评题数 {len(tasks)}", flush=True)

    hits, by_kind, times = 0, {}, []
    t1 = time.perf_counter()
    for idx, (source, state, name, q, g) in enumerate(tasks, 1):
        opts = options_of(q)
        k = q.get("type", "unknown")
        prompt = build_prompt(state, q)
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=ARGS.max_seq).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
        ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS[:len(opts)]]
        pick = int(torch.argmax(logits[ids]).item())
        want = label_to_index(k, gold_label(g), q)
        ok = int(pick == want)
        hits += ok
        by_kind.setdefault(k, [0, 0])
        by_kind[k][0] += ok
        by_kind[k][1] += 1
        times.append(time.perf_counter() - t1)
        t1 = time.perf_counter()
        if idx % 25 == 0 or idx == len(tasks):
            print(f"  [{idx}/{len(tasks)}] acc={hits/idx:.4f} "
                  f"avg={sum(times)/len(times):.3f}s/题", flush=True)

    result = {
        "method": "LM head logprob on option-letter tokens, zero new parameters",
        "base": ARGS.base,
        "test": ARGS.test,
        "n": len(tasks),
        "limit": ARGS.limit,
        "accuracy": round(hits / len(tasks), 4) if tasks else None,
        "by_kind": {k: {"accuracy": round(v[0] / v[1], 4), "n": v[1]}
                    for k, v in by_kind.items()},
        "avg_seconds_per_item": round(sum(times) / len(times), 3) if times else None,
        "total_seconds": round(sum(times), 1),
        "note": "零新增参数基线；prompt 为中性模板，未做 prompt 调优。"
                "与 V3 用同一份 test.parquet 与同一 4-bit 底座。",
    }
    with open(ARGS.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
