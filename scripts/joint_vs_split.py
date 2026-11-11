"""实验：拆分训练后的模型，是否还能「一个 state + 多问题一次直出」？"""
import json
import time

import pyarrow.parquet as pq
import torch
from unsloth import FastDecisionModel
from unsloth.models.clef import encode_record

ADAPTER = r"model/adapter_v3"
OUT = r"results/joint_vs_split.json"
full = pq.read_table(r"data/typed_full\train.parquet").to_pylist()

def norm(ans, kind):
    """把 predict 的 answer 归一成可与 gold label 比较的字符串。"""
    if kind == "noul":
        return "true" if ans is True else "false"
    return str(ans)

def pick(predicate, key, n=1):
    cands = [r for r in full if predicate(r)]
    cands.sort(key=key, reverse=True)
    return cands[:n]

# 样本 1：三种题型都有的 case
mixed = pick(
    lambda r: len(set(q["type"] for q in json.loads(r["questions"]).values())) >= 3,
    lambda r: len(json.loads(r["questions"])))[0]
# 样本 2：问题最多的 workflow case
many = max((r for r in full if r["source"].startswith("workflow-controls")),
           key=lambda r: len(json.loads(r["questions"])))

model, tok = FastDecisionModel.from_pretrained(
    model_name=ADAPTER, max_seq_length=1024, load_in_4bit=True)
model.eval()
hf = getattr(tok, "tokenizer", tok)
processor = tok if hasattr(tok, "tokenizer") else None

def run(case, label):
    """对比一次直出与逐条。"""
    qs = json.loads(case["questions"])
    gold = json.loads(case["gold"])
    state = case["state"]
    enc = encode_record(hf, {"state":state,"questions":qs,"images":None}, max_length=16384)
    print(f"\n=== {label}: {len(qs)} 个问题 / {len(enc.input_ids)} token ===", flush=True)

    # A: 一次直出
    t = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        joint = FastDecisionModel.predict(model, tok, state, qs)
    t_joint = time.perf_counter() - t

    # B: 逐条
    t = time.perf_counter()
    singles = {}
    for name, q in qs.items():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            a = FastDecisionModel.predict(model, tok, state, {name: q})
        singles[name] = a[name]
    t_single = time.perf_counter() - t

    rows = []
    for name, q in qs.items():
        kind = q["type"]
        lab = str(gold[name]["label"])
        pj = norm(joint[name].get("answer"), kind)
        ps = norm(singles[name].get("answer"), kind)
        probs_j = joint[name].get("probabilities") or {}
        probs_s = singles[name].get("probabilities") or {}
        agree = max(probs_j.values()) - max(probs_s.values()) if probs_j and probs_s else 0.0
        rows.append({
            "name": name, "type": kind, "gold": lab,
            "joint_answer": pj, "single_answer": ps,
            "joint_hit": pj == lab, "single_hit": ps == lab,
            "same": pj == ps,
            "joint_top_prob": round(max(probs_j.values()), 4) if probs_j else None,
            "single_top_prob": round(max(probs_s.values()), 4) if probs_s else None,
        })
    hj = sum(r["joint_hit"] for r in rows)
    hs = sum(r["single_hit"] for r in rows)
    same = sum(r["same"] for r in rows)
    print(f"  一次直出 : {hj}/{len(rows)} 命中, 耗时 {t_joint:.3f}s", flush=True)
    print(f"  逐条拆开 : {hs}/{len(rows)} 命中, 耗时 {t_single:.3f}s", flush=True)
    print(f"  两种方式答案相同: {same}/{len(rows)}", flush=True)
    for r in rows[:6]:
        print(f"    {r['name'][:26]:26s} gold={r['gold']:6s} "
              f"joint={r['joint_answer']:6s} single={r['single_answer']:6s} "
              f"p={r['joint_top_prob']}/{r['single_top_prob']}", flush=True)
    return {
        "label": label, "n_questions": len(qs), "input_tokens": len(enc.input_ids),
        "joint_hit": hj, "single_hit": hs, "same": same,
        "joint_seconds": round(t_joint, 3), "single_seconds": round(t_single, 3),
        "rows": rows,
    }

R = {"cases": [run(mixed, "混合题型"), run(many, "17 问题 workflow")]}
json.dump(R, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n[DONE]", OUT, flush=True)
