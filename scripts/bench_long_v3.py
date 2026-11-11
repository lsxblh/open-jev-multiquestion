"""V3 真实长序列的一次直出基准（A10 22GB 用）。

从 test 切分按编码长度取 p50 / p90 / max 三条真实 case，每条整条多问题
一次前向，测 p50/p90/p95/p99 与吞吐；再对最长 case 做小批量吞吐。

  python -u bench_long_v3.py \
    --adapter /mnt/workspace/runs/v3/adapter \
    --data /mnt/workspace/data/hybrid_v3 \
    --out /mnt/workspace/runs/v3/latency_long_a10.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import pyarrow.parquet as pq
import torch
from unsloth import FastDecisionModel
from unsloth.models.clef import encode_record


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, help="训练好的 adapter 目录")
    p.add_argument("--data", required=True, help="hybrid_v3 数据目录")
    p.add_argument("--out", required=True, help="结果 JSON 路径")
    p.add_argument("--max-seq", type=int, default=4500, help="必须显式与训练一致")
    p.add_argument("--n", type=int, default=32, help="每档计时次数")
    p.add_argument("--max-batch", type=int, default=4, help="最长 case 的批量上限")
    return p.parse_args()


ARGS = parse_args()


def sync() -> None:
    """等待 GPU 完成。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pct(ordered: list[float], q: float) -> float:
    """线性插值分位数。"""
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


t0 = time.perf_counter()
model, tok = FastDecisionModel.from_pretrained(
    model_name=ARGS.adapter, max_seq_length=ARGS.max_seq, load_in_4bit=True)
model.eval()
hf = getattr(tok, "tokenizer", tok)
sync()
load_s = round(time.perf_counter() - t0, 2)
print(f"[1] 加载 {load_s}s  decision_config.max_len="
      f"{model.decision_config.get('max_len')}", flush=True)
if model.decision_config.get("max_len") != ARGS.max_seq:
    raise SystemExit("max_len 与 --max-seq 不一致，会截断长序列，请检查")

rows = pq.read_table(f"{ARGS.data}/test.parquet").to_pylist()
sized = []
for r in rows:
    q = json.loads(r["questions"])
    n = len(encode_record(hf, {"state": r["state"], "questions": q, "images": None},
                          max_length=ARGS.max_seq).input_ids)
    sized.append((n, r, q))
sized.sort(key=lambda x: x[0])
print(f"[2] test 全量编码 {len(sized)} 条: min={sized[0][0]} "
      f"p50={sized[len(sized)//2][0]} max={sized[-1][0]}", flush=True)

picks = {
    "p50": sized[len(sized) // 2],
    "p90": sized[int(len(sized) * 0.9)],
    "max": sized[-1],
}
R = {
    "adapter": ARGS.adapter,
    "max_seq": ARGS.max_seq,
    "gpu": torch.cuda.get_device_name(0),
    "platform": platform.platform(),
    "cold_load_s": load_s,
    "n": ARGS.n,
    "pctl_method": "线性插值分位数，torch.cuda.synchronize() 后计时",
    "note": "整条多问题一次直出，无逐条合并；test 集真实 case，非合成样本",
    "cases": {},
}

for name, (ntok, row, q) in picks.items():
    state = row["state"]
    print(f"[3] {name}: {ntok} token / {len(q)} 题 / {row['source']}", flush=True)
    for _ in range(2):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            FastDecisionModel.predict(model, tok, state, q)
    sync()
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(ARGS.n):
        sync()
        t = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            FastDecisionModel.predict(model, tok, state, q)
        sync()
        times.append(time.perf_counter() - t)
    ordered = sorted(times)
    R["cases"][name] = {
        "source": row["source"],
        "input_tokens": ntok,
        "n_questions": len(q),
        "min": round(ordered[0], 3),
        "p50": round(pct(ordered, 0.50), 3),
        "p90": round(pct(ordered, 0.90), 3),
        "p95": round(pct(ordered, 0.95), 3),
        "p99": round(pct(ordered, 0.99), 3),
        "max": round(ordered[-1], 3),
        "mean": round(statistics.mean(times), 3),
        "throughput_qps": round(1 / statistics.mean(times), 3),
        "peak_mem_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 0),
    }
    print(f"    p50={R['cases'][name]['p50']}s  p95={R['cases'][name]['p95']}s  "
          f"qps={R['cases'][name]['throughput_qps']}  "
          f"peak={R['cases'][name]['peak_mem_mib']} MiB", flush=True)

# 最长 case 的批量吞吐（A10 22GB 也保守只到 4/8）
from unsloth.models.clef import collate_records

ntok, row, q = picks["max"]
device = next(model.parameters()).device
R["batch_on_max"] = {}
for bs in [x for x in (2, 4, 8) if x <= ARGS.max_batch]:
    try:
        recs = [encode_record(hf, {"state": row["state"], "questions": q,
                                   "images": None},
                              max_length=ARGS.max_seq) for _ in range(bs)]
        bb = collate_records(recs, hf.pad_token_id, device)
        sync()
        t = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=bb["input_ids"], attention_mask=bb["attention_mask"],
                  records=bb["records"], **bb["media"])
        sync()
        dt = time.perf_counter() - t
        R["batch_on_max"][str(bs)] = {
            "forward_s": round(dt, 3), "qps": round(bs / dt, 3),
            "peak_mem_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 0),
        }
        print(f"[4] batch={bs}: {dt:.3f}s  {bs/dt:.2f} req/s", flush=True)
    except Exception as exc:
        R["batch_on_max"][str(bs)] = {"error": str(exc)[:160]}
        print(f"[4] batch={bs} 失败: {str(exc)[:120]}", flush=True)

with open(ARGS.out, "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print("[DONE]", ARGS.out, flush=True)
