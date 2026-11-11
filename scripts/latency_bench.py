"""推理延时 benchmark：冷启动 / 单次 / 拆分 / 批量 / 显存。

本地：  python latency_bench.py
云端：  python latency_bench.py --adapter /mnt/workspace/runs/rv2/adapter \
                                --out /mnt/workspace/runs/latency_a10.json --max-batch 32
"""
import argparse
import json
import platform
import statistics
import time

import torch
from unsloth import FastDecisionModel
from unsloth.models.clef import encode_record


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=r"model/adapter_v3",
                        help="训练好的 adapter 目录")
    parser.add_argument("--out", default=r"results/latency_bench.json",
                        help="结果 JSON 路径")
    parser.add_argument("--max-batch", type=int, default=8,
                        help="最大测试 batch（本地 6GB 建议 8，A10 24GB 可到 32）")
    parser.add_argument("--max-seq", type=int, default=4500,
                        help="与训练时一致的 max_seq_length（v3 = 4500）")
    return parser.parse_args()


ARGS = parse_args()
ADAPTER = ARGS.adapter
OUT = ARGS.out
STATE = "客户反馈：订单被重复扣款两次，已经等了三天，要求立刻退款，否则就投诉。"
Q3 = {
    "route": {"type":"choice","instructions":"这个工单应该交给哪个团队处理？",
              "criteria":{"billing":"账单、发票、退款相关","account":"账号登录、资料修改相关",
                          "technical":"程序缺陷、接口集成相关","other":"以上都不是"}},
    "urgency": {"type":"score","instructions":"这个工单的紧急程度如何？",
                "criteria":["可以正常排队等待","需要优先处理","需要立即人工介入"]},
    "refund_requested": {"type":"noul","instructions":"客户是否明确提出了退款请求？"},
}
Q1 = {"route": Q3["route"]}
R = {}

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

t0 = time.perf_counter()
model, tok = FastDecisionModel.from_pretrained(
    model_name=ADAPTER, max_seq_length=ARGS.max_seq, load_in_4bit=True)
model.eval()
sync()
R["cold_load_s"] = round(time.perf_counter() - t0, 2)
print(f"[1] 冷启动加载: {R['cold_load_s']}s", flush=True)
hf = getattr(tok, "tokenizer", tok)
processor = tok if hasattr(tok, "tokenizer") else None
device = next(model.parameters()).device

def pct(sorted_values, quantile):
    """线性插值分位数。

    Args:
        sorted_values: 已升序排列的耗时列表。
        quantile: 0~1 之间的分位点。

    Returns:
        插值后的分位数值。
    """
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * quantile
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def measure(state, questions, n=32):
    """端到端计时，返回 p50 / p90 / p95 / p99 与吞吐。"""
    times = []
    for _ in range(n):
        sync()
        t = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            FastDecisionModel.predict(model, tok, state, questions)
        sync()
        times.append(time.perf_counter() - t)
    ordered = sorted(times)
    return {
        "n": n, "min": round(ordered[0], 3),
        "p50": round(pct(ordered, 0.50), 3),
        "p90": round(pct(ordered, 0.90), 3),
        "p95": round(pct(ordered, 0.95), 3),
        "p99": round(pct(ordered, 0.99), 3),
        "max": round(ordered[-1], 3), "mean": round(statistics.mean(times), 3),
        "throughput_qps": round(1 / statistics.mean(times), 3),
    }

# warmup
print("[2] warmup ...", flush=True)
for _ in range(2):
    FastDecisionModel.predict(model, tok, STATE, Q3)
sync()

R["e2e_3q"] = measure(STATE, Q3)
print(f"[3] 端到端 3 问题 / 9 选项: p50={R['e2e_3q']['p50']}s  "
      f"qps={R['e2e_3q']['throughput_qps']}", flush=True)
R["e2e_1q"] = measure(STATE, Q1)
print(f"[4] 端到端 1 问题 / 4 选项: p50={R['e2e_1q']['p50']}s", flush=True)

# 拆分：encode / forward / 后处理
enc = encode_record(hf, {"state":STATE,"questions":Q3,"images":None},
                    max_length=ARGS.max_seq, processor=processor)
from unsloth.models.clef import collate_records
batch = collate_records([enc], hf.pad_token_id, device)
parts = {"encode": [], "forward": [], "post": []}
for _ in range(6):
    t = time.perf_counter()
    e = encode_record(hf, {"state":STATE,"questions":Q3,"images":None},
                      max_length=ARGS.max_seq, processor=processor)
    b = collate_records([e], hf.pad_token_id, device)
    parts["encode"].append(time.perf_counter() - t)
    sync()
    t = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                          records=b["records"], **b["media"])
    sync()
    parts["forward"].append(time.perf_counter() - t)
    t = time.perf_counter()
    _ = (logits.float().cpu() / 1.0).softmax(-1)
    parts["post"].append(time.perf_counter() - t)
R["breakdown_s"] = {k: round(statistics.mean(v), 4) for k, v in parts.items()}
R["input_tokens"] = len(enc.input_ids)
print(f"[5] 拆分: {R['breakdown_s']}  tokens={R['input_tokens']}", flush=True)

# 批量吞吐（同一条重复 B 次）
R["batch"] = {}
batch_sizes = [x for x in (2, 4, 8, 16, 32) if x <= ARGS.max_batch]
for bs in batch_sizes:
    try:
        torch.cuda.reset_peak_memory_stats()
        recs = [encode_record(hf, {"state":STATE,"questions":Q3,"images":None},
                              max_length=ARGS.max_seq, processor=processor) for _ in range(bs)]
        bb = collate_records(recs, hf.pad_token_id, device)
        sync()
        t = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=bb["input_ids"], attention_mask=bb["attention_mask"],
                  records=bb["records"], **bb["media"])
        sync()
        dt = time.perf_counter() - t
        R["batch"][str(bs)] = {
            "forward_s": round(dt, 3),
            "qps": round(bs/dt, 3),
            "peak_mem_mib": round(torch.cuda.max_memory_allocated()/1024**2, 0),
        }
        print(f"[6] batch={bs}: {dt:.3f}s  {bs/dt:.2f} req/s  "
              f"peak={R['batch'][str(bs)]['peak_mem_mib']} MiB", flush=True)
    except Exception as exc:
        R["batch"][str(bs)] = {"error": str(exc)[:160]}
        print(f"[6] batch={bs} 失败: {str(exc)[:120]}", flush=True)

R["peak_mem_mib"] = round(torch.cuda.max_memory_allocated()/1024**2, 0)
R["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
R["adapter"] = ADAPTER
R["max_seq"] = ARGS.max_seq
R["sample"] = "1 state + 3 questions / 9 options（一次直出）"
R["pctl_method"] = "线性插值分位数，n=32，torch.cuda.synchronize() 后计时"
R["platform"] = platform.platform()
R["note"] = (f"{platform.system()} / {R['gpu']}；吞吐为单请求串行 qps 与批量 qps，"
             "非服务端并发压测；kernel 支持情况（FA2 / causal_conv1d）以启动日志为准")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print("[DONE]", OUT, flush=True)
