"""测首次推理（无 warmup）是否有编译开销。

本地： python cold_latency.py
云端： python cold_latency.py --adapter /mnt/workspace/runs/rv2/adapter \
                              --out /mnt/workspace/runs/cold_a10.json
"""
import argparse
import json
import time

import torch
from unsloth import FastDecisionModel


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=r"model/adapter_v3",
                        help="训练好的 adapter 目录")
    parser.add_argument("--out", default=r"results/cold_latency.json",
                        help="结果 JSON 路径")
    return parser.parse_args()


ARGS = parse_args()
ADAPTER = ARGS.adapter
STATE = "客户反馈：订单被重复扣款两次，已经等了三天，要求立刻退款，否则就投诉。"
Q = {
    "route": {"type":"choice","instructions":"这个工单应该交给哪个团队处理？",
              "criteria":{"billing":"账单、发票、退款相关","account":"账号登录、资料修改相关",
                          "technical":"程序缺陷、接口集成相关","other":"以上都不是"}},
    "urgency": {"type":"score","instructions":"这个工单的紧急程度如何？",
                "criteria":["可以正常排队等待","需要优先处理","需要立即人工介入"]},
    "refund_requested": {"type":"noul","instructions":"客户是否明确提出了退款请求？"},
}
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

t0 = time.perf_counter()
model, tok = FastDecisionModel.from_pretrained(
    model_name=ADAPTER, max_seq_length=4500, load_in_4bit=True)
model.eval()
sync()
load = time.perf_counter() - t0
ts = []
for i in range(5):
    sync()
    t = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        FastDecisionModel.predict(model, tok, STATE, Q)
    sync()
    ts.append(round(time.perf_counter() - t, 3))
print("load_s =", round(load, 2), flush=True)
print("第 1..5 次推理耗时(s) =", ts, flush=True)
r = {"load_s": round(load, 2), "first_5_latencies_s": ts}
with open(ARGS.out, "w", encoding="utf-8") as fh:
    json.dump(r, fh, ensure_ascii=False, indent=2)
