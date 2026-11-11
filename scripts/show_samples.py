"""摘出各 source 的代表性真实样本，写成可读 md。"""
import json
import os
import statistics

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from unsloth.models.clef import encode_record

SRC = r"data/typed_full\train.parquet"
OUT = r"results/data_samples.md"
rows = pq.read_table(SRC).to_pylist()
tok = AutoTokenizer.from_pretrained(r"model/adapter_v3")

WANT = [
    "customer-control-v1",
    "workflow-controls-v1/invoice_processing",
    "workflow-controls-v1/customer_service",
    "workflow-controls-v1/security_incidents",
    "painting-geometry-v1",
    "snake-v1",
    "tic_tac_toe-v1",
    "reasoning-control-v1",
]

def enc_len(state, qs):
    return len(encode_record(tok, {"state":state,"questions":qs,"images":None},
                             max_length=200000).input_ids)

def short(s, n=1100):
    return s if len(s) <= n else s[:n] + f"\n... [原文共 {len(s)} 字符，此处截断]"

L = []
a = L.append
a("# 数据集真实样本\n")
for src in WANT:
    cands = [r for r in rows if r["source"] == src]
    if not cands:
        continue
    med = statistics.median([len(json.loads(r["questions"])) for r in cands])
    pick = min(cands, key=lambda r: (abs(len(json.loads(r["questions"])) - med),
                                     len(r["state"])))
    qs = json.loads(pick["questions"])
    gold = json.loads(pick["gold"])
    a(f"\n## {src}\n")
    a(f"- `group_id`: `{pick['group_id']}`")
    a(f"- 问题数: **{len(qs)}**　题型: `{sorted({q['type'] for q in qs.values()})}`")
    a(f"- 编码长度: **{enc_len(pick['state'], qs)} token**　state 字符数: {len(pick['state'])}")
    a(f"- 该 source 在 train 里共 {len(cands)} 个 case\n")
    a("### state（原文）\n")
    a("```text")
    a(short(pick["state"]))
    a("```\n")
    a("### questions（原文）\n")
    names = list(qs)
    show = qs if len(qs) <= 6 else {k: qs[k] for k in names[:2]}
    a("```json")
    a(json.dumps(show, ensure_ascii=False, indent=2)[:4200])
    a("```\n")
    if len(qs) > 6:
        a(f"> 只展示前 2 个问题，实际共 **{len(qs)}** 个：`{names[0]}`, `{names[1]}`, … , `{names[-1]}`\n")
    a("### gold（原文）\n")
    showg = gold if len(qs) <= 6 else {k: gold[k] for k in names[:2]}
    a("```json")
    a(json.dumps(showg, ensure_ascii=False, indent=2)[:4200])
    a("```\n")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("written", OUT, os.path.getsize(OUT), "bytes")
