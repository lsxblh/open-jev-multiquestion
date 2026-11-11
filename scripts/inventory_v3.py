"""盘点 V3 数据集：5 个切分的条数、形态、来源构成。只读，不改数据。"""
from __future__ import annotations

import collections
import json
import os

import pyarrow.parquet as pq

ROOT = r"data/hybrid_v3"
FULL = r"data/typed_full"
SPLITS = ("train", "calibration", "validation", "test", "ood")
OUT = r"results/v3_inventory.md"
PT = "painting-geometry-v1"

L = []
a = L.append
a("# V3 数据集盘点\n")
a("数据目录：`jev/data/hybrid_v3/`，`max_seq=4500`，规则见 `manifest.json`。\n")

a("## 1. 五个切分\n")
a("| split | case 数 | 题数 | N=1 case | N>1 case | 每题最多候选 | groups | 文件 KB |")
a("|---|---:|---:|---:|---:|---:|---:|---:|")
summary = {}
for sp in SPLITS:
    p = os.path.join(ROOT, sp + ".parquet")
    rows = pq.read_table(p).to_pylist()
    nq = [len(json.loads(r["questions"])) for r in rows]
    max_opt = 0
    for r in rows:
        for q in json.loads(r["questions"]).values():
            c = q.get("criteria")
            max_opt = max(max_opt, len(c) if isinstance(c, (dict, list)) else 2)
    groups = {r["group_id"] for r in rows}
    summary[sp] = rows
    a(f"| `{sp}` | {len(rows):,} | {sum(nq):,} | {sum(1 for x in nq if x==1):,} | "
      f"{sum(1 for x in nq if x>1):,} | {max_opt} | {len(groups):,} | "
      f"{os.path.getsize(p)/1024:.0f} |")

a("\n## 2. train 的构成：原生 case + painting 拆出的片段\n")
train = summary["train"]
native = [r for r in train if r["source"] != PT]
frag = [r for r in train if r["source"] == PT]
a(f"- **原生 case：{len(native):,}**（1 state + 多问题，不拆不合并）")
a(f"- **painting 单问题片段：{len(frag):,}**（23,552 条里分层抽 5%，seed 3407）")
a(f"- **合计：{len(train):,}**\n")
a("| 原生 case 按 source | case 数 | 题数 | 题型 |")
a("|---|---:|---:|---|")
by_src = collections.defaultdict(lambda: {"case": 0, "q": 0, "types": collections.Counter()})
for r in native:
    qs = json.loads(r["questions"])
    d = by_src[r["source"]]
    d["case"] += 1
    d["q"] += len(qs)
    for q in qs.values():
        d["types"][q["type"]] += 1
for s, d in sorted(by_src.items(), key=lambda kv: -kv[1]["q"]):
    t = "/".join(f"{k} {v:,}" for k, v in d["types"].most_common())
    a(f"| `{s}` | {d['case']:,} | {d['q']:,} | {t} |")

a("\n## 3. 全库题型分布（V3）\n")
a("| split | choice | score | noul | 合计 |")
a("|---|---:|---:|---:|---:|")
for sp in SPLITS:
    c = collections.Counter()
    for r in summary[sp]:
        for q in json.loads(r["questions"]).values():
            c[q["type"]] += 1
    a(f"| `{sp}` | {c['choice']:,} | {c['score']:,} | {c['noul']:,} | {sum(c.values()):,} |")

a("\n## 4. 评测集相对原始切分排除了什么\n")
a("排除的只有 painting（4500 装不下），其余原样保留。\n")
a("| split | V3 case | 原始 case | 排除 case | 排除题数 | 占原始题数 |")
a("|---|---:|---:|---:|---:|---:|")
for sp in SPLITS[1:]:
    orig = pq.read_table(os.path.join(FULL, sp + ".parquet")).to_pylist()
    pt = [r for r in orig if r["source"] == PT]
    ptq = sum(len(json.loads(r["questions"])) for r in pt)
    origq = sum(len(json.loads(r["questions"])) for r in orig)
    a(f"| `{sp}` | {len(summary[sp]):,} | {len(orig):,} | {len(pt)} | "
      f"{ptq:,} | {100*ptq/origq:.1f}% |")

a("\n## 5. 一条 case 的字段\n")
a("```text")
a("state      情境文本（JSON 字符串或对话）")
a("questions  {问题名: {type, instructions, criteria}}   JSON 字符串")
a("gold       {问题名: {type, label, probabilities, ...}} JSON 字符串")
a("group_id   同一原始 case 的分身共享，防跨 split 泄漏")
a("source     12 选 1（评测集里 painting 已排除，剩 11 个）")
a("n_questions")
a("```")
a("\n题型：`choice`（多项，criteria 是 dict）、`score`（有序档位，criteria 是 list）、"
  "`noul`（布尔，无 criteria）。\n")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print("written", OUT)
