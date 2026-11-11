"""逐条审查数据：哪些 case 不该拆、哪些必须拆、拆开后语义是否还成立。

不按 source 名称推断，只读每一条 questions 的 instructions 原文。
"""
from __future__ import annotations

import collections
import json
import os
import re
import statistics

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from unsloth.models.clef import encode_record

ROOT = r"data/typed_full\train.parquet"
OUT = r"results/split_legality.md"
TOK = r"model/adapter_v3"
MAX_SEQ = 4096

# 原文里表示「多个问题之间存在联合约束」的句子
JOINT = {
    "恰好选一个": r"Exactly one final disposition is selected",
    "动作集合": r"The output is an action set",
    "集合可为空": r"potentially empty",
    "动作基数约束": r"respect the workflow action cardinality",
    "各动作独立判断": r"Each action is an independent yes/no judgment",
}
# 原文里表示「本题引用了别的题的答案」
REF = r"(previous question|question above|answer above|prior answer|the selected action|chosen action|other actions|other selected)"


def enc_len(tok, state, questions) -> int:
    return len(encode_record(
        tok, {"state": state, "questions": questions, "images": None},
        max_length=200000).input_ids)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOK)
    rows = pq.read_table(ROOT).to_pylist()
    by_src = collections.defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)

    out = ["# 逐条审查：拆分合法性\n",
           f"审查对象：`typed_full/train.parquet` 全部 {len(rows):,} 个 case，"
           f"{sum(len(json.loads(r['questions'])) for r in rows):,} 道题。\n",
           "判定依据：逐条读 `instructions` 原文，而不是 source 名称。\n"]

    out.append("\n## 一、逐 source 裁决表\n")
    out.append("| source | case数 | 题数 | 编码p50/max | >4096的case | 含联合约束句 | 引用别题 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")

    detail = collections.defaultdict(lambda: {
        "n": 0, "q": 0, "lens": [], "over": 0,
        "joint": collections.Counter(), "ref": 0,
        "single_lens": [], "examples": [],
    })

    for src in sorted(by_src):
        d = detail[src]
        for r in by_src[src]:
            qs = json.loads(r["questions"])
            n = enc_len(tok, r["state"], qs)
            d["n"] += 1
            d["q"] += len(qs)
            d["lens"].append(n)
            if n > MAX_SEQ:
                d["over"] += 1
            for name, q in qs.items():
                ins = q.get("instructions", "") or ""
                for label, pat in JOINT.items():
                    if re.search(pat, ins, re.I):
                        d["joint"][label] += 1
                if re.search(REF, ins, re.I):
                    d["ref"] += 1
            # 试拆：单问题时的长度
            for name, q in qs.items():
                d["single_lens"].append(enc_len(tok, r["state"], {name: q}))
                break
            if len(d["examples"]) < 1 and len(qs) > 1:
                d["examples"].append((list(qs)[0], qs[list(qs)[0]].get("instructions", "")))
        v = sorted(d["lens"])
        out.append(
            f"| `{src}` | {d['n']} | {d['q']} | {v[len(v)//2]}/{v[-1]} | "
            f"{d['over']} | {sum(d['joint'].values())} | {d['ref']} |")

    out.append("\n## 二、联合约束原文（按 source）\n")
    for src in sorted(detail):
        d = detail[src]
        if not d["joint"]:
            continue
        out.append(f"\n### `{src}`\n")
        out.append(f"- 出现次数：{dict(d['joint'])}（共 {d['q']} 道题）")
        out.append(f"- 单题 instructions 原文（第一题）：\n")
        name, ins = d["examples"][0] if d["examples"] else ("", "")
        out.append("```text")
        out.append(f"[{name}]")
        out.append((ins or "")[:1400])
        out.append("```")

    out.append("\n## 三、试拆结果：拆成单问题后能否装进 4096\n")
    out.append("| source | 单题长度 p50/max | 是否全部 ≤4096 |")
    out.append("|---|---:|---|")
    for src in sorted(detail):
        d = detail[src]
        v = sorted(d["single_lens"])
        ok = "是" if v[-1] <= MAX_SEQ else "**否**"
        out.append(f"| `{src}` | {v[len(v)//2]}/{v[-1]} | {ok} |")

    out.append("\n## 四、机械结论\n")
    out.append("- `编码 >4096 的 case`：物理上必须处理（拆或排除）")
    out.append("- `含联合约束句`：拆开后该句仍留在单题 instructions 里，但约束对象已不存在 → 语义失真")
    out.append("- `引用别题`：题目文本依赖其他题的答案，拆开后信息缺失")
    out.append("- `试拆长度`：判断「这样拆」是否可行\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print("written", OUT)


if __name__ == "__main__":
    main()
