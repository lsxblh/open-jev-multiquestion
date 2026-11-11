"""生成 v3 数据集。

规则（用户 2026-11-06 确认）：
- max_seq = 4500
- train: 排除 painting 的 14,403 个原生 case 原样保留；
         painting 的 138 个超长 case 拆成 23,552 个单问题片段，
         按 (case 尺寸, label) 分层随机抽 5% = 1,177 条。
- calibration / validation / test / ood: 排除 painting，其余原样保留，
  不拆、不合并，保持原生多问题形态。
- 所有切分的 group_id 不动，跨 split 不泄漏。
"""
from __future__ import annotations

import collections
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

SRC = r"data/typed_full"
DST = r"data/hybrid_v3"
MAX_SEQ = 4500
SAMPLE_RATE = 0.05
SEED = 3407
PT = "painting-geometry-v1"
SCHEMA = pa.schema([
    ("state", pa.string()), ("questions", pa.string()), ("gold", pa.string()),
    ("group_id", pa.string()), ("source", pa.string()), ("n_questions", pa.int64()),
])


def split_one(row: dict) -> list[dict]:
    """把一个超长 painting case 拆成多个单问题 case。

    Args:
        row: 原始 parquet 行。

    Returns:
        单问题 case 列表。
    """
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    out = []
    for name, question in questions.items():
        if name not in gold:
            continue
        out.append({
            "state": row["state"],
            "questions": json.dumps({name: question}, ensure_ascii=False),
            "gold": json.dumps({name: gold[name]}, ensure_ascii=False),
            "group_id": row["group_id"],
            "source": row["source"],
            "n_questions": 1,
        })
    return out


def stratified_sample(fragments: list[dict], rate: float, seed: int) -> list[dict]:
    """按 (case 尺寸, label) 分层抽样，稀有层保底 1 条。

    Args:
        fragments: painting 拆出的单问题片段。
        rate: 抽样比例。
        seed: 随机种子。

    Returns:
        抽样后的片段列表。
    """
    rng = random.Random(seed)
    sizes = collections.Counter()
    for r in pq.read_table(os.path.join(SRC, "train.parquet")).to_pylist():
        if r["source"] == PT:
            sizes[r["state"]] = len(json.loads(r["questions"]))
    buckets = collections.defaultdict(list)
    for f in fragments:
        label = str(list(json.loads(f["gold"]).values())[0].get("label"))
        buckets[(sizes.get(f["state"], 0), label)].append(f)
    picked = []
    for key in sorted(buckets):
        group = buckets[key]
        k = max(1, round(len(group) * rate))
        picked += rng.sample(group, min(k, len(group)))
    rng.shuffle(picked)
    target = int(len(fragments) * rate)
    if len(picked) > target:
        picked = picked[:target]
    elif len(picked) < target:
        chosen = {id(x) for x in picked}
        pool = [x for x in fragments if id(x) not in chosen]
        picked += rng.sample(pool, min(target - len(picked), len(pool)))
    return picked


def main() -> None:
    """生成 5 个切分并写 manifest。"""
    os.makedirs(DST, exist_ok=True)
    manifest = {
        "max_seq": MAX_SEQ,
        "painting_sample_rate": SAMPLE_RATE,
        "painting_seed": SEED,
        "policy": "train: 原生保留 + painting 拆后分层抽 5%; eval: 排除 painting, 原样不拆不合并",
        "splits": {},
    }
    for split in ("train", "calibration", "validation", "test", "ood"):
        rows = pq.read_table(os.path.join(SRC, f"{split}.parquet")).to_pylist()
        painting = [r for r in rows if r["source"] == PT]
        native = [r for r in rows if r["source"] != PT]
        added = []
        fragments_total = 0
        if split == "train":
            fragments = []
            for r in painting:
                fragments += split_one(r)
            fragments_total = len(fragments)
            added = stratified_sample(fragments, SAMPLE_RATE, SEED)
        out_rows = native + added
        pq.write_table(pa.Table.from_pylist(out_rows, schema=SCHEMA),
                       os.path.join(DST, f"{split}.parquet"), compression="zstd")
        nq = sum(len(json.loads(r["questions"])) for r in out_rows)
        manifest["splits"][split] = {
            "native_cases": len(native),
            "painting_cases_in_source": len(painting),
            "painting_fragments_total": fragments_total,
            "painting_fragments_kept": len(added),
            "total_cases": len(out_rows),
            "total_questions": nq,
            "multi_question_cases": sum(1 for r in out_rows
                                       if len(json.loads(r["questions"])) > 1),
            "groups": len({r["group_id"] for r in out_rows}),
            "size_mb": round(os.path.getsize(os.path.join(DST, f"{split}.parquet")) / 1e6, 3),
        }
        m = manifest["splits"][split]
        print(f"{split:12s} 原生 {m['native_cases']:6d} + painting片段 {m['painting_fragments_kept']:5d} "
              f"= {m['total_cases']:6d} case / {m['total_questions']:6d} 题, "
              f"N>1 {m['multi_question_cases']:5d}, groups {m['groups']:5d}", flush=True)
    with open(os.path.join(DST, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print("[DONE]", DST, flush=True)


if __name__ == "__main__":
    main()
