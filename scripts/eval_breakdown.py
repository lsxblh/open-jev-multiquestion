"""分语料 / 分题型评测：看模型在哪类任务上强、在哪类任务上弱。

用法：
    python eval_breakdown.py \
        --adapter /mnt/workspace/runs/rv2/adapter \
        --data /mnt/workspace/data/release-v2 \
        --split test
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import pyarrow.parquet as pq
from datasets import Dataset
from unsloth import FastDecisionModel


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="训练好的 adapter 目录")
    parser.add_argument("--data", required=True, help="含 {split}.parquet 的目录")
    parser.add_argument("--split", default="test", help="评测切分")
    parser.add_argument("--max-seq", type=int, default=4500,
                        help="最大序列长度（V3 = 4500；小于此值会截断长序列）")
    parser.add_argument("--limit", type=int, default=0, help="每组最多取多少条，0 为全部")
    parser.add_argument("--out", default="", help="可选：把分组结果写成 JSON")
    return parser.parse_args()


def load_model(adapter: str, max_seq: int):
    """加载训练好的决策模型。

    Args:
        adapter: adapter 目录。
        max_seq: 最大序列长度。

    Returns:
        (model, tokenizer) 二元组。
    """
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=adapter,
        max_seq_length=max_seq,
        load_in_4bit=True,
    )
    return model, tokenizer


def evaluate_group(model, tokenizer, rows, name, limit):
    """对一组同质样本评测并打印一行。

    Args:
        model: 决策模型。
        tokenizer: 分词器。
        rows: 样本字典列表。
        name: 组名。
        limit: 每组上限，0 为全部。

    Returns:
        (accuracy, ece, loss, 样本数) 四元组，失败时为 None。
    """
    if limit:
        rows = rows[:limit]
    if not rows:
        return None
    try:
        items, report = FastDecisionModel.build_dataset(Dataset.from_list(rows), tokenizer, model)
        if not items:
            return None
        result = FastDecisionModel.evaluate(model, tokenizer, items)
    except Exception as exc:
        print(f"{name:46s} 评测失败: {str(exc)[:80]}")
        return None
    n = len(items)
    print(f"{name:46s} n={n:6d}  acc={result['accuracy']:.4f}  "
          f"ece={result['ece']:.4f}  loss={result['loss']:.4f}")
    return result["accuracy"], result["ece"], result["loss"], n


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    model, tokenizer = load_model(args.adapter, args.max_seq)

    path = f"{args.data}/{args.split}.parquet"
    rows = pq.read_table(path).to_pylist()
    print(f"[data] {args.split}: {len(rows)} 行")

    print("\n" + "=" * 90)
    print("按 source 分组")
    print("=" * 90)
    by_src = defaultdict(list)
    for row in rows:
        by_src[row["source"]].append(row)
    totals = []
    for source in sorted(by_src):
        outcome = evaluate_group(model, tokenizer, by_src[source], source, args.limit)
        if outcome:
            totals.append((source, outcome))

    print("\n" + "=" * 90)
    print("按题型分组（每个 case 只含一个问题，所以题��=case）")
    print("=" * 90)
    by_kind = defaultdict(list)
    for row in rows:
        questions = json.loads(row["questions"])
        for name, question in questions.items():
            kind = question.get("type", "unknown")
            by_kind[kind].append({
                "state": row["state"],
                "questions": json.dumps({name: question}, ensure_ascii=False),
                "gold": row["gold"],
                "group_id": row["group_id"],
                "source": row["source"],
                "n_questions": 1,
            })
    for kind in sorted(by_kind):
        evaluate_group(model, tokenizer, by_kind[kind], f"kind={kind}", args.limit)

    print("\n" + "=" * 90)
    print("最难/最易 source 排序（按 accuracy）")
    print("=" * 90)
    ranked = sorted(totals, key=lambda item: item[1][0])
    for source, (accuracy, ece, loss, n) in ranked:
        print(f"  {accuracy:.4f}  ece={ece:.4f}  n={n:6d}  {source}")


if __name__ == "__main__":
    main()
