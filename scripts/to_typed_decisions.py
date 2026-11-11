"""把 Open-Jev 原始记录转换为 typed-decisions 格式。

转换后每行含 state / questions / gold 三个 JSON 字符串列，另附
<问题名>__label / __probabilities / __score / __probability_true 扁平列。

聚合规则：同一 (group_id, state) 的多个 head 合并为一个 case，
使决策头能在一次前向里同时回答该 case 的全部问题。

格式一致性由官方源码保证：请求用 jev.api.compile_request 编译，
答案用 jev.api.format_response 格式化，与 typed-decisions 同源同构。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_SRC = Path(os.environ.get("OPEN_JEV_SRC", "./Open-Jev/reproduce/source-code"))
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from jev.api import compile_request, format_response  # noqa: E402

SPLITS = ("train", "calibration", "validation", "test", "ood")


def iter_rows(path: Path) -> Any:
    """逐行读取一个切分文件，支持 .jsonl 与 .jsonl.gz。

    Args:
        path: 切分文件路径。

    Yields:
        解析后的记录字典。
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def state_fingerprint(state: Any) -> str:
    """计算状态指纹，用于把同一 case 的多个 head 归到一起。

    Args:
        state: 原始状态。

    Returns:
        十六进制指纹。
    """
    text = state if isinstance(state, str) else json.dumps(
        state, ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def split_option(option: str) -> tuple[str, str | None]:
    """把选项字符串拆成名称与描述。

    Args:
        option: 选项字符串，形如 "name: description" 或仅有名称。

    Returns:
        (名称, 描述或 None)。
    """
    if ": " in option:
        name, _, description = option.partition(": ")
        return name, description
    return option, None


def build_choice_criteria(options: list[str]) -> dict[str, str | None]:
    """为 choice 构造 criteria 映射。

    Args:
        options: 原始选项字符串列表。

    Returns:
        名称到描述的映射；名称不唯一时退化为整串作名称。
    """
    pairs = [split_option(option) for option in options]
    if len({name for name, _ in pairs}) != len(pairs):
        return {option: None for option in options}
    return {name: description for name, description in pairs}


def question_id(row: dict) -> str:
    """为一行记录生成稳定的问题名。

    Args:
        row: 原始记录。

    Returns:
        问题名，优先使用 metadata 中的 question_id。
    """
    metadata = row.get("metadata") or {}
    return str(metadata.get("question_id") or row.get("kind") or "question")


def build_case(rows: list[dict]) -> tuple[str, str, str] | None:
    """把同 case 的多行记录编译成一个 typed-decisions 行。

    Args:
        rows: 属于同一 case 的记录列表。

    Returns:
        (state_json, questions_json, gold_json) 三元组；无法编译时返回 None。
    """
    first = rows[0]
    state = first["state"]
    questions: dict[str, dict] = {}
    records: list[dict] = []
    probabilities: list[list[float]] = []

    for row in rows:
        kind = row["kind"]
        options = row["options"]
        target = [float(value) for value in row["target"]]
        if kind == "choice":
            definition = {
                "type": "choice",
                "instructions": row["question"],
                "criteria": build_choice_criteria(options),
            }
        elif kind == "score":
            definition = {
                "type": "score",
                "instructions": row["question"],
                "criteria": list(options),
            }
        else:
            definition = {"type": "noul", "instructions": row["question"]}

        compiled = compile_request(state, {row["_qid"]: definition})
        record = compiled[0]
        keys = record["answer_keys"]
        if len(keys) != len(target):
            return None

        if kind == "choice":
            position = {option: index for index, option in enumerate(options)}
            values = []
            for index in range(len(keys)):
                option = record["options"][index]
                values.append(target[position[option]] if option in position else 0.0)
        else:
            values = list(target)

        records.append(record)
        probabilities.append(values)
        questions[record["id"]] = definition

    response = format_response(records, probabilities)
    gold = response["answers"]
    for answer in gold.values():
        if answer["type"] == "noul":
            truth = float(answer["noul"])
            answer["label"] = "true" if truth >= 0.5 else "false"
            answer["probabilities"] = {"false": 1.0 - truth, "true": truth}
        elif answer["type"] == "choice":
            answer["label"] = answer["choice"]
        else:
            distribution = answer["probabilities"]
            answer["label"] = max(distribution, key=distribution.get)
    state_json = state if isinstance(state, str) else json.dumps(
        state, ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    return (
        state_json,
        json.dumps(questions, ensure_ascii=False),
        json.dumps(gold, ensure_ascii=False),
    )


def flatten(gold: dict) -> dict:
    """把 gold 展开成 typed-decisions 的扁平便捷列。

    Args:
        gold: 问题名到答案的映射。

    Returns:
        扁平列字典。
    """
    flat: dict[str, Any] = {}
    for name, answer in gold.items():
        if answer["type"] == "noul":
            flat[f"{name}__probability_true"] = answer["noul"]
            continue
        flat[f"{name}__label"] = answer.get("choice", answer.get("score"))
        flat[f"{name}__probabilities"] = json.dumps(answer["probabilities"], ensure_ascii=False)
        if answer["type"] == "score":
            flat[f"{name}__score"] = answer["score"]
    return flat


def convert(config: str, split: str, raw_root: Path) -> tuple[list[dict], dict]:
    """转换一个 config 的一个切分。

    Args:
        config: config 名。
        split: 切分名。
        raw_root: 原始数据根目录。

    Returns:
        (typed-decisions 行列表, 统计信息)。
    """
    path = raw_root / config / f"{split}.jsonl.gz"
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen: dict[tuple[str, str], dict[str, tuple[str, tuple[float, ...]]]] = defaultdict(dict)
    dropped = 0
    for row in iter_rows(path):
        key = (row["group_id"], state_fingerprint(row["state"]))
        name = question_id(row)
        signature = hashlib.sha1(
            json.dumps([row["kind"], row["question"], row["options"]],
                       ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:8]
        target = tuple(float(value) for value in row["target"])
        assigned = seen[key]
        if name in assigned:
            previous_signature, previous_target = assigned[name]
            if previous_signature == signature and previous_target == target:
                dropped += 1
                continue
            name = f"{name}~{signature}"
            if name in assigned:
                previous_signature, previous_target = assigned[name]
                if previous_signature == signature and previous_target == target:
                    dropped += 1
                    continue
                suffix = 2
                while f"{name}~{suffix}" in assigned:
                    suffix += 1
                name = f"{name}~{suffix}"
        assigned[name] = (signature, target)
        buckets[key].append({**row, "_qid": name})

    out: list[dict] = []
    failed = 0
    for (group_id, _), rows in buckets.items():
        built = build_case(rows)
        if built is None:
            failed += 1
            continue
        state_json, questions_json, gold_json = built
        gold = json.loads(gold_json)
        record = {
            "state": state_json,
            "questions": questions_json,
            "gold": gold_json,
            "group_id": group_id,
            "source": rows[0]["source"],
            "n_questions": len(gold),
        }
        record.update(flatten(gold))
        out.append(record)

    stats = {
        "cases": len(out),
        "rows_in": sum(len(rows) for rows in buckets.values()),
        "dropped_conflicting": dropped,
        "failed_compile": failed,
        "questions_per_case": dict(Counter(record["n_questions"] for record in out)),
    }
    return out, stats


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="config 名")
    parser.add_argument("--splits", default=",".join(SPLITS), help="逗号分隔的切分名")
    parser.add_argument("--raw-root", default=os.environ.get("OPEN_JEV_RAW", "./Open-Jev/raw"), help="原始数据根目录")
    parser.add_argument("--out", default=r"data/typed_full_src", help="输出目录")
    parser.add_argument("--parquet", action="store_true", help="额外输出 parquet")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out) / args.config
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        source = raw_root / args.config / f"{split}.jsonl.gz"
        if not source.exists():
            continue
        rows, stats = convert(args.config, split, raw_root)
        path = out_root / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.parquet:
            import pandas as pd

            pd.DataFrame(rows).to_parquet(out_root / f"{split}.parquet", index=False)
        manifest[split] = stats
        print(
            f"{args.config:42s} {split:12s} cases={stats['cases']:7d} "
            f"rows_in={stats['rows_in']:7d} dropped={stats['dropped_conflicting']} "
            f"failed={stats['failed_compile']} q/case={stats['questions_per_case']}"
        )

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"out -> {out_root}")


if __name__ == "__main__":
    main()
