"""在 typed-decisions 数据上微调 Unsloth 决策模型。

用法（魔搭 Notebook / 任意 GPU 机器）：
    python train_decision.py \
        --data /mnt/workspace/data/release-v2 \
        --model /mnt/workspace/models/Qwen3.5-4B \
        --out runs/rv2 --steps 200 --max-seq 1024 --eval-every 50

流程：build_dataset -> DecisionTrainer -> (每 N 步中途评测) -> calibrate -> evaluate -> save
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrainerCallback, TrainingArguments
from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported

SPLITS = ("train", "calibration", "validation", "test", "ood")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="含 {split}.parquet 的目录")
    parser.add_argument("--model", default="unsloth/Qwen3.5-4B", help="底座模型或本地路径")
    parser.add_argument("--out", default="runs/decision", help="输出目录")
    parser.add_argument("--max-seq", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA 秩")
    parser.add_argument("--steps", type=int, default=0, help="最大步数，0 表示按 epoch")
    parser.add_argument("--epochs", type=int, default=2, help="epoch 数（steps=0 时生效）")
    parser.add_argument("--bs", type=int, default=8, help="每设备 batch size")
    parser.add_argument("--accum", type=int, default=4, help="梯度累积")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--warmup", type=int, default=10, help="warmup 步数")
    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--max-rows", type=int, default=0, help="训练集行数上限，0 为全部")
    parser.add_argument("--eval-rows", type=int, default=0, help="评测集行数上限，0 为全部")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="每 N 步用 validation 做一次监控评测，0 表示只在训练结束后")
    parser.add_argument("--save-every", type=int, default=0,
                        help="每 N 步保存一次 adapter，0 表示只在训练结束后")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="跳过训练前基线评测（评测较慢时可用）")
    parser.add_argument("--resume", action="store_true",
                        help="从 out/checkpoints 里步数最大的 checkpoint 恢复，跳过重新训练")
    return parser.parse_args()


def load_splits(data_dir: Path) -> dict:
    """从目录加载各切分的 parquet。

    Args:
        data_dir: 含 {split}.parquet 的目录。

    Returns:
        切分名到 Dataset 的映射。
    """
    files = {}
    for split in SPLITS:
        path = data_dir / f"{split}.parquet"
        if path.exists():
            files[split] = str(path)
    if "train" not in files:
        raise FileNotFoundError(f"{data_dir} 下没有 train.parquet")
    print(f"[data] 找到切分: {sorted(files)}")
    return load_dataset("parquet", data_files=files)


def build_items(model, tokenizer, dataset, split: str, limit: int = 0):
    """把一个切分编译成决策模型的训练项。

    Args:
        model: 决策模型。
        tokenizer: 分词器。
        dataset: 原始数据集。
        split: 切分名，用于日志。
        limit: 行数上限，0 为全部。

    Returns:
        (items, report) 二元组。
    """
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
    print(f"[data] {split}: {len(items)} items, skipped {report['skipped']}/{report['total']}")
    reasons = report.get("reason")
    if reasons:
        counts = {}
        for value in (reasons.values() if isinstance(reasons, dict) else []):
            counts[value] = counts.get(value, 0) + 1
        for reason, count in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
            print(f"        skip 原因 {count:6d}  {reason}")
    return items, report


class MidEvalCallback(TrainerCallback):
    """训练中途按固定步数做一次评测，并把指标追加写入 jsonl。"""

    def __init__(self, every, model, tokenizer, items, log_path, split_name):
        """初始化回调。

        Args:
            every: 每多少步评测一次，0 表示关闭。
            model: 决策模型。
            tokenizer: 分词器。
            items: 评测样本。
            log_path: 指标日志路径。
            split_name: 评测切分名，写入日志用。
        """
        self.every = every
        self.model = model
        self.tokenizer = tokenizer
        self.items = items
        self.log_path = log_path
        self.split_name = split_name
        self._t0 = time.time()

    def _write(self, record):
        """把一条指标记录追加写入 jsonl。

        Args:
            record: 指标字典。
        """
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_step_end(self, args, state, control, **kwargs):
        """每个优化步结束时判断是否评测。

        Args:
            args: 训练参数。
            state: Trainer 状态。
            control: Trainer 控制流。
            **kwargs: 其余回调参数。

        Returns:
            恒为 None。
        """
        if not self.every or state.global_step % self.every != 0:
            return None
        was_training = self.model.training
        self.model.eval()
        try:
            report = FastDecisionModel.evaluate(self.model, self.tokenizer, self.items)
        except Exception as exc:  # 评测失败不应中断训练
            print(f"[eval] step {state.global_step} 评测失败: {exc}")
            return None
        finally:
            if was_training:
                self.model.train()
        record = {
            "step": state.global_step,
            "split": self.split_name,
            "elapsed_s": round(time.time() - self._t0, 1),
            **{k: round(float(v), 6) for k, v in report.items()},
        }
        self._write(record)
        print(f"[eval] step {state.global_step:5d}  {self.split_name}(未标定): "
              f"acc={report['accuracy']:.4f}  ece={report['ece']:.4f}  "
              f"loss={report['loss']:.4f}  record_acc={report['record_accuracy']:.4f}")
        return None


def _fmt(report: dict) -> str:
    """把评测报告格式化成一行。

    Args:
        report: evaluate / calibrate 返回的指标字典，值可能是标量或列表。

    Returns:
        可读的一行文本。
    """
    parts = []
    for key, value in report.items():
        if isinstance(value, bool):
            parts.append(f"{key}={value}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}={float(value):.4f}")
        elif isinstance(value, (list, tuple)):
            parts.append(f"{key}={[round(float(x), 4) for x in value]}")
        else:
            parts.append(f"{key}={value}")
    return "  ".join(parts)


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_log.jsonl"
    started = time.time()

    print(f"[model] 加载 {args.model}")
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
    )
    model = FastDecisionModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] 可训练参数 {trainable/1e6:.2f}M / 总计 {total/1e9:.2f}B")

    datasets = load_splits(Path(args.data))

    resume_dir = None
    if args.resume:
        ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                       key=lambda path: int(path.name.split("-")[-1]))
        if ckpts:
            resume_dir = str(ckpts[-1])
            print(f"[resume] 从 {resume_dir} 恢复（跳过训练，直接做标定与评测）")
        else:
            print("[resume] 找不到 checkpoint，改为从头训练")

    # 恢复训练时不必重建完整训练集，只要有样本让 Trainer 能构造即可。
    train_limit = args.max_rows
    if resume_dir and not args.max_rows:
        train_limit = 64
        print(f"[resume] 训练集只加载 {train_limit} 条（省去编译时间）")

    train_items, _ = build_items(model, tokenizer, datasets["train"], "train", train_limit)
    calib_items, _ = build_items(
        model, tokenizer, datasets["calibration"], "calibration", args.eval_rows)
    test_items, _ = build_items(
        model, tokenizer, datasets["test"], "test", args.eval_rows)
    ood_items = None
    if "ood" in datasets:
        ood_items, _ = build_items(
            model, tokenizer, datasets["ood"], "ood", args.eval_rows)

    # validation 负责监控收敛与选 checkpoint；缺失时退回 calibration。
    if "validation" in datasets:
        valid_items, _ = build_items(
            model, tokenizer, datasets["validation"], "validation", args.eval_rows)
    else:
        valid_items = calib_items

    if not args.skip_baseline:
        print("[eval] 训练前基线（validation）")
        report = FastDecisionModel.evaluate(model, tokenizer, valid_items)
        print(f"[eval] baseline validation: {_fmt(report)}")

    train_args = TrainingArguments(
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        warmup_steps=args.warmup,
        num_train_epochs=args.epochs,
        max_steps=args.steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        eval_strategy="no",
        logging_steps=10,
        save_strategy="steps" if args.save_every else "no",
        save_steps=args.save_every or None,
        save_total_limit=3,
        output_dir=str(out_dir / "checkpoints"),
        report_to="none",
        seed=args.seed,
    )
    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_items,
        eval_dataset=valid_items,
        args=train_args,
    )
    callback = MidEvalCallback(args.eval_every, model, tokenizer, valid_items,
                               str(metrics_path), "validation")
    trainer.add_callback(callback)

    stats = trainer.train(resume_from_checkpoint=resume_dir)
    print(f"[train] 用时 {stats.metrics.get('train_runtime', 0)/60:.1f} 分钟")

    print("[calib] 在 calibration 上拟合温度（只用这一次）")
    calib_report = FastDecisionModel.calibrate(model, tokenizer, calib_items)
    print(f"[calib] {_fmt(calib_report) if isinstance(calib_report, dict) else calib_report}")

    print("[eval] 最终 test")
    test_report = FastDecisionModel.evaluate(model, tokenizer, test_items)
    print(f"[eval] test: {_fmt(test_report)}")
    ood_report = None
    if ood_items:
        print("[eval] 最终 ood")
        ood_report = FastDecisionModel.evaluate(model, tokenizer, ood_items)
        print(f"[eval] ood: {_fmt(ood_report)}")

    model.save_pretrained(str(out_dir / "adapter"))
    FastDecisionModel.for_inference(model)

    final = {
        "config": vars(args),
        "train_runtime_s": stats.metrics.get("train_runtime"),
        "calibration": calib_report if isinstance(calib_report, dict) else str(calib_report),
        "validation": _fmt(FastDecisionModel.evaluate(model, tokenizer, valid_items)),
        "test": test_report,
        "ood": ood_report,
        "total_minutes": round((time.time() - started) / 60, 1),
    }
    (out_dir / "run_report.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] 总用时 {final['total_minutes']} 分钟 -> {out_dir}")


if __name__ == "__main__":
    main()
