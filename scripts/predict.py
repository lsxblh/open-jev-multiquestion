"""用微调好的决策模型做推理。

两种模式：

1) 单条决策
   python predict.py --adapter ADAPTER --model BASE --demo

2) 信息流批量打分并排序（画像条件化）
   python predict.py --adapter ADAPTER --model BASE \
       --stream stream_demo.json
"""
from __future__ import annotations

import argparse
import json
import sys

from unsloth import FastDecisionModel


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="训练好的 adapter 目录")
    parser.add_argument("--model", default="",
                        help="可选：底座路径说明；实际以 adapter_config.json 的 base_model_name_or_path 为准")
    parser.add_argument("--demo", action="store_true", help="运行内置单条示例")
    parser.add_argument("--stream", help="信息流 JSON 文件路径")
    parser.add_argument("--max-seq", type=int, default=4500,
                        help="最大序列长度（V3 = 4500；小于此值会截断长序列）")
    return parser.parse_args()


def bar(prob: float, width: int = 26) -> str:
    """把概率画成条形图。

    Args:
        prob: 概率值。
        width: 条形宽度。

    Returns:
        条形字符串。
    """
    filled = int(round(prob * width))
    return "█" * filled + "·" * (width - filled)


def show_answers(answers: dict, questions: dict) -> None:
    """打印一组决策结果。

    Args:
        answers: 模型返回的答案字典。
        questions: 原始问题定义。
    """
    for name, answer in answers.items():
        kind = answer["type"]
        question = questions.get(name, {})
        print(f"\n[{kind}] {name}")
        print(f"    {question.get('instructions', '')}")
        for key, prob in sorted(answer["probabilities"].items(), key=lambda kv: -kv[1]):
            print(f"      {key:<16} {prob:.3f}  {bar(prob)}")
        if kind == "choice":
            print(f"    -> 选择：{answer['answer']}")
            print(f"    -> confidence = {answer['confidence']:.3f}")
        elif kind == "noul":
            print(f"    -> P(为真) = {answer['noul']:.3f}")
            print(f"    -> confidence = {max(answer['probabilities'].values()):.3f}")
        else:
            best = max(answer["probabilities"], key=answer["probabilities"].get)
            print(f"    -> 期望档位 {answer['score']:.2f}（最可能 {best}）")
            print(f"    -> confidence = {answer['probabilities'][best]:.3f}")


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    print(f"[load] {args.adapter} + {args.model}")
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
    )

    if args.stream:
        with open(args.stream, encoding="utf-8") as fh:
            payload = json.load(fh)
        profile = payload.get("profile", "")
        questions = payload["questions"]
        items = payload["items"]
        print(f"\n[stream] 画像：{profile}")
        print(f"[stream] {len(items)} 条信息")

        scored = []
        for item in items:
            state = item["content"]
            if profile:
                state = {"profile_evidence": profile.split("；"), "content": state}
            answers = FastDecisionModel.predict(model, tokenizer, state, questions)
            key = "relevance" if "relevance" in answers else next(iter(answers))
            answer = answers[key]
            prob = (answer["probabilities"].get(answer["answer"])
                    if answer["type"] == "choice" else answer.get("noul", answer.get("score", 0)))
            scored.append((float(prob or 0.0), item.get("name", ""), item["content"], answers))

        scored.sort(key=lambda row: -row[0])
        print("\n" + "=" * 78)
        print("按相关度排序")
        print("=" * 78)
        for prob, name, content, answers in scored:
            print(f"  {prob:.3f}  {bar(prob)}  {name}")
            print(f"         {content[:90]}")
        print("\n" + "=" * 78)
        print("第一条的完整决策")
        print("=" * 78)
        show_answers(scored[0][3], questions)
        return

    record = {
        "state": "客户反馈：订单被重复扣款两次，已经等了三天，要求立刻退款，否则就投诉。",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "这个工单应该交给哪个团队处理？",
                "criteria": {
                    "billing": "账单、发票、退款相关",
                    "account": "账号登录、资料修改相关",
                    "technical": "程序缺陷、接口集成相关",
                    "other": "以上都不是",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "这个工单的紧急程度如何？",
                "criteria": ["可以正常排队等待", "需要优先处理", "需要立即人工介入"],
            },
            "refund_requested": {
                "type": "noul",
                "instructions": "客户是否明确提出了退款请求？",
            },
        },
    }
    answers = FastDecisionModel.predict(model, tokenizer, record["state"], record["questions"])
    show_answers(answers, record["questions"])


if __name__ == "__main__":
    main()
