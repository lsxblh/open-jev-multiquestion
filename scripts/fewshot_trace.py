"""一次完整的 few-shot 走查：dump 一条工单从输入到输出的每一站。"""
import json
import os
import sys

import torch

from unsloth import FastDecisionModel
sys.path.insert(0, r".")
from unsloth.models.clef import collate_records, encode_record, systemone_answer
from unsloth.models.decision import _served_temperatures
from unsloth.models._decision_common import QUESTION_TYPES  # ("choice","score","noul")

ADAPTER = r"model/adapter_v3"
OUT = r"results/fewshot_trace.json"

STATE = "客户反馈：订单被重复扣款两次，已经等了三天，要求立刻退款，否则就投诉。"
QUESTIONS = {
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
}

result = {"state": STATE, "questions": QUESTIONS}
print("[1] 加载模型 ...", flush=True)
model, tok = FastDecisionModel.from_pretrained(
    model_name=ADAPTER, max_seq_length=1024, load_in_4bit=True)
model.eval()
processor = tok if hasattr(tok, "tokenizer") else None
hf_tok = getattr(tok, "tokenizer", tok)

print("[2] encode_record ...", flush=True)
encoded = encode_record(
    hf_tok,
    {"state": STATE, "questions": QUESTIONS, "images": None},
    max_length=max(int(model.decision_config.get("max_len", 1024)), 4096),
    processor=processor,
)
result["input_tokens"] = len(encoded.input_ids)
result["decoded_input"] = hf_tok.decode(encoded.input_ids)
result["encoded_questions"] = [
    {
        "question_id": q.question_id,
        "question_type": q.question_type,
        "question_span": list(q.question_span),
        "option_ids": list(q.option_ids),
        "option_spans": [list(s) for s in q.option_spans],
    }
    for q in encoded.questions
]
print(f"    input_tokens = {len(encoded.input_ids)}", flush=True)
for q in encoded.questions:
    print(f"    {q.question_id:18s} type={q.question_type}  options={q.option_ids}", flush=True)

# 定位 head（含 type_embedding 的模块）
head = None
head_name = None
for name, mod in model.named_modules():
    if hasattr(mod, "type_embedding"):
        head, head_name = mod, name
        break
print(f"[3] head = {head_name}", flush=True)
result["head_name"] = head_name
result["head_config"] = {
    "hidden_size": head.hidden_norm.normalized_shape[0],
    "width": head.type_embedding.weight.shape[1],
    "n_types": head.type_embedding.weight.shape[0],
    "routing_layers": len(head.evidence_layers),
    "layers": len(head.layers),
    "option_projections": [n for n, _ in head.named_children() if "projection" in n],
}

# 挂 hook 记录关键模块的输出 shape
trace = []
targets = [
    "hidden_norm", "memory_projection", "question_projection", "global_projection",
    "option_context_projection", "option_lexical_projection", "option_question_projection",
    "type_embedding", "option_summary_norm", "field_norm", "option_norm", "residual_scorer",
    "evidence_layers.0", "evidence_layers.1", "layers.0", "layers.3",
]
handles = []
for name in targets:
    mod = dict(head.named_modules()).get(name)
    if mod is None:
        continue

    def make_hook(n):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                trace.append({"module": n, "shape": list(t.shape),
                              "min": round(float(t.min()), 4),
                              "max": round(float(t.max()), 4),
                              "mean": round(float(t.mean()), 4)})
        return hook
    handles.append(mod.register_forward_hook(make_hook(name)))

print("[4] forward ...", flush=True)
device = next(model.parameters()).device
batch = collate_records([encoded], hf_tok.pad_token_id, device)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits, _ = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        records=batch["records"],
        **batch["media"],
    )
for h in handles:
    h.remove()
result["module_trace"] = trace

rows = [row[:len(q.option_ids)] for q, row in zip(encoded.questions, logits.float().cpu())]
qtypes = [{"qtype": QUESTION_TYPES.index(QUESTIONS[q.question_id]["type"])}
          for q in encoded.questions]
scales = _served_temperatures(model.decision_config, rows, qtypes)
result["decision_config"] = {
    k: v for k, v in model.decision_config.items()
    if k in ("layout", "max_len", "temperature", "folded_temperature", "fine_tuned")
}

answers = {}
result["per_question"] = []
for q, row, scale in zip(encoded.questions, rows, scales):
    probs_t = (row / scale).softmax(-1)
    probs = {k: round(float(v), 6) for k, v in zip(q.option_ids, probs_t.tolist())}
    asked = QUESTIONS[q.question_id]
    ans = systemone_answer(asked, probs)
    answers[q.question_id] = ans
    result["per_question"].append({
        "question_id": q.question_id,
        "type": asked["type"],
        "option_ids": list(q.option_ids),
        "raw_logits": [round(float(v), 4) for v in row.tolist()],
        "temperature": round(float(scale), 4),
        "probabilities": probs,
        "answer": ans,
    })
    print(f"    {q.question_id:18s} logits={[round(float(v),3) for v in row.tolist()]} "
          f"T={float(scale):.3f} probs={probs}", flush=True)

result["answers"] = answers
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
print(f"[DONE] -> {OUT}", flush=True)
