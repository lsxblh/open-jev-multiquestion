"""把中间张量的"具体长相"dump 出来：真实数字，不是示意。"""
import json

import torch
from unsloth import FastDecisionModel
from unsloth.models.clef import encode_record, collate_records

ADAPTER = r"model/adapter_v3"
OUT = r"results/tensor_shapes.json"
STATE = "客户反馈：订单被重复扣款两次，已经等了三天，要求立刻退款，否则就投诉。"
QUESTIONS = {
    "route": {"type": "choice", "instructions": "这个工单应该交给哪个团队处理？",
              "criteria": {"billing": "账单、发票、退款相关", "account": "账号登录、资料修改相关",
                           "technical": "程序缺陷、接口集成相关", "other": "以上都不是"}},
    "urgency": {"type": "score", "instructions": "这个工单的紧急程度如何？",
                "criteria": ["可以正常排队等待", "需要优先处理", "需要立即人工介入"]},
    "refund_requested": {"type": "noul", "instructions": "客户是否明确提出了退款请求？"},
}

def head8(t, n=8):
    """取张量前 n 个数。"""
    return [round(float(x), 5) for x in t.detach().float().cpu().flatten()[:n]]

model, tok = FastDecisionModel.from_pretrained(
    model_name=ADAPTER, max_seq_length=1024, load_in_4bit=True)
model.eval()
hf = getattr(tok, "tokenizer", tok)
processor = tok if hasattr(tok, "tokenizer") else None
enc = encode_record(hf, {"state": STATE, "questions": QUESTIONS, "images": None},
                    max_length=4096, processor=processor)
R = {}
ids = list(enc.input_ids)

# ---- 1. token 到底是什么 ----
R["tokens"] = {
    "count": len(ids),
    "first_12": [{"i": i, "id": ids[i], "text": hf.decode([ids[i]])} for i in range(12)],
}
billing = enc.questions[0].option_spans[1]  # OPTION 2 = billing
R["billing_option"] = {
    "span": list(billing),
    "text": hf.decode(ids[billing[0]:billing[1]]),
    "n_tokens": billing[1] - billing[0],
    "token_list": [hf.decode([ids[i]]) for i in range(billing[0], billing[1])],
}
R["state_span_hint"] = "STATE: 之后到 SCHEMA FIELDS: 之前"

# ---- 2. 6 个投影矩阵的真实形状与权重 ----
head = None
for _, m in model.named_modules():
    if hasattr(m, "type_embedding"):
        head = m
        break
R["projection_matrices"] = {}
for name, mod in head.named_children():
    if "projection" in name and hasattr(mod, "weight"):
        w = mod.weight
        R["projection_matrices"][name] = {
            "shape": list(w.shape),
            "n_params": w.numel(),
            "first_8_weights": head8(w),
        }
R["type_embedding"] = {
    "shape": list(head.type_embedding.weight.shape),
    "noul_first8": head8(head.type_embedding.weight[0]),
    "choice_first8": head8(head.type_embedding.weight[1]),
    "score_first8": head8(head.type_embedding.weight[2]),
}

# ---- 3. hook 抓真实中间向量 ----
captured = {}
modules = dict(head.named_modules())
targets = ["option_context_projection", "option_lexical_projection",
           "option_question_projection", "question_projection",
           "evidence_layers.0", "evidence_layers.1", "layers.0", "layers.3",
           "residual_scorer"]
handles = []
for n in targets:
    m = modules.get(n)
    if m is None:
        continue
    def mk(name):
        def hook(_m, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                captured[name] = {
                    "shape": list(t.shape),
                    "first8": head8(t),
                    "billing_pos_first8": head8(t[0, 1]) if t.dim() >= 2 and t.shape[1] > 1 else None,
                }
        return hook
    handles.append(m.register_forward_hook(mk(n)))

# 顺带抓 encoder 输出
enc_out = {}
for n, m in model.named_modules():
    if n == "encoder":
        def hook_enc(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor) and t.dim() == 3:
                enc_out["hidden"] = {
                    "shape": list(t.shape),
                    "token0_first8": head8(t[0, 0]),
                    "billing_span_pool_first8": head8(t[0, billing[0]:billing[1]].mean(0)),
                }
        handles.append(m.register_forward_hook(hook_enc))
        break

device = next(model.parameters()).device
batch = collate_records([enc], hf.pad_token_id, device)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
          records=batch["records"], **batch["media"])
for h in handles:
    h.remove()

R["encoder_output"] = enc_out.get("hidden")
R["head_activations"] = captured

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print("[DONE]", OUT, flush=True)
