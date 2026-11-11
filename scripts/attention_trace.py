"""抓真实 attention 权重表 + 统计模型各部分的 dtype。"""
import collections
import json

import torch
import torch.nn.functional as F
from unsloth import FastDecisionModel
from unsloth.models.clef import encode_record, collate_records

ADAPTER = r"model/adapter_v3"
OUT = r"results/attention_trace.json"
STATE = "客户反馈：订单被重复扣款两次，已经等了三天，要求立刻退款，否则就投诉。"
QUESTIONS = {
    "route": {"type":"choice","instructions":"这个工单应该交给哪个团队处理？",
              "criteria":{"billing":"账单、发票、退款相关","account":"账号登录、资料修改相关",
                          "technical":"程序缺陷、接口集成相关","other":"以上都不是"}},
    "urgency": {"type":"score","instructions":"这个工单的紧急程度如何？",
                "criteria":["可以正常排队等待","需要优先处理","需要立即人工介入"]},
    "refund_requested": {"type":"noul","instructions":"客户是否明确提出了退款请求？"},
}

model, tok = FastDecisionModel.from_pretrained(
    model_name=ADAPTER, max_seq_length=1024, load_in_4bit=True)
model.eval()
hf = getattr(tok, "tokenizer", tok)
processor = tok if hasattr(tok, "tokenizer") else None
enc = encode_record(hf, {"state":STATE,"questions":QUESTIONS,"images":None},
                    max_length=4096, processor=processor)
ids = list(enc.input_ids)

# ---- 1. dtype 分布 ----
by_dtype = collections.Counter()
by_part = collections.Counter()
for n, p in model.named_parameters():
    key = str(p.dtype)
    by_dtype[(key, p.numel())] += 1
    by_part[(key, p.dtype)] += p.numel()
dtype_params = collections.Counter()
for n, p in model.named_parameters():
    dtype_params[str(p.dtype)] += p.numel()
R = {"dtype_params": dict(dtype_params),
     "dtype_tensors": {k: sum(1 for _, p in model.named_parameters() if str(p.dtype) == k)
                       for k in dtype_params}}
print("[1] dtype:", R["dtype_params"], flush=True)

head = None
for _, m in model.named_modules():
    if hasattr(m, "type_embedding"):
        head = m
        break
R["head_dtype"] = str(next(head.parameters()).dtype)
R["scale_params"] = {
    "prior_logit_scale": float(head.prior_logit_scale),
    "joint_logit_scale": float(head.joint_logit_scale),
    "residual_gate": float(head.residual_gate),
    "sigmoid_gate": float(torch.sigmoid(head.residual_gate)),
    "exp_prior_scale": float(head.prior_logit_scale.exp()),
    "exp_joint_scale": float(head.joint_logit_scale.exp()),
}

# ---- 2. hook attention 拿 Q/K/V，手算权重 ----
cap = {}
handles = []
def mk(name):
    def hook(_m, inp, _out):
        cap[name] = [t.detach().float().cpu() for t in inp[:3] if isinstance(t, torch.Tensor)]
    return hook
handles.append(head.evidence_layers[0].attention.register_forward_hook(mk("route_attn")))
handles.append(head.layers[0].self_attn.register_forward_hook(mk("field_self_attn")))

device = next(model.parameters()).device
batch = collate_records([enc], hf.pad_token_id, device)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
          records=batch["records"], **batch["media"])
for h in handles:
    h.remove()

def attn_weights(mha, q, k, n_q, n_k):
    """按 8 头手算 scaled dot-product attention 权重。"""
    W = mha.in_proj_weight.detach().float().cpu()
    b = mha.in_proj_bias.detach().float().cpu() if mha.in_proj_bias is not None else None
    wq, wk, _ = W.split(W.shape[0] // 3, dim=0)
    bq, bk, _ = b.split(W.shape[0] // 3, dim=0) if b is not None else (None, None, None)
    Q = F.linear(q, wq, bq)
    K = F.linear(k, wk, bk)
    B, Nq, D = Q.shape
    Nk = K.shape[1]
    H = mha.num_heads
    dh = D // H
    Q = Q.view(B, Nq, H, dh).transpose(1, 2)
    K = K.view(B, Nk, H, dh).transpose(1, 2)
    s = (Q @ K.transpose(-2, -1)) / (dh ** 0.5)
    return s.softmax(-1)

# 选项 -> 全文：9 x 370
q, k, _v = cap["route_attn"]
k = k[:, :len(ids)]
w = attn_weights(head.evidence_layers[0].attention, q, k, q.shape[1], k.shape[1])
avg = w[0].mean(0)  # 8 头平均 -> [9, 370]
R["option_to_memory"] = {
    "shape_heads": list(w.shape), "shape_avg": list(avg.shape),
    "rows": [],
}
print("[2] option->memory attention:", list(w.shape), "avg", list(avg.shape), flush=True)
for i, opt in enumerate(["account","billing","other","technical","0","1","2","true","false"]):
    row = avg[i]
    top = torch.topk(row, 8)
    R["option_to_memory"]["rows"].append({
        "option": opt,
        "sum": round(float(row.sum()), 4),
        "top8": [{"pos": int(p), "w": round(float(v), 4),
                  "token": hf.decode([ids[p]])} for v, p in zip(top.values, top.indices)],
    })
    print(f"    {opt:10s} top1 pos={int(top.indices[0])} "
          f"w={float(top.values[0]):.4f} token={hf.decode([ids[int(top.indices[0])]])!r}", flush=True)

# 3 个 field 之间：3 x 3
q2, k2, _ = cap["field_self_attn"]
w2 = attn_weights(head.layers[0].self_attn, q2, k2, q2.shape[1], k2.shape[1])
R["field_self_attention"] = {
    "shape": list(w2.shape),
    "avg_3x3": [[round(float(x), 4) for x in row] for row in w2[0].mean(0)],
}
print("[3] field self-attn 3x3:", R["field_self_attention"]["avg_3x3"], flush=True)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print("[DONE]", OUT, flush=True)
