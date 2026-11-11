"""「逐条」的正确实现：多条单问题序列组成一个 batch，一次 forward。"""
import json
import time

import pyarrow.parquet as pq
import torch
from unsloth import FastDecisionModel
from unsloth.models.clef import encode_record, collate_records

ADAPTER = r"model/adapter_v3"
full = pq.read_table(r"data/typed_full\train.parquet").to_pylist()
case = max((r for r in full if r["source"].startswith("workflow-controls")),
           key=lambda r: len(json.loads(r["questions"])))
qs = json.loads(case["questions"])

model, tok = FastDecisionModel.from_pretrained(
    model_name=ADAPTER, max_seq_length=1024, load_in_4bit=True)
model.eval()
hf = getattr(tok, "tokenizer", tok)
processor = tok if hasattr(tok, "tokenizer") else None
device = next(model.parameters()).device

recs = [encode_record(hf, {"state":case["state"], "questions":{n:q}, "images":None},
                      max_length=16384, processor=processor) for n, q in qs.items()]
print("17 条单问题序列长度:", [len(r.input_ids) for r in recs][:4], "...", flush=True)

def sync():
    torch.cuda.synchronize()

# warmup
b = collate_records(recs[:4], hf.pad_token_id, device)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
          records=b["records"], **b["media"])
sync()

results = {}
for bs in (1, 4, 8, 17):
    chunk = recs[:bs]
    try:
        torch.cuda.reset_peak_memory_stats()
        bb = collate_records(chunk, hf.pad_token_id, device)
        sync()
        t = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids=bb["input_ids"], attention_mask=bb["attention_mask"],
                              records=bb["records"], **bb["media"])
        sync()
        dt = time.perf_counter() - t
        results[str(bs)] = {"seconds": round(dt, 3), "qps": round(bs/dt, 2),
                            "peak_mib": round(torch.cuda.max_memory_allocated()/1024**2)}
        print(f"batch={bs:2d}  {dt:.3f}s  {bs/dt:.1f} 题/s  peak {results[str(bs)]['peak_mib']} MiB",
              flush=True)
    except Exception as e:
        results[str(bs)] = {"error": str(e)[:120]}
        print(f"batch={bs} 失败: {str(e)[:100]}", flush=True)

json.dump(results, open(r"results/per_question_batch.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
print("[DONE]", flush=True)
