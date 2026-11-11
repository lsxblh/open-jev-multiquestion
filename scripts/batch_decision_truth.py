"""分层版：N=2..6 各取若干组，记录每题明细，判断两种方式的对错。"""
import collections
import json
import random
import time

import pyarrow.parquet as pq
import torch
from unsloth import FastDecisionModel
from unsloth.models._decision_common import QUESTION_TYPES
from unsloth.models.clef import encode_record, collate_records, systemone_answer
from unsloth.models.decision import _served_temperatures, _predicted

ADAPTER = r"model/adapter_v3"
OUT = r"results/batch_decision_truth.json"
DETAIL = r"results/batch_decision_detail.json"
rows = pq.read_table(r"data/typed_split\test.parquet").to_pylist()
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["group_id"], r["state"])].append(r)

by_n = collections.defaultdict(list)
for k, v in groups.items():
    if 2 <= len(v) <= 6:
        by_n[len(v)].append((k, v))
random.seed(3407)
sample = []
for n in sorted(by_n):
    pick = random.sample(by_n[n], min(12, len(by_n[n])))
    sample += pick
    print(f"N={n} 可用 {len(by_n[n])} 组, 取 {len(pick)}", flush=True)
print("总组数", len(sample), flush=True)

def norm(ans, kind):
    return "true" if (kind == "noul" and ans is True) else ("false" if kind == "noul" else str(ans))

model, tok = FastDecisionModel.from_pretrained(
    model_name=ADAPTER, max_seq_length=1024, load_in_4bit=True)
model.eval()
hf = getattr(tok, "tokenizer", tok)
processor = tok if hasattr(tok, "tokenizer") else None
device = next(model.parameters()).device

def per_question(state, items, bs=4):
    recs, names = [], []
    for r in items:
        q = json.loads(r["questions"])
        names.append(list(q)[0])
        recs.append(encode_record(hf, {"state":state,"questions":q,"images":None},
                                 max_length=16384, processor=processor))
    out = {}
    for s in range(0, len(recs), bs):
        cr, cn, orig = recs[s:s+bs], names[s:s+bs], items[s:s+bs]
        batch = collate_records(cr, hf.pad_token_id, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              records=batch["records"], **batch["media"])
        rr = [row[:len(rec.questions[0].option_ids)] for rec, row in zip(cr, logits.float().cpu())]
        qt = [{"qtype": QUESTION_TYPES.index(json.loads(o["questions"])[n]["type"])}
              for o, n in zip(orig, cn)]
        scales = _served_temperatures(model.decision_config, rr, qt)
        for rec, name, row, scale, o in zip(cr, cn, rr, scales, orig):
            q = rec.questions[0]
            probs = {k: float(v) for k,v in zip(q.option_ids, (row/scale).softmax(-1).tolist())}
            asked = json.loads(o["questions"])[name]
            out[name] = _predicted(asked, systemone_answer(asked, probs), probs)
    return out

def joint(state, questions):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return FastDecisionModel.predict(model, tok, state, questions)

# warmup
k, v = sample[0]
_ = per_question(k[1], v[:2])
_ = joint(k[1], {list(json.loads(x["questions"]))[0]: list(json.loads(x["questions"]).values())[0] for x in v[:2]})

stats = collections.defaultdict(lambda: {"n":0,"ah":0,"bh":0,"ac":0.0,"bc":0.0,
                                          "both":0,"only_a":0,"only_b":0,"neither":0})
details = []
t0 = time.perf_counter()
for gi, (key, items) in enumerate(sample):
    state = key[1]
    qs, gold = {}, {}
    for r in items:
        q = json.loads(r["questions"]); name = list(q)[0]
        qs[name] = q[name]
        gold[name] = str(json.loads(r["gold"])[name]["label"])
    try:
        A = per_question(state, items)
        B = joint(state, qs)
    except Exception as e:
        print("skip", gi, str(e)[:80], flush=True); continue
    st = stats[len(items)]
    for name in qs:
        kind = qs[name]["type"]
        ha = norm(A[name].get("answer"), kind) == gold[name]
        hb = norm(B[name].get("answer"), kind) == gold[name]
        pa = max((A[name].get("probabilities") or {"x":0}).values())
        pb = max((B[name].get("probabilities") or {"x":0}).values())
        st["n"] += 1; st["ah"] += ha; st["bh"] += hb; st["ac"] += pa; st["bc"] += pb
        st["both"] += (ha and hb); st["only_a"] += (ha and not hb)
        st["only_b"] += (hb and not ha); st["neither"] += (not ha and not hb)
        details.append({"N":len(items),"q":name,"type":kind,"gold":gold[name],
                        "a":str(A[name].get("answer")),"b":str(B[name].get("answer")),
                        "ha":ha,"hb":hb,"pa":round(pa,4),"pb":round(pb,4)})
    if (gi+1) % 15 == 0:
        print(f"  {gi+1}/{len(sample)} 组, {time.perf_counter()-t0:.0f}s", flush=True)

result = {}
for n, s in sorted(stats.items()):
    result[f"N={n}"] = {"questions":s["n"], "per_question_acc":round(s["ah"]/s["n"],4),
        "joint_acc":round(s["bh"]/s["n"],4), "conf_a":round(s["ac"]/s["n"],4),
        "conf_b":round(s["bc"]/s["n"],4), "both_right":s["both"],
        "only_per_question":s["only_a"], "only_joint":s["only_b"], "both_wrong":s["neither"]}
    print(f"N={n}: 逐条 {s['ah']}/{s['n']} ({s['ah']/s['n']:.3f})  "
          f"一次 {s['bh']/s['n']} ({s['bh']/s['n']:.3f})  "
          f"仅逐条对 {s['only_a']} / 仅一次对 {s['only_b']}", flush=True)
T = {k:sum(s[k] for s in stats.values()) for k in ("n","ah","bh","both","only_a","only_b","neither")}
result["TOTAL"] = {"questions":T["n"], "per_question_acc":round(T["ah"]/T["n"],4),
    "joint_acc":round(T["bh"]/T["n"],4), "both_right":T["both"],
    "only_per_question":T["only_a"], "only_joint":T["only_b"], "both_wrong":T["neither"]}
print("TOTAL:", result["TOTAL"], flush=True)
json.dump(result, open(OUT,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(details, open(DETAIL,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("[DONE]", flush=True)
