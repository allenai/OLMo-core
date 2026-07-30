"""Classify NQ k20 retrieval FAILURES by mode, joining eval details w/ data."""
import json, re

DATA = "data/nq_validation_k20_hn19_500.jsonl"
EVAL = "outputs/eval_results/nq_k20_std_4ep.json"
_T = re.compile(r'^\s*"(.*?)"')
def title(t):
    m=_T.match(t); return m.group(1).strip().lower() if m else None

rows=[json.loads(l) for l in open(DATA)]
det=json.load(open(EVAL))['results']['nq_validation_k20_hn19_500']['details']
assert len(rows)==len(det), (len(rows),len(det))

n=len(rows); wrong=0; same_title=0; diff_title=0; no_pred=0; gold_incidental=0
samples=[]
for ex,e in zip(rows,det):
    if e['exact_match']==1: continue
    wrong+=1
    docs=ex['documents']; gi=set(ex['gold_doc_indices'])
    gtitles={title(docs[g]['text']) for g in gi}; gtitles.discard(None)
    pids=[p-1 for p in e['predicted_ids'] if 0<=p-1<len(docs)]
    if not pids: no_pred+=1; continue
    ptitle=title(docs[pids[0]]['text'])
    if ptitle in gtitles: same_title+=1
    else: diff_title+=1
    if len(samples)<10:
        samples.append((ex['queries'][0], list(gtitles)[0] if gtitles else None,
                        ptitle, e['prediction'].replace(chr(10),' / ')[:80]))

print(f"k20: {n} ex, EM-wrong={wrong} ({100*wrong/n:.1f}%)")
print(f"  wrong & predicted SAME-title sibling of gold: {same_title} ({100*same_title/wrong:.1f}% of wrong)")
print(f"  wrong & predicted DIFFERENT-title doc:        {diff_title} ({100*diff_title/wrong:.1f}% of wrong)")
print(f"  wrong & no parseable prediction:              {no_pred}")
print("\n  sample wrong cases (question | gold_title | pred_title | pred):")
for q,gt,pt,pr in samples:
    print(f"   Q:{q!r}\n     gold={gt!r} pred={pt!r}  [{pr}]")
