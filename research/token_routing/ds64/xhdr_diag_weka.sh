set -uo pipefail
R=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/outlier_lengthmix/eval_rungs/outlier
/opt/conda/bin/python - "$R" <<'EOF'
import json,sys,statistics,os,glob
R=sys.argv[1]
for f in ["rung_2048.jsonl","rung_8192.jsonl","rung_16384.jsonl","rung_32768.jsonl"]:
    p=os.path.join(R,f)
    if not os.path.exists(p): print("MISSING",p); continue
    nd=[];ng=[]
    for i,l in enumerate(open(p)):
        if i>=500: break
        e=json.loads(l); e=e.get("ex",e)
        nd.append(len(e["documents"])); ng.append(len(e.get("gold_doc_indices",[])))
    md=statistics.mean(nd); mg=statistics.mean(ng)
    print(f, "rows",len(nd),"n_docs mean",round(md,1),"min",min(nd),"max",max(nd),
          "| k mean",round(mg,2), "| uniform-guess f1 = k/n =", round(mg/md,4))
print("=== RESULT JSONS ===")
for pat in ["/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ds64-outlier-*/eval/outlier_multirung.json",
            "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results/ds64-outlier-*_outlier_multirung.json",
            "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/ds64-outlier-*/eval/outlier_multirung.json"]:
    for f in sorted(glob.glob(pat)):
        try: print(os.path.basename(os.path.dirname(os.path.dirname(f)))or f, f, json.load(open(f)))
        except Exception as ex: print("ERR",f,ex)
EOF
