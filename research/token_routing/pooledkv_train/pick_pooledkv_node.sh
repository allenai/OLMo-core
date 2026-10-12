#!/bin/bash
# Pick the staged H200 node where a pooledkv job will start soonest.
#
# All three nodes below hold the pooledkv bases + shards on node-local /data (sneetches staged
# 2026-08-30 by job 3486712), so train jobs can go to ANY of them -- only evals/resumes must stay
# on the node that owns the checkpoint. For each candidate we count GPUs that are free outright
# plus GPUs held by preemptible QOS jobs (normal/preemptive), since our submissions ride the
# partition's preempting QOS and start by preempting those.
#
# Prints ready-to-use sbatch args (-w node -p partition -q qos); exits 1 if none can host. The
# QOS must match the node's partition: horton lives in berkeleynlp, whose preempting QOS is
# preemptive_high_sewonm -- plain preemptive_high is jsteinhardt-only and a job submitted with it
# to horton pends forever. An optional BUSY env var (space-separated node names) excludes nodes
# this same submit loop already claimed, since slurm can't see our just-submitted pending jobs.
#
#   ARGS=$(bash pick_pooledkv_node.sh [ngpu]); sbatch $ARGS --export=... run_q4b_pooledkv_horton.sbatch
#   ARGS2=$(BUSY=horton bash pick_pooledkv_node.sh 2)
set -uo pipefail
NGPU="${1:-2}"
CANDIDATES=(horton mooney sneetches)
declare -A PART=([horton]=berkeleynlp [mooney]=jsteinhardt [sneetches]=jsteinhardt)
declare -A QOS=([horton]=preemptive_high_sewonm [mooney]=preemptive_high [sneetches]=preemptive_high)
best="" best_score=-1
for n in "${CANDIDATES[@]}"; do
  case " ${BUSY:-} " in *" $n "*) continue ;; esac
  total=$(scontrol show node "$n" | grep -oP 'CfgTRES=.*?gres/gpu=\K\d+' | head -1)
  alloc=$(scontrol show node "$n" | grep -oP 'AllocTRES=.*?gres/gpu=\K\d+' | head -1)
  free=$(( ${total:-0} - ${alloc:-0} ))
  # GPUs held by jobs we can preempt (any partition; QOS normal or preemptive)
  preemptible=$(squeue -w "$n" -t RUNNING -o "%q %b" -h \
    | awk '$1=="normal" || $1=="preemptive" {n=0; if (match($2, /gpu[:=]([A-Za-z0-9]+:)?[0-9]+/)) {m=substr($2,RSTART,RLENGTH); sub(/.*[:=]/,"",m); n=m} s+=n} END {print s+0}')
  score=$(( free * 100 + preemptible ))          # prefer truly-free over preemption
  if [ $(( free + preemptible )) -ge "$NGPU" ] && [ "$score" -gt "$best_score" ]; then
    best=$n; best_score=$score
  fi
done
[ -n "$best" ] || { echo "no candidate node can host ${NGPU} GPUs" >&2; exit 1; }
echo "-w $best -p ${PART[$best]} -q ${QOS[$best]}"
