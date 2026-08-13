# Per-task base ladder definition -- the SINGLE source of truth for which rungs each task runs at.
#
# Sourced (not executed) by every on-node eval runner:
#   run_beaker_multirung_eval.sh          (olmo_core backend: dense/landmark/compressive/docchunk)
#   ../hils_eval/run_beaker_hf_eval.sh    (hf backend: HiLS-Attention, Olmo-3 base, any HF model)
#
# In:  $TASK
# Out: $RUNGS (comma list), $LTASK (the evaluator's ladder-task key), $EXTRA (extra CLI flags)
#
# This lives in its own file because it is the EVAL DEFINITION, not runner plumbing: two runners
# with two copies of the table is how a model ends up scored on a different ladder than the model
# it is being plotted against, with nothing in either result file to show it. Changing a row here
# changes what every backend measures -- which is the intent.
case "$TASK" in
  contra)  RUNGS="2k,8k,16k,32k"; LTASK=contradiction; EXTRA="--contra-max-new-tokens 512" ;;
  nq)      RUNGS="3k,8k,16k,32k"; LTASK=nq;            EXTRA="" ;;
  outlier) RUNGS="3k,8k,16k,32k"; LTASK=outlier;       EXTRA="" ;;
  rerank)  RUNGS="3k,8k,16k";     LTASK=rerank;        EXTRA="" ;;  # CE-graded, no 32k pool
  oolong)  RUNGS="8k,16k,32k";    LTASK=oolong;        EXTRA="" ;;
  fiqa)    RUNGS="2k,4k,8k,16k";  LTASK=fiqa;          EXTRA="" ;;  # OOD generalization (BEIR)
  scifact) RUNGS="4k,8k,16k,32k"; LTASK=scifact;       EXTRA="" ;;  # OOD generalization (BEIR)
  outlier_review) RUNGS="3k,8k,16k,32k"; LTASK=outlier_review; EXTRA="" ;;  # OOD (Amazon reviews)
  contra_fever)   RUNGS="2k,8k,16k,32k"; LTASK=contra_fever;   EXTRA="--contra-max-new-tokens 512" ;;  # OOD (FEVER)
  *) echo "ERROR unknown TASK=$TASK"; exit 2 ;;
esac
