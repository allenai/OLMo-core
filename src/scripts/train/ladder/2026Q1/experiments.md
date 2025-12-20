# Experiments

## mxfp8

### all linear layers (except lm_head)

```bash
uv run src/scripts/train/ladder/mxfp8_ladder.py launch-all --max-size 760M --cluster ai2/titan --name "olmo3-mxfp8-all-linear" --preemptible --chinchilla-multiple 8.0 --callbacks.gap_monitor.enabled=True --callbacks.gap_monitor.enabled=True --callbacks.gap_monitor.interval=50
```
