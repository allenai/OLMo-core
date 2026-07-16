# v2 model configurations

`hybrid_wide.py` is the audited model-config source for the `expand_v=1`
integration-wide GDN family. It supports `275m`, `480m`, `810m`, and `1p2b`.
The Cx1 converted wide config is the canonical source for each size; the model
payload is identical across Cx1/Cx2/Cx4/Cx8.

The builder replaces only layers selected for sliding-window attention with
GatedDeltaNet. It derives the GDN head count from the source attention, fixes
`n_v_heads=n_heads`, `head_dim=128`, and `expand_v=1`, and verifies that all
non-mixer block fields remain identical. It does not alter MoE widths merely to
force exact parameter equality.

`geometry_matched_275m.py` designs the next 275M candidates on the dense
ladder's `d_model=640`, 10-layer, four-GDN/one-full-attention geometry. It
reports both a strict geometry-only profile and a profile that additionally
matches the dense 275M rung's 8-Q/8-KV full attention and elementwise gate.
Both use the dense hybrid's `expand_v=2` and deliberately retain RoPE and
`init_std=0.01`. The primary profile keeps our 8-Q/4-KV ungated full attention
and all previously audited FFN widths unchanged.

Run the structural and parameter audit without creating a training job:

```bash
PYTHONPATH=src .venv/bin/python \
  src/scripts/train/jacobm_olmoe_ladder/v2/models/hybrid_wide.py
```

Expected active-parameter comparison:

| Size | Wide active | Hybrid active | Delta | Active non-embedding delta |
|---|---:|---:|---:|---:|
| 275M | 280,207,872 | 288,194,512 | +2.850% | +3.932% |
| 480M | 486,348,800 | 501,228,784 | +3.060% | +3.879% |
| 810M | 823,569,920 | 859,400,792 | +4.351% | +5.155% |
| 1.2B | 1,225,011,712 | 1,288,662,592 | +5.196% | +5.944% |

These are active-near-matched architecture controls, consistent with the 275M
experiment. Exact equality would require changing a second architectural field
in addition to the mixer and would no longer isolate the GDN replacement.
