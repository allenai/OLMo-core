"""
Compute parameter counts and FLOPs for hybrid GatedDeltaNet-Transformer
and hybrid/pure Mamba2 models.

Creates the same OLMo3 configs the hybrid ladders use and computes exact
parameter counts and FLOP estimates for each model size.
"""

from olmo_core.model_ladder.analysis.model_specs import (
    OLMO3_SPECS,
    ModelSpec,
    compute_hybrid_mamba2_specs,
    compute_hybrid_specs,
    compute_olmo3_specs,
    fmt,
    gdn_dims,
    gdn_head_dim,
    mamba2_dims,
)


def print_detailed_gdn(result: dict):
    spec: ModelSpec = result["spec"]
    d = spec.d_model
    nh = spec.n_heads
    print(f"\n{'='*70}")
    print(
        f"  {spec.name}  (d_model={d}, n_heads={nh}, n_layers={spec.n_layers}, "
        f"mlp_hidden={spec.mlp_hidden_size})"
    )
    print(
        f"  GDN head_dim={gdn_head_dim(d, nh)}, "
        f"key_dim={gdn_dims(d, nh)[0]}, value_dim={gdn_dims(d, nh)[1]}"
    )
    print(f"  Attn layers: {result['n_attn']} at indices {result['attn_indices']}")
    print(f"  GDN  layers: {result['n_gdn']}")
    print(f"{'='*70}")

    print(f"\n  {'Component':<30} {'Params':>12}  {'FLOPs/tok':>14}")
    print(f"  {'-'*58}")
    print(f"  {'Embeddings':<30} {fmt(result['embed_params']):>12}  {'(lookup)':>14}")
    print(f"  {'LM Head':<30} {fmt(result['head_params']):>12}  {fmt(2*result['head_macs']):>14}")
    print(f"  {'Final LayerNorm':<30} {fmt(result['final_norm_params']):>12}  {'~0':>14}")

    n_gdn = result["n_gdn"]
    if n_gdn > 0:
        print(
            f"  {f'GDN layers (x{n_gdn})':<30} "
            f"{fmt(n_gdn * result['gdn_layer_params']):>12}  "
            f"{fmt(n_gdn * 2 * result['gdn_layer_macs']):>14}"
        )
        print(
            f"    {'(per layer)':<28} "
            f"{fmt(result['gdn_layer_params']):>12}  "
            f"{fmt(2 * result['gdn_layer_macs']):>14}"
        )

    n_attn = result["n_attn"]
    print(
        f"  {f'Attn layers (x{n_attn})':<30} "
        f"{fmt(n_attn * result['attn_layer_params']):>12}  "
        f"{fmt(n_attn * 2 * result['attn_layer_macs']):>14}"
    )
    print(
        f"    {'(per layer)':<28} "
        f"{fmt(result['attn_layer_params']):>12}  "
        f"{fmt(2 * result['attn_layer_macs']):>14}"
    )

    print(f"  {'-'*58}")
    print(
        f"  {'TOTAL':<30} {fmt(result['total_params']):>12}  {fmt(result['total_flops_per_token']):>14}"
    )
    print(f"  {'Non-embedding':<30} {fmt(result['non_embed_params']):>12}")


def print_detailed_mamba2(result: dict):
    spec: ModelSpec = result["spec"]
    d = spec.d_model
    nh = spec.n_heads
    dims = mamba2_dims(d, nh)
    print(f"\n{'='*70}")
    print(
        f"  {spec.name}  (d_model={d}, n_heads={nh}, n_layers={spec.n_layers}, "
        f"mlp_hidden={spec.mlp_hidden_size})"
    )
    print(
        f"  Mamba2 head_dim={dims['head_dim']}, "
        f"intermediate={dims['intermediate_size']}, conv_dim={dims['conv_dim']}"
    )
    print(f"  Attn layers: {result['n_attn']} at indices {result['attn_indices']}")
    print(f"  Mamba2 layers: {result['n_mamba2']}")
    print(f"{'='*70}")

    print(f"\n  {'Component':<30} {'Params':>12}  {'FLOPs/tok':>14}")
    print(f"  {'-'*58}")
    print(f"  {'Embeddings':<30} {fmt(result['embed_params']):>12}  {'(lookup)':>14}")
    print(f"  {'LM Head':<30} {fmt(result['head_params']):>12}  {fmt(2*result['head_macs']):>14}")
    print(f"  {'Final LayerNorm':<30} {fmt(result['final_norm_params']):>12}  {'~0':>14}")

    n_m2 = result["n_mamba2"]
    if n_m2 > 0:
        print(
            f"  {f'Mamba2 layers (x{n_m2})':<30} "
            f"{fmt(n_m2 * result['mamba2_layer_params']):>12}  "
            f"{fmt(n_m2 * 2 * result['mamba2_layer_macs']):>14}"
        )
        print(
            f"    {'(per layer)':<28} "
            f"{fmt(result['mamba2_layer_params']):>12}  "
            f"{fmt(2 * result['mamba2_layer_macs']):>14}"
        )

    n_attn = result["n_attn"]
    if n_attn > 0:
        print(
            f"  {f'Attn layers (x{n_attn})':<30} "
            f"{fmt(n_attn * result['attn_layer_params']):>12}  "
            f"{fmt(n_attn * 2 * result['attn_layer_macs']):>14}"
        )
        print(
            f"    {'(per layer)':<28} "
            f"{fmt(result['attn_layer_params']):>12}  "
            f"{fmt(2 * result['attn_layer_macs']):>14}"
        )

    print(f"  {'-'*58}")
    print(
        f"  {'TOTAL':<30} {fmt(result['total_params']):>12}  {fmt(result['total_flops_per_token']):>14}"
    )
    print(f"  {'Non-embedding':<30} {fmt(result['non_embed_params']):>12}")


def print_summary_table(hybrid_results: list, olmo3_results: list, seq_len: int, layer_type: str):
    n_ssm_key = "n_gdn" if layer_type == "gdn" else "n_mamba2"
    ssm_label = "GDN" if layer_type == "gdn" else "Mamba2"
    hybrid_label = f"Hybrid {ssm_label}"

    print(f"\n{'='*120}")
    print(f"  SUMMARY TABLE  (seq_len={seq_len}, layer_type={layer_type})")
    print(f"{'='*120}")
    header = (
        f"  {'Size':<6} | {'d_model':>7} | {'n_L':>3} | {'n_attn':>6} | {f'n_{ssm_label.lower()[:3]}':>5} | "
        f"{f'{hybrid_label} Params':>14} | {f'{hybrid_label} NonEmb':>14} | {f'{hybrid_label} FLOP/tok':>16} | "
        f"{'OLMo3 Params':>13} | {'OLMo3 NonEmb':>13} | {'OLMo3 FLOP/tok':>15} | "
        f"{'Param Ratio':>11}"
    )
    print(header)
    print(f"  {'-'*(len(header)-2)}")
    for h, o in zip(hybrid_results, olmo3_results):
        s = h["spec"]
        ratio = h["non_embed_params"] / o["non_embed_params"] if o["non_embed_params"] else 0
        print(
            f"  {s.name:<6} | {s.d_model:>7} | {s.n_layers:>3} | {h['n_attn']:>6} | {h[n_ssm_key]:>5} | "
            f"{fmt(h['total_params']):>14} | {fmt(h['non_embed_params']):>14} | {fmt(h['total_flops_per_token']):>16} | "
            f"{fmt(o['total_params']):>13} | {fmt(o['non_embed_params']):>13} | {fmt(o['total_flops_per_token']):>15} | "
            f"{ratio:>10.4f}x"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute hybrid model specs")
    parser.add_argument(
        "--layer-type",
        choices=["gdn", "mamba2"],
        default="gdn",
        help="SSM layer type: gdn (GatedDeltaNet) or mamba2 (default: gdn)",
    )
    parser.add_argument(
        "--transformer-ratio",
        type=int,
        default=4,
        help="Ratio of layers between transformer blocks (default: 4). Set to 0 for pure SSM.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8192,
        help="Sequence length for FLOP calculation (default: 8192)",
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        default=None,
        help="Specific sizes to compute (e.g., 100M 1B 7B). Default: all.",
    )
    parser.add_argument("--summary-only", action="store_true", help="Only print the summary table")
    args = parser.parse_args()

    specs = OLMO3_SPECS
    if args.sizes:
        specs = [s for s in OLMO3_SPECS if s.name in args.sizes]

    hybrid_results = []
    olmo3_results = []
    for spec in specs:
        if args.layer_type == "mamba2":
            h = compute_hybrid_mamba2_specs(
                spec,
                transformer_ratio=args.transformer_ratio,
                seq_len=args.seq_len,
                force_final_attention=args.transformer_ratio > 0,
            )
        else:
            h = compute_hybrid_specs(
                spec,
                transformer_ratio=args.transformer_ratio,
                seq_len=args.seq_len,
                force_final_attention=args.transformer_ratio > 0,
            )
        o = compute_olmo3_specs(spec, seq_len=args.seq_len)
        hybrid_results.append(h)
        olmo3_results.append(o)

        if not args.summary_only:
            if args.layer_type == "mamba2":
                print_detailed_mamba2(h)
            else:
                print_detailed_gdn(h)

    print_summary_table(hybrid_results, olmo3_results, args.seq_len, args.layer_type)


if __name__ == "__main__":
    main()
