
def num_floating_point_operations_for_single_layer(args, batch_size):

    """Calculate FLOPs for a standard Transformer model."""
    # Attention projection size.
    # general
    assert hasattr(args, 'd_model')
    assert hasattr(args, 'swiglu')
    assert hasattr(args, 'seq_length')
    # attn
    assert hasattr(args, 'multi_latent_attention')
    assert hasattr(args, 'num_query_groups')
    
    # dense mlp
    assert hasattr(args, 'ffn_hidden_size')
    assert hasattr(args, 'dense_moe_mlp')
    if args.ffn_hidden_size is None: # no dense mlp
        args.ffn_hidden_size = 0
    
    # routed mlp
    assert hasattr(args, 'num_experts')
    assert hasattr(args, 'moe_router_topk')
    assert hasattr(args, 'moe_ffn_hidden_size')
    
    # shared mlp
    assert hasattr(args, 'moe_shared_expert_intermediate_size')
    assert hasattr(args, 'shared_expert_count')
    

    num_experts_routed_to = (
        args.moe_router_topk if args.moe_router_topk is not None 
        else 0
        )

    moe_ffn_hidden_size = args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else 0
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )
    # SwiGLU.
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    if args.dense_moe_mlp:
        dense_mlp_multiplier = 4 / 3
    else:
        dense_mlp_multiplier = 1
    # The 12x term below comes from the following factors; for more details, see
    # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
    # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
    #       backward wgrad [weight gradient], backward dgrad [data gradient]).
    # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
    #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
    #       in MLP layer).
    # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
    expansion_factor = 3 * 2 * 2

    if args.multi_latent_attention:
        assert not args.group_query_attention
        '''
        Basic arithmetic
        let B is batch size, s is seq_len, h is embedding dim,
        for one self_attnetion block (prenorm is not included)
        qkv projection:  6Bsh^2
        attn:            2Bs^2h
        attn over value: 2Bs^2h
        oproj:           2Bsh^2

        references
        https://arxiv.org/abs/2305.10403
        https://arxiv.org/abs/2205.05198
        '''
        ## MLA
        if args.q_lora_rank is None:
            q_term = args.hidden_size * args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
        else:
            q_term = args.q_lora_rank * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim) + 1) 
        self_attn_term = (
            3*2 # fwd(1) + bwd(2) *FMA 
            * (
                ## q lora + rope + q norm
                q_term

                ## kv lora + rope + kv norm
                + args.kv_lora_rank
                * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim) + 1)
                + args.hidden_size * args.qk_pos_emb_head_dim

                ## o proj
                + (args.num_attention_heads * args.v_head_dim) * args.hidden_size

                ## core attn
                + args.seq_length * (args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)) / 2
                + args.seq_length * args.num_attention_heads * args.v_head_dim / 2
            )
        )

    else:
        ## MHA or GQA
        self_attn_term = (
            expansion_factor
            * args.d_model
            * args.d_model
            * (
                (
                    1
                    + (args.num_query_groups / args.num_attention_heads)
                    # Only half of the attention matrix is non-zero and needs to be multiplied with V.
                    + (args.seq_length / args.d_model / 2)
                ) 
            )
        )
    attn_flops = batch_size * args.seq_length * ( self_attn_term )
    mlp_flops = batch_size * args.seq_length * (
        expansion_factor
        * 1 # num_layers
        * args.d_model
        * (
            # dense mlp
            (
                args.ffn_hidden_size
                * gated_linear_multiplier * dense_mlp_multiplier
            )
            # routed experts
            + (
                moe_ffn_hidden_size
                * gated_linear_multiplier
            ) * num_experts_routed_to
            # Shared Experts.
            + (
                shared_expert_ffn_hidden_size 
                * gated_linear_multiplier
            ) * args.shared_expert_count
        )
    )
        # Se
    total_floating_point_operations = attn_flops + mlp_flops
    return total_floating_point_operations

def num_floating_point_operations_for_logits(config, seq_len): 
    # for one sequence
    total_floating_point_operations = 1 * seq_len * (
        3*2
        * config.d_model
        * config.vocab_size 
        * 1
    )
    return total_floating_point_operations