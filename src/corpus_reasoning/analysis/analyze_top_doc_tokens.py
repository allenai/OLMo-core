"""Analyze top document tokens across all configs to find patterns."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

def load_all_files(pattern_prefix, configs):
    all_data = {}
    for cfg in configs:
        path = f"outputs/{pattern_prefix}_{cfg}.json"
        try:
            with open(path) as f:
                all_data[cfg] = json.load(f)
        except FileNotFoundError:
            print(f"  Skipping {path}")
    return all_data

def get_query_answer_words(result):
    """Extract query and answer words from the example."""
    tokens = result['tokens']
    token_types = result['token_types']
    
    # Get query tokens
    query_words = set()
    for tok, tt in zip(tokens, token_types):
        if tt == 'query':
            w = tok.strip().lower()
            if len(w) > 2:
                query_words.add(w)
    
    # Get output tokens
    output_words = set()
    for tok, tt in zip(tokens, token_types):
        if tt == 'output':
            w = tok.strip().lower()
            if len(w) > 2:
                output_words.add(w)
    
    return query_words, output_words

def analyze_gradient_top_tokens(all_data, top_k=400):
    """Find top-k document tokens by gradient norm across all examples/configs."""
    
    # Collect all document token entries with their grad norms
    entries = []  # (avg_grad, token, config, example_idx, token_idx, in_query, in_answer)
    
    for cfg, data in all_data.items():
        for result in data['results']:
            tokens = result['tokens']
            token_types = result['token_types']
            grad_norms = result['gradient_norms']
            
            query_words, output_words = get_query_answer_words(result)
            
            # Get document token indices
            for i, (tok, tt) in enumerate(zip(tokens, token_types)):
                if tt != 'document':
                    continue
                
                # Average gradient across all layers
                layer_grads = []
                for layer_key, norms in grad_norms.items():
                    layer_grads.append(norms[i])
                avg_grad = sum(layer_grads) / len(layer_grads) if layer_grads else 0
                
                tok_clean = tok.strip().lower()
                in_query = tok_clean in query_words or any(tok_clean in qw for qw in query_words)
                in_answer = tok_clean in output_words or any(tok_clean in aw for aw in output_words)
                
                entries.append({
                    'avg_grad': avg_grad,
                    'token': tok,
                    'token_clean': tok_clean,
                    'config': cfg,
                    'example_idx': result['example_idx'],
                    'token_idx': i,
                    'in_query': in_query,
                    'in_answer': in_answer,
                })
    
    # Sort by gradient norm
    entries.sort(key=lambda x: x['avg_grad'], reverse=True)
    top = entries[:top_k]
    
    return top, entries

def analyze_repr_top_tokens(all_data, top_k=400):
    """Find top-k document tokens by representation change."""
    entries = []
    
    for cfg, data in all_data.items():
        for result in data['results']:
            tokens = result['tokens']
            token_types = result['token_types']
            cosine_dists = result['cosine_distances']
            
            query_words, output_words = get_query_answer_words(result)
            
            for i, (tok, tt) in enumerate(zip(tokens, token_types)):
                if tt != 'document':
                    continue
                
                # Average cosine distance across layers (skip layer 0)
                layer_dists = []
                for layer_key, dists in cosine_dists.items():
                    if layer_key == '0':
                        continue
                    layer_dists.append(dists[i])
                avg_dist = sum(layer_dists) / len(layer_dists) if layer_dists else 0
                
                tok_clean = tok.strip().lower()
                in_query = tok_clean in query_words or any(tok_clean in qw for qw in query_words)
                in_answer = tok_clean in output_words or any(tok_clean in aw for aw in output_words)
                
                entries.append({
                    'avg_cosine_dist': avg_dist,
                    'token': tok,
                    'token_clean': tok_clean,
                    'config': cfg,
                    'example_idx': result['example_idx'],
                    'token_idx': i,
                    'in_query': in_query,
                    'in_answer': in_answer,
                })
    
    entries.sort(key=lambda x: x['avg_cosine_dist'], reverse=True)
    top = entries[:top_k]
    return top, entries

def print_analysis(top, metric_key, metric_name):
    print(f"\n{'='*80}")
    print(f"TOP 400 DOCUMENT TOKENS BY {metric_name.upper()}")
    print(f"{'='*80}")
    
    # Stats
    n_in_query = sum(1 for e in top if e['in_query'])
    n_in_answer = sum(1 for e in top if e['in_answer'])
    n_in_either = sum(1 for e in top if e['in_query'] or e['in_answer'])
    n_in_both = sum(1 for e in top if e['in_query'] and e['in_answer'])
    
    print(f"\n--- Overlap with query/answer ---")
    print(f"  In query words:  {n_in_query}/400 ({n_in_query/4:.1f}%)")
    print(f"  In answer words: {n_in_answer}/400 ({n_in_answer/4:.1f}%)")
    print(f"  In either:       {n_in_either}/400 ({n_in_either/4:.1f}%)")
    print(f"  In both:         {n_in_both}/400 ({n_in_both/4:.1f}%)")
    
    # Token frequency
    print(f"\n--- Most common tokens in top 400 ---")
    tok_counts = Counter(e['token_clean'] for e in top)
    for tok, count in tok_counts.most_common(30):
        # Check if these are commonly in query/answer
        in_q = sum(1 for e in top if e['token_clean'] == tok and e['in_query'])
        in_a = sum(1 for e in top if e['token_clean'] == tok and e['in_answer'])
        flags = []
        if in_q > 0: flags.append(f"query:{in_q}")
        if in_a > 0: flags.append(f"answer:{in_a}")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"  '{tok}': {count}{flag_str}")
    
    # Config distribution
    print(f"\n--- Config distribution ---")
    cfg_counts = Counter(e['config'] for e in top)
    for cfg, count in cfg_counts.most_common():
        print(f"  {cfg}: {count}")
    
    # Position within document analysis
    print(f"\n--- Token position analysis ---")
    # Check if high-grad tokens tend to be at doc boundaries
    boundary_tokens = {'<', '>', 'doc', 'document', '\n', ':', '/'}
    n_boundary = sum(1 for e in top if any(b in e['token_clean'] for b in boundary_tokens))
    print(f"  Near doc boundaries (special tokens): {n_boundary}/400")
    
    # Show top 50 with context
    print(f"\n--- Top 50 tokens ---")
    print(f"{'Rank':>4} | {metric_name:>10} | {'Token':>15} | {'Config':>20} | {'Ex':>3} | {'Pos':>5} | Flags")
    print("-" * 90)
    for rank, e in enumerate(top[:50], 1):
        flags = []
        if e['in_query']: flags.append("Q")
        if e['in_answer']: flags.append("A")
        flag_str = ",".join(flags) if flags else "-"
        print(f"{rank:4d} | {e[metric_key]:10.6f} | {repr(e['token']):>15} | {e['config']:>20} | {e['example_idx']:3d} | {e['token_idx']:5d} | {flag_str}")

    # Look at surrounding context for top 20
    print(f"\n--- Context for top 20 (±3 tokens) ---")

configs = ['std_qafter', 'std_qbefore', 'std_qboth', 'chunked_qafter', 'chunked_qbefore', 'chunked_qboth']

print("Loading gradient files...")
grad_data = load_all_files('grad', configs)
print("Loading repr files...")
repr_data = load_all_files('repr', configs)

# Gradient analysis
top_grad, all_grad = analyze_gradient_top_tokens(grad_data, 400)
print_analysis(top_grad, 'avg_grad', 'Avg Gradient')

# Also check: what fraction of ALL document tokens overlap with query/answer?
print(f"\n--- Baseline: overlap in ALL document tokens ---")
n_total = len(all_grad)
n_q_all = sum(1 for e in all_grad if e['in_query'])
n_a_all = sum(1 for e in all_grad if e['in_answer'])
n_either_all = sum(1 for e in all_grad if e['in_query'] or e['in_answer'])
print(f"  Total doc tokens: {n_total}")
print(f"  In query words:  {n_q_all}/{n_total} ({n_q_all/n_total*100:.1f}%)")
print(f"  In answer words: {n_a_all}/{n_total} ({n_a_all/n_total*100:.1f}%)")
print(f"  In either:       {n_either_all}/{n_total} ({n_either_all/n_total*100:.1f}%)")

# Representation analysis
top_repr, all_repr = analyze_repr_top_tokens(repr_data, 400)
print_analysis(top_repr, 'avg_cosine_dist', 'Avg Cosine Dist')

print(f"\n--- Baseline: overlap in ALL document tokens (repr) ---")
n_total = len(all_repr)
n_q_all = sum(1 for e in all_repr if e['in_query'])
n_a_all = sum(1 for e in all_repr if e['in_answer'])
n_either_all = sum(1 for e in all_repr if e['in_query'] or e['in_answer'])
print(f"  Total doc tokens: {n_total}")
print(f"  In query words:  {n_q_all}/{n_total} ({n_q_all/n_total*100:.1f}%)")
print(f"  In answer words: {n_a_all}/{n_total} ({n_a_all/n_total*100:.1f}%)")
print(f"  In either:       {n_either_all}/{n_total} ({n_either_all/n_total*100:.1f}%)")
