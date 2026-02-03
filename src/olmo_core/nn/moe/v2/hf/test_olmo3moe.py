import torch
import transformers
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM


@torch.no_grad()
def topk_match_rate(logits1: torch.Tensor, logits2: torch.Tensor, k: int = 5):
    """
    Compute top-k match statistics between two logits tensors.

    logits1, logits2: shape [..., vocab] (e.g. [seq, vocab] or [batch, seq, vocab])
    Returns:
      - per-position top-k overlap count (0..k)
      - top-k "hit" rate: whether argmax of one is in top-k of the other (both directions)
      - exact top-k set match rate (order-insensitive)
      - average Jaccard overlap of top-k sets
    """
    assert logits1.shape == logits2.shape, (logits1.shape, logits2.shape)
    assert logits1.shape[-1] >= k

    # flatten everything except vocab
    v = logits1.shape[-1]
    a = logits1.reshape(-1, v)
    b = logits2.reshape(-1, v)

    topk_a = torch.topk(a, k=k, dim=-1).indices  # [N, k]
    topk_b = torch.topk(b, k=k, dim=-1).indices  # [N, k]

    # (1) overlap count per position
    # Use broadcasting to test membership: [N,k,1] vs [N,1,k] -> [N,k,k]
    overlap = (topk_a.unsqueeze(-1) == topk_b.unsqueeze(-2)).any(dim=-1)  # [N,k] each a-item in b-topk?
    overlap_count = overlap.sum(dim=-1)  # [N]

    # (2) "hit" rates: argmax in other's top-k
    argmax_a = topk_a[:, 0]
    argmax_b = topk_b[:, 0]
    a_in_b = (argmax_a.unsqueeze(-1) == topk_b).any(dim=-1).float()  # [N]
    b_in_a = (argmax_b.unsqueeze(-1) == topk_a).any(dim=-1).float()  # [N]

    # (3) exact top-k set match (order-insensitive): compare sorted indices
    sorted_a, _ = torch.sort(topk_a, dim=-1)
    sorted_b, _ = torch.sort(topk_b, dim=-1)
    exact_set_match = (sorted_a == sorted_b).all(dim=-1).float()  # [N]

    # (4) Jaccard overlap of top-k sets: |A∩B| / |A∪B| = overlap_count / (2k - overlap_count)
    jaccard = overlap_count.float() / (2 * k - overlap_count).clamp_min(1).float()

    stats = {
        "N_positions": int(a.shape[0]),
        "k": k,
        "mean_overlap_count": float(overlap_count.float().mean().item()),
        "topk_hit_rate_a_in_b": float(a_in_b.mean().item()),
        "topk_hit_rate_b_in_a": float(b_in_a.mean().item()),
        "exact_topk_set_match_rate": float(exact_set_match.mean().item()),
        "mean_jaccard": float(jaccard.mean().item()),
    }
    return stats

if __name__ == "__main__":

    tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # option 1: randomly initialize a model
    # config = Olmo3MoeConfig(num_hidden_layers=4,)
    # model = Olmo3MoeForCausalLM(config).to(device)

    # option 2: load a model from converted checkpoint
    load_path = '/workspace/tmp/step10000_hf_model3'
    model = Olmo3MoeForCausalLM.from_pretrained(load_path).to(device).to(torch.bfloat16)

    input_ids = '/workspace/tmp/input_ids.pt'
    with open(input_ids, 'rb') as f:
        input_ids = torch.load(f).to(device)

    # x = torch.randint(0, model.config.vocab_size, (2, 16)).to(device)

    outputs = model(input_ids=input_ids)
    print(outputs.logits.shape)  # should be (2, 16, vocab_size)

    ref_lm_head_logits = '/workspace/tmp/lm_head_logits.pt'
    with open(ref_lm_head_logits, 'rb') as f:
        ref_lm_head_logits = torch.load(f).to(device)

    # compare with reference logits atol=1e-3
    print('HF logits:')
    print(outputs.logits)
    print('Reference logits:')
    print(ref_lm_head_logits)
    

    ref_argmax_ids = '/workspace/tmp/lm_head_logits_argmax.pt'
    with open(ref_argmax_ids, 'rb') as f:
        ref_argmax_ids = torch.load(f).to(device)

    decode_ids = torch.argmax(outputs.logits, dim=-1)

    print('Decoded token IDs:')
    print(decode_ids)
    print('Reference token IDs:')
    print(ref_argmax_ids)   


    input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    print('Input text:')
    print(input_text[0])

    decoded_text = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)
    print('Output text:')
    print(decoded_text[0])


    # generate samples
    input_prompt = "In a distant future, humanity has"
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(input_ids, max_new_tokens=50, do_sample=True, top_k=50, top_p=0.95, temperature=1.0)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print('***** Generated text ******')
    print(generated_text[0])

    print("****** SUMMARY ******")

    # percentage of matching token ids
    num_matching = (decode_ids == ref_argmax_ids).sum().item()
    total_ids = decode_ids.numel()
    print(f'Number of matching token IDs: {num_matching}/{total_ids} ({num_matching/total_ids*100:.2f}%)')

    # percentage of elements with atol>1e-3
    atol = 0.01
    diff = torch.abs(outputs.logits - ref_lm_head_logits)
    num_total = diff.numel()
    num_exceed = (diff < atol).sum().item()
    print(f'Number of elements within atol={atol}: {num_exceed}/{num_total} ({num_exceed/num_total*100:.2f}%)')

    # relative tolerance check
    rtol = 0.01
    rel_diff = diff / (torch.abs(ref_lm_head_logits) + 1e-6)
    num_exceed_rel = (rel_diff < rtol).sum().item()
    print(f'Number of elements within rtol={rtol}: {num_exceed_rel}/{num_total} ({num_exceed_rel/num_total*100:.2f}%)')  
    
    # top-k match statistics
    topk_stats = topk_match_rate(outputs.logits, ref_lm_head_logits, k=5)
    print(f'Top-5 match statistics:')
    for key, value in topk_stats.items():
        print(f'  {key}: {value}')

    print("****** END ******")
    