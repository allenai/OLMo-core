#include <torch/extension.h>
#include <optional>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<int64_t> rowwise_bf16_mega_moe_forward_config(
    int64_t num_rows,
    int64_t top_k,
    int64_t hidden,
    int64_t intermediate,
    int64_t num_local_experts,
    int64_t num_sms);

torch::Tensor rowwise_bf16_mega_moe_sm100_tma_umma_contract_debug();

torch::Tensor rowwise_bf16_mega_moe_sm100_tma_load_contract_debug(
    torch::Tensor& source);

torch::Tensor rowwise_bf16_mega_moe_sm100_tma_umma_tile_contract_debug(
    torch::Tensor& a,
    torch::Tensor& b);

torch::Tensor rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_debug(
    torch::Tensor& a,
    torch::Tensor& b);

torch::Tensor rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_b_mn_debug(
    torch::Tensor& a,
    torch::Tensor& b);

torch::Tensor rowwise_bf16_mega_moe_forward_plan_debug(
    int64_t num_rows,
    int64_t top_k,
    int64_t hidden,
    int64_t intermediate,
    int64_t num_local_experts,
    int64_t num_sms);

torch::Tensor rowwise_bf16_mega_moe_route_counts_debug(
    torch::Tensor& route_expert_indices,
    int64_t num_local_experts);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_route_pack_debug(
    torch::Tensor& route_expert_indices,
    int64_t num_local_experts);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_route_pack_inputs_debug(
    torch::Tensor& source_input,
    torch::Tensor& route_expert_indices,
    torch::Tensor& probs,
    int64_t num_local_experts);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_peer_route_metadata_debug(
    torch::Tensor& dst_ranks,
    torch::Tensor& dst_rows,
    torch::Tensor& probs,
    int64_t ep_world_size,
    int64_t rank_capacity,
    int64_t static_route_budget);

torch::Tensor rowwise_bf16_mega_moe_peer_window_dispatch_debug(
    torch::Tensor& source_input,
    torch::Tensor& dst_ranks,
    torch::Tensor& dst_rows,
    int64_t ep_world_size,
    int64_t rank_capacity);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_peer_window_combine_debug(
    torch::Tensor& peer_payload,
    torch::Tensor& src_ranks,
    torch::Tensor& src_rows,
    torch::Tensor& probs);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_grouped_gemm_metadata_debug(
    torch::Tensor& route_expert_indices,
    int64_t num_local_experts,
    int64_t block_m);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_grouped_gemm_tile_debug(
    torch::Tensor& route_expert_indices,
    int64_t num_local_experts,
    int64_t block_m,
    int64_t n_tiles);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_combine_debug(
    torch::Tensor& packed_expert_out,
    torch::Tensor& packed_token_topk_indices,
    torch::Tensor& probs);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_w1_wmma_debug(
    torch::Tensor& source_input,
    torch::Tensor& route_expert_indices,
    torch::Tensor& up_gate_weight);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_forward_debug(
    torch::Tensor& source_input,
    torch::Tensor& route_expert_indices,
    torch::Tensor& probs,
    torch::Tensor& up_gate_weight,
    torch::Tensor& down_weight);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_local_persistent_forward_debug(
    torch::Tensor& source_input,
    torch::Tensor& route_expert_indices,
    torch::Tensor& probs,
    torch::Tensor& up_gate_weight,
    torch::Tensor& down_weight);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_local_full_forward_megakernel_debug(
    torch::Tensor& source_input,
    torch::Tensor& route_expert_indices,
    torch::Tensor& probs,
    torch::Tensor& up_gate_weight,
    torch::Tensor& down_weight);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_standard_scheduler_debug(
    torch::Tensor& expert_counts);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug(
    torch::Tensor& route_expert_indices);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_peer_map_debug(
    torch::Tensor& route_expert_indices);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_standard_ep_dispatch_pack_inputs_debug(
    torch::Tensor& source_input,
    torch::Tensor& route_expert_indices);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug(
    torch::Tensor& source_input,
    torch::Tensor& route_expert_indices,
    torch::Tensor& probs,
    torch::Tensor& up_gate_weight,
    torch::Tensor& down_weight);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_umma_debug(
    torch::Tensor& source_input,
    torch::Tensor& route_expert_indices,
    torch::Tensor& probs,
    torch::Tensor& up_gate_weight,
    torch::Tensor& down_weight);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_local_umma_compute_debug(
    torch::Tensor& packed_input,
    torch::Tensor& expert_counts,
    torch::Tensor& up_gate_weight,
    torch::Tensor& down_weight);

torch::Tensor rowwise_bf16_mega_moe_local_umma_compute(
    torch::Tensor& packed_input,
    torch::Tensor& expert_counts,
    torch::Tensor& up_gate_weight,
    torch::Tensor& down_weight);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel_debug(
    torch::Tensor& source_input,
    torch::Tensor& route_expert_indices,
    torch::Tensor& probs,
    torch::Tensor& up_gate_weight,
    torch::Tensor& down_weight);

std::vector<torch::Tensor> rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel(
    torch::Tensor& source_input,
    torch::Tensor& route_expert_indices,
    torch::Tensor& probs,
    torch::Tensor& up_gate_weight,
    torch::Tensor& down_weight);

std::vector<int64_t> rowwise_bf16_mega_moe_standard_ep_workspace_config(
    int64_t num_tokens,
    int64_t hidden,
    int64_t intermediate);

void rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace(
    torch::Tensor& source_input,
    torch::Tensor& gathered_out,
    torch::Tensor& out,
    torch::Tensor& route_expert_indices,
    torch::Tensor& probs,
    torch::Tensor& up_gate_weight,
    torch::Tensor& down_weight,
    torch::Tensor& workspace,
    torch::Tensor& rank_workspace_bases,
    torch::Tensor& global_counts,
    torch::Tensor& global_offsets,
    torch::Tensor& expert_cursors,
    torch::Tensor& packed_route,
    torch::Tensor& route_to_slot,
    torch::Tensor& packed_input,
    torch::Tensor& h,
    torch::Tensor& packed_expert_out,
    torch::Tensor& barrier_state,
    int64_t caller_rank_idx,
    bool use_peer_workspace_bases,
    bool enable_cross_rank_barriers,
    bool rank_local_expert_owner,
    bool use_nvshmem_world_collective,
    const std::optional<torch::Tensor>& w1_up,
    const std::optional<torch::Tensor>& w1_gate,
    bool use_umma_compute);

void rowwise_bf16_mega_moe_standard_ep_forward_persistent(
    torch::Tensor& source_input,
    torch::Tensor& gathered_out,
    torch::Tensor& out,
    torch::Tensor& route_expert_indices,
    torch::Tensor& probs,
    torch::Tensor& up_gate_weight,
    torch::Tensor& down_weight);

void rowwise_bf16_mega_moe_forward_persistent(
    torch::Tensor& source_input,
    torch::Tensor& gathered_out,
    torch::Tensor& out,
    torch::Tensor& route_dst_ranks,
    torch::Tensor& route_dst_rows,
    torch::Tensor& route_expert_indices,
    torch::Tensor& probs,
    torch::Tensor& up_gate_weight,
    torch::Tensor& down_weight,
    torch::Tensor& expert_offsets,
    const std::string& group_name,
    const std::optional<torch::Tensor>& route_done_counts,
    const std::optional<torch::Tensor>& symm_probs,
    bool pre_barrier,
    bool post_barrier);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "rowwise_bf16_mega_moe_forward_config",
      &rowwise_bf16_mega_moe_forward_config,
      "Return the MegaMoE-style BF16 fused forward launch/scheduler config",
      py::arg("num_rows"),
      py::arg("top_k"),
      py::arg("hidden"),
      py::arg("intermediate"),
      py::arg("num_local_experts"),
      py::arg("num_sms"));

  m.def(
      "rowwise_bf16_mega_moe_sm100_tma_umma_contract_debug",
      &rowwise_bf16_mega_moe_sm100_tma_umma_contract_debug,
      "Run an SM100 feature-architecture TMA/UMMA/TMEM primitive contract probe");

  m.def(
      "rowwise_bf16_mega_moe_sm100_tma_load_contract_debug",
      &rowwise_bf16_mega_moe_sm100_tma_load_contract_debug,
      "Run a BF16 TMA tensor-map load contract probe",
      py::arg("source"));

  m.def(
      "rowwise_bf16_mega_moe_sm100_tma_umma_tile_contract_debug",
      &rowwise_bf16_mega_moe_sm100_tma_umma_tile_contract_debug,
      "Run a BF16 TMA-to-UMMA tile contract probe",
      py::arg("a"),
      py::arg("b"));

  m.def(
      "rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_debug",
      &rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_debug,
      "Run a BF16 TMA-to-UMMA tile forward probe and return the FP32 accumulator tile",
      py::arg("a"),
      py::arg("b"));

  m.def(
      "rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_b_mn_debug",
      &rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_b_mn_debug,
      "Run a BF16 TMA-to-UMMA tile forward probe with B in MN-major [K, N] layout",
      py::arg("a"),
      py::arg("b"));

  m.def(
      "rowwise_bf16_mega_moe_forward_plan_debug",
      &rowwise_bf16_mega_moe_forward_plan_debug,
      "Run CUDA F1/F2 BF16 MegaMoE task-contract kernels and return debug counters",
      py::arg("num_rows"),
      py::arg("top_k"),
      py::arg("hidden"),
      py::arg("intermediate"),
      py::arg("num_local_experts"),
      py::arg("num_sms"));

  m.def(
      "rowwise_bf16_mega_moe_route_counts_debug",
      &rowwise_bf16_mega_moe_route_counts_debug,
      "Run the CUDA F1 route-counting phase and return per-local-expert counts",
      py::arg("route_expert_indices"),
      py::arg("num_local_experts"));

  m.def(
      "rowwise_bf16_mega_moe_route_pack_debug",
      &rowwise_bf16_mega_moe_route_pack_debug,
      "Run CUDA F1 route count/prefix/pack phases",
      py::arg("route_expert_indices"),
      py::arg("num_local_experts"));

  m.def(
      "rowwise_bf16_mega_moe_route_pack_inputs_debug",
      &rowwise_bf16_mega_moe_route_pack_inputs_debug,
      "Run CUDA F1 route count/prefix/pack plus BF16 input/prob packing",
      py::arg("source_input"),
      py::arg("route_expert_indices"),
      py::arg("probs"),
      py::arg("num_local_experts"));

  m.def(
      "rowwise_bf16_mega_moe_peer_route_metadata_debug",
      &rowwise_bf16_mega_moe_peer_route_metadata_debug,
      "Build generic peer-window route records/counts/offsets for BF16 wave EP bring-up",
      py::arg("dst_ranks"),
      py::arg("dst_rows"),
      py::arg("probs"),
      py::arg("ep_world_size"),
      py::arg("rank_capacity"),
      py::arg("static_route_budget") = 0);

  m.def(
      "rowwise_bf16_mega_moe_peer_window_dispatch_debug",
      &rowwise_bf16_mega_moe_peer_window_dispatch_debug,
      "Dispatch BF16 rows into a generic [ep_world_size, rank_capacity, hidden] peer-window payload",
      py::arg("source_input"),
      py::arg("dst_ranks"),
      py::arg("dst_rows"),
      py::arg("ep_world_size"),
      py::arg("rank_capacity"));

  m.def(
      "rowwise_bf16_mega_moe_peer_window_combine_debug",
      &rowwise_bf16_mega_moe_peer_window_combine_debug,
      "Gather and weighted-combine BF16 rows from a generic peer-window payload",
      py::arg("peer_payload"),
      py::arg("src_ranks"),
      py::arg("src_rows"),
      py::arg("probs"));

  m.def(
      "rowwise_bf16_mega_moe_grouped_gemm_metadata_debug",
      &rowwise_bf16_mega_moe_grouped_gemm_metadata_debug,
      "Build CUDA grouped-GEMM metadata from route expert ids",
      py::arg("route_expert_indices"),
      py::arg("num_local_experts"),
      py::arg("block_m"));

  m.def(
      "rowwise_bf16_mega_moe_grouped_gemm_tile_debug",
      &rowwise_bf16_mega_moe_grouped_gemm_tile_debug,
      "Build grouped-GEMM metadata and run the CUDA tile scheduler contract",
      py::arg("route_expert_indices"),
      py::arg("num_local_experts"),
      py::arg("block_m"),
      py::arg("n_tiles"));

  m.def(
      "rowwise_bf16_mega_moe_combine_debug",
      &rowwise_bf16_mega_moe_combine_debug,
      "Run CUDA F2 packed-row scatter and top-k weighted reduce",
      py::arg("packed_expert_out"),
      py::arg("packed_token_topk_indices"),
      py::arg("probs"));

  m.def(
      "rowwise_bf16_mega_moe_w1_wmma_debug",
      &rowwise_bf16_mega_moe_w1_wmma_debug,
      "Run CUDA BF16 tensor-core W1 grouped GEMM over packed expert routes",
      py::arg("source_input"),
      py::arg("route_expert_indices"),
      py::arg("up_gate_weight"));

  m.def(
      "rowwise_bf16_mega_moe_forward_debug",
      &rowwise_bf16_mega_moe_forward_debug,
      "Run staged CUDA BF16 MegaMoE forward debug path: pack, W1, SwiGLU, W2, combine",
      py::arg("source_input"),
      py::arg("route_expert_indices"),
      py::arg("probs"),
      py::arg("up_gate_weight"),
      py::arg("down_weight"));

  m.def(
      "rowwise_bf16_mega_moe_local_persistent_forward_debug",
      &rowwise_bf16_mega_moe_local_persistent_forward_debug,
      "Run local one-launch persistent CUDA BF16 MegaMoE forward debug compute path",
      py::arg("source_input"),
      py::arg("route_expert_indices"),
      py::arg("probs"),
      py::arg("up_gate_weight"),
      py::arg("down_weight"));

  m.def(
      "rowwise_bf16_mega_moe_local_full_forward_megakernel_debug",
      &rowwise_bf16_mega_moe_local_full_forward_megakernel_debug,
      "Run local single-launch CUDA BF16 MegaMoE forward debug path including route pack and combine",
      py::arg("source_input"),
      py::arg("route_expert_indices"),
      py::arg("probs"),
      py::arg("up_gate_weight"),
      py::arg("down_weight"));

  m.def(
      "rowwise_bf16_mega_moe_standard_scheduler_debug",
      &rowwise_bf16_mega_moe_standard_scheduler_debug,
      "Run the standard-shape OLMo BF16 MegaMoE wave scheduler contract probe",
      py::arg("expert_counts"));

  m.def(
      "rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug",
      &rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug,
      "Run the standard-shape OLMo BF16 MegaMoE EP dispatch metadata contract probe",
      py::arg("route_expert_indices"));

  m.def(
      "rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_peer_map_debug",
      &rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_peer_map_debug,
      "Run standard-shape EP dispatch metadata through a rank workspace pointer map",
      py::arg("route_expert_indices"));

  m.def(
      "rowwise_bf16_mega_moe_standard_ep_dispatch_pack_inputs_debug",
      &rowwise_bf16_mega_moe_standard_ep_dispatch_pack_inputs_debug,
      "Pack BF16 source rows from standard EP dispatch workspaces",
      py::arg("source_input"),
      py::arg("route_expert_indices"));

  m.def(
      "rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug",
      &rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug,
      "Run standard-shape BF16 wave compute using EP dispatch workspace-packed inputs",
      py::arg("source_input"),
      py::arg("route_expert_indices"),
      py::arg("probs"),
      py::arg("up_gate_weight"),
      py::arg("down_weight"));

  m.def(
      "rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_umma_debug",
      &rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_umma_debug,
      "Run standard-shape SM100 UMMA BF16 wave compute using EP dispatch workspace-packed inputs",
      py::arg("source_input"),
      py::arg("route_expert_indices"),
      py::arg("probs"),
      py::arg("up_gate_weight"),
      py::arg("down_weight"));

  m.def(
      "rowwise_bf16_mega_moe_local_umma_compute_debug",
      &rowwise_bf16_mega_moe_local_umma_compute_debug,
      "Run local-expert SM100 UMMA BF16 W1/SwiGLU/W2 compute on an already grouped rank-local buffer",
      py::arg("packed_input"),
      py::arg("expert_counts"),
      py::arg("up_gate_weight"),
      py::arg("down_weight"));

  m.def(
      "rowwise_bf16_mega_moe_local_umma_compute",
      &rowwise_bf16_mega_moe_local_umma_compute,
      "Run local-expert SM100 UMMA BF16 W1/SwiGLU/W2 compute on an already grouped rank-local buffer",
      py::arg("packed_input"),
      py::arg("expert_counts"),
      py::arg("up_gate_weight"),
      py::arg("down_weight"));

  m.def(
      "rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel_debug",
      &rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel_debug,
      "Run standard-shape BF16 EP dispatch, wave compute, scatter, and combine in one debug megakernel",
      py::arg("source_input"),
      py::arg("route_expert_indices"),
      py::arg("probs"),
      py::arg("up_gate_weight"),
      py::arg("down_weight"));

  m.def(
      "rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel",
      &rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel,
      "Run standard-shape BF16 EP dispatch, wave compute, scatter, and combine in one fused megakernel",
      py::arg("source_input"),
      py::arg("route_expert_indices"),
      py::arg("probs"),
      py::arg("up_gate_weight"),
      py::arg("down_weight"));

  m.def(
      "rowwise_bf16_mega_moe_standard_ep_forward_persistent",
      &rowwise_bf16_mega_moe_standard_ep_forward_persistent,
      "Run standard-shape BF16 EP fused forward and fill caller-provided gathered/out tensors",
      py::arg("source_input"),
      py::arg("gathered_out"),
      py::arg("out"),
      py::arg("route_expert_indices"),
      py::arg("probs"),
      py::arg("up_gate_weight"),
      py::arg("down_weight"));

  m.def(
      "rowwise_bf16_mega_moe_standard_ep_workspace_config",
      &rowwise_bf16_mega_moe_standard_ep_workspace_config,
      "Return reusable scratch tensor sizes for the standard-shape BF16 EP fused forward",
      py::arg("num_tokens"),
      py::arg("hidden"),
      py::arg("intermediate"));

  m.def(
      "rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace",
      &rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace,
      "Run standard-shape BF16 EP fused forward with caller-provided reusable scratch tensors",
      py::arg("source_input"),
      py::arg("gathered_out"),
      py::arg("out"),
      py::arg("route_expert_indices"),
      py::arg("probs"),
      py::arg("up_gate_weight"),
      py::arg("down_weight"),
      py::arg("workspace"),
      py::arg("rank_workspace_bases"),
      py::arg("global_counts"),
      py::arg("global_offsets"),
      py::arg("expert_cursors"),
      py::arg("packed_route"),
      py::arg("route_to_slot"),
      py::arg("packed_input"),
      py::arg("h"),
      py::arg("packed_expert_out"),
      py::arg("barrier_state"),
      py::arg("caller_rank_idx") = 0,
      py::arg("use_peer_workspace_bases") = false,
      py::arg("enable_cross_rank_barriers") = false,
      py::arg("rank_local_expert_owner") = false,
      py::arg("use_nvshmem_world_collective") = false,
      py::arg("w1_up") = std::nullopt,
      py::arg("w1_gate") = std::nullopt,
      py::arg("use_umma_compute") = false);

  m.def(
      "rowwise_bf16_mega_moe_forward_persistent",
      &rowwise_bf16_mega_moe_forward_persistent,
      "Production BF16 MegaMoE persistent fused forward entry point; supports constrained local CUDA BF16 forward and fails closed for unsupported peer-window modes",
      py::arg("source_input"),
      py::arg("gathered_out"),
      py::arg("out"),
      py::arg("route_dst_ranks"),
      py::arg("route_dst_rows"),
      py::arg("route_expert_indices"),
      py::arg("probs"),
      py::arg("up_gate_weight"),
      py::arg("down_weight"),
      py::arg("expert_offsets"),
      py::arg("group_name"),
      py::arg("route_done_counts") = std::nullopt,
      py::arg("symm_probs") = std::nullopt,
      py::arg("pre_barrier") = true,
      py::arg("post_barrier") = false);
}
