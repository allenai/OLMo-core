from typing import Dict, List

# For training runs where we don't expect the model to acquire MC (e.g., 1B-5xC, short 7B training runs)
FULL_TASKS_SMALL_COMPUTE = [
    # OLMES Core 9(-ish) RC
    "arc_challenge_test_rc_5shot",
    "arc_easy_test_rc_5shot",
    "hellaswag_rc_5shot",  # 1K subset of HellaSwag
    "winogrande_val_rc_5shot",  # Helpful after 750M-5xC scale
    "csqa_val_rc_5shot",
    "piqa_val_rc_5shot",
    "socialiqa_val_rc_5shot",
    # Too noisy to be worth tracking
    # "boolq_val_rc_5shot",
    # "openbookqa_test_rc_5shot",
    # MMLU RC
    "mmlu_stem_val_rc_5shot",
    "mmlu_humanities_val_rc_5shot",
    "mmlu_social_sciences_val_rc_5shot",
    "mmlu_other_val_rc_5shot",
    "mmlu_stem_test_rc_5shot",
    "mmlu_humanities_test_rc_5shot",
    "mmlu_social_sciences_test_rc_5shot",
    "mmlu_other_test_rc_5shot",
    # Gen tasks BPB
    "gsm8k_gold_bpb_5shot",
    "minerva_math_algebra_gold_bpb_0shot",
    "minerva_math_counting_and_probability_gold_bpb_0shot",
    "minerva_math_geometry_gold_bpb_0shot",
    "minerva_math_intermediate_algebra_gold_bpb_0shot",
    "minerva_math_number_theory_gold_bpb_0shot",
    "minerva_math_prealgebra_gold_bpb_0shot",
    "minerva_math_precalculus_gold_bpb_0shot",
    "codex_humaneval_gold_bpb_3shot",
    "codex_mbpp_gold_bpb_3shot",
    # MT MBPP tasks
    "mt_mbpp_rust_gold_bpb_3shot",
    "mt_mbpp_java_gold_bpb_3shot",
    "mt_mbpp_cpp_gold_bpb_3shot",
    # Sanity check for MCQA ability
    "copycolors_10way_fast",
    # Basic Skills
    "basic_skills_arithmetic_rc_5shot",
    "basic_skills_coding_rc_5shot",
    "basic_skills_common_knowledge_rc_5shot",
    "basic_skills_logical_reasoning_rc_5shot",
    "basic_skills_pattern_rc_5shot",
    "basic_skills_string_operations_rc_5shot",
]

# For training runs where we expect the model to acquire MC
FULL_TASKS_LARGE_COMPUTE = [
    # OLMES Core 9(-ish) MC
    "arc_challenge_test_mc_5shot_fast",
    "arc_easy_test_mc_5shot_fast",
    "hellaswag_rc_5shot",  # 1K subset of HellaSwag
    "csqa_val_mc_5shot_fast",
    "piqa_val_mc_5shot_fast",
    "socialiqa_val_mc_5shot_fast",
    "winogrande_val_rc_5shot",
    # Too noisy to be worth tracking
    # "boolq_val_mc_5shot_fast",
    # "openbookqa_test_mc_5shot_fast",
    # MMLU MC BPB
    "mmlu_stem_val_mc_5shot_fast",
    "mmlu_humanities_val_mc_5shot_fast",
    "mmlu_social_sciences_val_mc_5shot_fast",
    "mmlu_other_val_mc_5shot_fast",
    "mmlu_stem_test_mc_5shot_fast",
    "mmlu_humanities_test_mc_5shot_fast",
    "mmlu_social_sciences_test_mc_5shot_fast",
    "mmlu_other_test_mc_5shot_fast",
    # Gen tasks BPB
    "gsm8k_gold_bpb_5shot",
    "minerva_math_500_gold_bpb_0shot",
    "minerva_math_algebra_gold_bpb_0shot",
    "minerva_math_counting_and_probability_gold_bpb_0shot",
    "minerva_math_geometry_gold_bpb_0shot",
    "minerva_math_intermediate_algebra_gold_bpb_0shot",
    "minerva_math_number_theory_gold_bpb_0shot",
    "minerva_math_prealgebra_gold_bpb_0shot",
    "minerva_math_precalculus_gold_bpb_0shot",
    "codex_humaneval_gold_bpb_3shot",
    "codex_mbpp_gold_bpb_3shot",
    # MT MBPP tasks
    "mt_mbpp_rust_gold_bpb_3shot",
    "mt_mbpp_java_gold_bpb_3shot",
    "mt_mbpp_cpp_gold_bpb_3shot",
    # Sanity check for MCQA ability
    "copycolors_10way_fast",
    # Basic Skills
    "basic_skills_arithmetic_rc_5shot",
    "basic_skills_coding_rc_5shot",
    "basic_skills_common_knowledge_rc_5shot",
    "basic_skills_logical_reasoning_rc_5shot",
    "basic_skills_pattern_rc_5shot",
    "basic_skills_string_operations_rc_5shot",
]

# Unfortunately we need the same metrics for everything, so we run them all.
FULL_TASKS = list(set(FULL_TASKS_SMALL_COMPUTE + FULL_TASKS_LARGE_COMPUTE))

# Subset of "full" task set that is roughly 2-3x faster
FAST_TASKS = [
    # Subset of OLMES
    "arc_challenge_test_bpb_5shot",
    "arc_challenge_test_mc_5shot_fast",
    "arc_easy_test_bpb_5shot",
    "arc_easy_test_mc_5shot_fast",
    "hellaswag_bpb_5shot",
    "mmlu_humanities_test_bpb_5shot",
    "mmlu_humanities_test_mc_5shot_fast",
    "mmlu_other_test_bpb_5shot",
    "mmlu_other_test_mc_5shot_fast",
    "mmlu_social_sciences_test_bpb_5shot",
    "mmlu_social_sciences_test_mc_5shot_fast",
    "mmlu_stem_test_bpb_5shot",
    "mmlu_stem_test_mc_5shot_fast",
    # Basic Skills
    "basic_skills_arithmetic_rc_5shot",
    "basic_skills_coding_rc_5shot",
    "basic_skills_common_knowledge_rc_5shot",
    "basic_skills_logical_reasoning_rc_5shot",
    "basic_skills_pattern_rc_5shot",
    "basic_skills_string_operations_rc_5shot",
    # Gen tasks BPB
    "codex_humaneval_gold_bpb_3shot",
    "codex_mbpp_gold_bpb_3shot",
    "minerva_math_500_gold_bpb_0shot",
    "mt_mbpp_cpp_gold_bpb_3shot",
    "mt_mbpp_java_gold_bpb_3shot",
    "mt_mbpp_rust_gold_bpb_3shot",
    # Sanity check for MCQA ability
    "copycolors_10way_fast",
]


TASK_GROUPS: Dict[str, List[str]] = {
    "full__small_compute": FULL_TASKS_SMALL_COMPUTE,
    "full__large_compute": FULL_TASKS_LARGE_COMPUTE,
    "full": FULL_TASKS,
    "fast": FAST_TASKS,
}
