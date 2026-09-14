# Small hero LC → matched SFT sweep

Branch: `codex/small-hero-sft-20260914`, based on the qualified LC branch.
Campaign-specific example files are added, plus a compatibility fix that passes FA4
variable-length boundaries by keyword (FA4 4.0.0b16 inserted a positional argument).
The non-packed PT/LC attention path and original jobs are unchanged.

Six runs: each arm's final LC step5961 × learning rates **1e-5, 5e-5, 1e-4**.
The former proposed2.5e-5 point is replaced by1e-5. Sources, split, seed, batches,
schedule, masks, precision and training duration match between arms.

| Setting | Value |
|---|---|
| Hardware | One8×B300 node per trial; allocated urgent in `ai2/olmo3p5-training` |
| Architecture | Qualified small794,233,472 active /12,496,341,632 total params; per-head QK gains; each arm preserves its EMO setting |
| Sequence / batch |64K packed input tokens /524,288 global tokens; one sequence/rank |
| Duration | Two epochs; exact packed epoch length recorded by the data gate |
| Schedule | Linear decay to zero;3% warmup over the two-epoch horizon |
| Optimizer | Fresh qualified AdamW, weight decay0; no inherited PT/MT/LC moments |
| Loss | Assistant-only masks; global supervised-token normalization; no z-loss |
| Memory / kernels | Eager model execution, block recomputation; FLA0.5.2 packed-document KDA, not the non-varlen custom KDA path |
| Data | `allenai/gptoss120b-deduped`@`f105cae040563c3b43801dbeab356293c146510d` |
| Template | Open Instruct `olmo_thinker_no_think_sft_tokenization`; reasoning and final responses retained |
| Train / validation |106,952 /1,024 prompt-disjoint conversations;474,855,664 /4,427,480 input tokens |
| Saves | Epoch1 and2; synchronous full-state, immutable completion notifications |
| Upload | Existing uploader; distinct SFT prefixes; guarded apply cleanup, keep2,1hgrace |
| Evaluation | Assistant-token-weighted held-out CE before training, every200steps, at both epochs; matched across arms |
| Export | Both epoch checkpoints automatically converted/qualified using the frozen LC converter, exact saved chat template; no raw deletion by workers |

Generation benchmark selection is intentionally not inferred from pretraining's OLMoBase recipes.
That suite is still a separate decision before launching SFT generation evaluations.

## Safety and launch order

The CPU-only Phobos config job prepares packing, checks the real configs and publishes
`data-plan.json`. It requests **no resources**, not even CPU/memory reservations.
The independent CPU controller then runs two four-update qualification jobs, one per arm:
64→8 weights-only transfer with fresh optimizer/data → save step2 → fresh process full-state
resume → step4. It checks all eight ranks, assistant masks, finite train/validation loss,
save invariance, model/buffer samples, RNG and data-loader restoration. Only after both
pass does it submit all six runs. Failed jobs are reported, not blindly resubmitted.

The packed compiled path produced nonfinite initial CE on both sources. An eager
diagnostic of the same EMO batch had finite KDA inputs/outputs and assistant losses
on all eight ranks. SFT therefore disables model compilation; training/restart
smokes must still qualify that execution mode before full trials are submitted.
The early GPU attention test compares packed vs separate-document FA4 and float64
SDPA, checking both forward values and gradients without changing precision.

Packing uses the existing OLMo-core bin packer with the converter's authoritative CSV
boundaries. Original tokenization shards cut one conversation; the local consolidated
source is byte-identical. There are73 literal interior EOS tokens in the content, so
scanning EOS to derive document boundaries would be incorrect. The SFT-specific dataset
uses CSV boundaries for packing and `doc_lens`, retains all tokens/masks, and includes a
focused regression test. Original source artifacts are never overwritten.

Entry points: `olmoe3_hero_sft.py`, `olmoe3_hero_sft_node.py`,
`olmoe3_hero_sft_control.py`, `olmoe3_hero_sft_convert.py`.
The controller uses the existing durable submit-once intent implementation.
Campaign state lives under
`/weka/olmo-3p5-checkpoints/uploader/automation/olmo35-small-gptoss-sft-20260914`.
No output data are saved to Beaker result datasets.
