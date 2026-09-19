# hybridish_sft tooling

Export and eval tooling for the hybridish SFT → CTC-suite comparison.

**The recipe lives with the SFT script, on the branch that can actually train these models:**
`prasann/ctc-sft-hybridish` → `src/scripts/train/hybrid-small-suite/README.md`

That branch carries a copy of this directory, so one checkout runs the whole loop (SFT → export →
self-contain → evaluate → harvest). Train from there, not from `prasann/landmark`: this branch's
`olmo_core` has no `scalable_softmax`, so training here loads a hybrid checkpoint and silently
drops a trained component.

What stays here and only here: the SFT **data builders**, in
`src/scripts/data/hybridish/` (`build_ctc_sft_mix.py` → `convert_ctc_to_sft_completion.py`). You
only need them to change the task roster, the length band, or the tokenizer — the dataset itself is
prebuilt on weka.
