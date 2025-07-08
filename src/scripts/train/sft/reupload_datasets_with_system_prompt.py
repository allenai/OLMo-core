from datasets import load_dataset

datasets = [
    "jacobmorrison/OpenThoughts3-456k-no-cot"
]

for dataset in datasets:
    ds = load_dataset(dataset)
    new_system_prompt = "You are OLMo 2, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions."

    def replace_system_prompt(example):
        # Check if there's a system message
        replaced = False
        for msg in example['messages']:
            if msg['role'] == 'system':
                msg['content'] = new_system_prompt
                replaced = True
                break
        if not replaced:
            # Prepend if no system message exists
            example['messages'].insert(0, {"role": "system", "content": new_system_prompt})
            # print("help!")
        return example

    # Use `map` with `batched=False` to modify each example in-place
    ds = ds.map(replace_system_prompt, desc="Updating system prompts")
    ds.push_to_hub(repo_id=f"jacobmorrison/{dataset.split('/')[-1]}-with-olmo-system-prompt")
    print(f"uploaded jacobmorrison/{dataset.split('/')[-1]}-with-olmo-system-prompt")
    del ds