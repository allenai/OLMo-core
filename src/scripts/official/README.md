## OLMo-2 Model Training

Below is a comprehensive table showing the Stage 2 training details for all OLMo 2 32B.

| Model Size | Training | Checkpoint | Monitoring |
|------------|----------|------------|------------|
| **32B** | random seed 1110, 100B tokens | [stage2-ingredient1-step11921-tokens100B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage2-ingredient1-step11921-tokens101B) | coming soon |
|  | random seed 2662, 100B tokens | [stage2-ingredient2-step11921-tokens100B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage2-ingredient2-step11921-tokens101B) | coming soon |
|  | random seed 6209, 100B tokens | [stage2-ingredient3-step11921-tokens100B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient3-step11931-tokens100B) | coming soon |
|  | **Final Souped Model** | [main](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/main) | No config, weights averaged in Python | - |

Note: You can find all the configs and checkpoints for 7B and 13B in the [OLMo](https://github.com/allenai/OLMo) repository.
## Official public training scripts

Please read the scripts carefully before attempting to run them. You may need to adjust hyperparameters based on your hardware.
