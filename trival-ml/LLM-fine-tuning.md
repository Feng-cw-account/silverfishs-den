#computer-science/deep-learning #technique/tmp

## define
modify the weight of LLM, to suit down-stream tasks.usually for specific input/output format, but sometime for new knowledge (a system, rather the some facts).

## Kinds
### 1. reinforce-learning
[[RLHF]]
### 2. [[SFT| supervised fine tuining]]
1. [[full-fine-tuning]] | [[FFT]]
2. 参数高效微调|[[PEFT]]
	1. [[prompt-tuning]]
	2. [[p-tuning]]
	3. [[prefix-tuning]]
3. [[Reparameterization-based-PEFT]]
	1. [[LoRA]]
	2. [[QLoRA]]

## tools
### environment
- hugging face transformers: provides a [[SFT]] environment。
### specific tool framework
- LLaMA-Factory
### low-code pletform
- H2O LLM Studio
- Auto Train
### other
- Unsloth: 