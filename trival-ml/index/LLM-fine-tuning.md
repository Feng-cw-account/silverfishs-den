#computer-science/deep-learning #technique/tmp

## define
modify the weight of LLM, to suit down-stream tasks.usually for specific input/output format, but sometime for new knowledge (a system, rather the some facts).

## Kinds
### 1. reinforce-learning
[[RLHF]]: 
train a LM for reward function
> needs human-generated prefered dataset
> needs finetuned supervised LM
> needs human engaged 
### 2. [[SFT| supervised fine tuining]]
1. [[full-fine-tuning]] | [[FFT]]
	1. pros: high performence ceiling
	2. cons: 100% weight affected
2. 参数高效微调|[[PEFT]]
	1. [[prefix-tuning]]: adding learn-able prefix to emedings
	2. [[prompt-tuning]] : simplified prefix-tuning
	3. [[p-tuning]]: impoved prompt-tuning
	4. [[p-tuning-v2]]
3. [[Reparameterization-based-PEFT]]
	1. [[LoRA]]：adding a adaptor
	2. [[QLoRA]] : quantization model
4. adaptors: kind of old?

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