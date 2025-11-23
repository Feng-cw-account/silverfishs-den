#cs/deep-learning #technique/fine-tuning #LLM #PEFT #RL

## 定义与核心概念

修改 LLM 权重以适应下游任务。通常用于：

- **格式适配**：特定的输入/输出格式
- **知识获取**：新的系统性知识或领域知识（如专业知识库）
- **行为对齐**：通过偏好优化与模型对齐（符合人类价值观）

**核心假设**：LLM 的通用能力已在预训练中获得，微调目标是快速适应特定场景。

---

## 快速导航与方法选择

### 📌 7 大核心微调技术

| 方法 | 参数量 | 显存 | 时间 | 性能 | 推荐场景 |
|------|--------|------|------|------|---------|
| [[SFT]] | 100% | 高 | 慢 | 最高 | 数据充足（>100k） |
| [[LoRA]] | 0.5~3% | 低 | 快 | 中等 | 显存有限、快速迭代 |
| [[OFT]] | 1~2% | 低 | 快 | 中等 | 需要数值稳定性 |
| [[PPO]] | 100% | 极高 | 极慢 | 很高 | 大规模标注数据 |
| [[DPO]] | 0~100% | 中 | 中 | 高 | 中等数据，成本敏感 |
| [[KTO]] | 0~100% | 中 | 中 | 高 | 不平衡数据 |
| [[参数冻结]] | 变 | 低 | 快 | 中 | 与各方法正交，防遗忘 |

### 🎯 场景指南速查

**数据量指南**
- **> 100k 高质数据** → SFT（全量微调）
- **10k ~ 100k 数据** → LoRA/OFT + 参数冻结
- **< 10k 数据** → LoRA + 参数冻结 + 偏好对齐（DPO/KTO）

**硬件指南**
- **< 8GB GPU** → LoRA + QLoRA（量化）+ 梯度累积
- **8~16GB GPU** → LoRA + 梯度累积 + 参数冻结
- **16~80GB GPU** → LoRA 或 OFT
- **> 80GB GPU** → SFT 或多任务并行

**对齐方法选择**
- 有大规模人工标注 → [[PPO]]（工业标准，成本高）
- 中等标注数据 → [[DPO]]（开源流行，成本低）
- 数据严重不平衡 → [[KTO]]（理论改进，处理不平衡更好）

---

## 核心方法论详解 (Core Methodologies)

### 1. 监督式微调 (SFT | Supervised Fine-Tuning)

#### 定义
用有标签的数据集训练 LLM，使其学习特定的任务行为和输出格式。

#### 流程
1. **准备数据**：收集 (instruction, response) 对的高质量数据集
2. **设置损失函数**：通常使用标准交叉熵损失
3. **训练**：在微调数据上迭代更新权重
4. **评估**：在验证集上测试任务相关指标

#### 关键参数
- `learning_rate`: 通常 2e-5 ~ 5e-5（比预训练小得多）
- `batch_size`: 8 ~ 32（取决于显存）
- `epochs`: 2 ~ 5（防止过拟合）
- `warmup_ratio`: 0.1 ~ 0.2

#### 优缺点
**优点**：
- 性能天花板高，能适配各类任务
- 直接、易于理解

**缺点**：
- 更新全部权重，显存与计算量大
- 易发生灾难性遗忘（catastrophic forgetting）

#### 常用实践
```bash
# HuggingFace Transformers 示例
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  train.py \
  --model_name_or_path meta-llama/Llama-2-7b \
  --data_path ./train_data.json \
  --bf16 True \
  --output_dir ./output \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine
```

---

### 2. LoRA (Low-Rank Adaptation | 低秩适配)

#### 定义
一种参数高效微调 (PEFT) 方法，通过在原权重旁添加可学习的低秩矩阵对 $\Delta W = AB^T$ 来快速适配任务，其中 $A, B$ 的秩 $r \ll d$（模型隐藏维度）。

#### 核心原理
```
新权重 = 原权重 + α/r × A(r×d) × B(d×r)^T

其中 α 是缩放因子，通常设为学习率的倍数
```

#### 优点
- **参数高效**：仅需微调 2-3% 的参数（e.g. 7B 模型 → 4M 可训练参数）
- **速度快**：显存占用少，训练速度快 5-10 倍
- **模块化**：可为不同任务训练多个 LoRA 适配器，灵活切换
- **防止灾难性遗忘**：原权重不变，微调隐藏在低秩空间

#### 实现要点
- **秩选择**：通常 8 ~ 64，平衡性能与参数量
- **target_modules**：选择要添加 LoRA 的层（常见：q_proj, v_proj, down_proj）
- **lora_alpha**：缩放因子，通常为秩的 2 倍

#### 调参建议
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# LoRA 配置
lora_config = LoraConfig(
    r=64,                              # 秩
    lora_alpha=128,                    # 缩放因子 = 2 × r
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
model = get_peft_model(model, lora_config)

# 查看可训练参数量
model.print_trainable_parameters()
# 输出: trainable params: 4194304 || all params: 6738415616 || trainable%: 0.06
```

#### 命令示例
```bash
# 使用 LLaMA-Factory 进行 LoRA 微调
python src/train.py \
  --model_name_or_path meta-llama/Llama-2-7b \
  --dataset_path ./data \
  --dataset alpaca \
  --cutoff_len 512 \
  --learning_rate 5e-4 \
  --lr_scheduler_type cosine \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --save_steps 100 \
  --lora_target q_proj,v_proj,k_proj,o_proj,up_proj,down_proj \
  --output_dir ./output
```

---

### 3. OFT (Orthogonal Finetuning | 正交微调)

#### 定义
一种参数高效方法，通过学习正交变换矩阵对权重进行微调。与 LoRA 不同，OFT 在**权重矩阵的行空间**上施加正交约束，保证变换的数值稳定性和几何意义。

#### 与 SFT/LoRA 的区别

| 特性 | SFT | LoRA | OFT |
|------|-----|------|-----|
| 参数量 | 100% | 0.5~3% | 1~2% |
| 可训练参数类型 | 全部权重 | 低秩适配 | 正交变换 |
| 几何约束 | 无 | 秩约束 | 正交约束 |
| 数值稳定性 | 中等 | 中等 | 高（保证不改变权重模 |
| 计算复杂度 | 高 | 低 | 中低 |

#### 原理
原权重 $W$ 经正交变换 $Q$ 映射：
$$W' = W \cdot Q, \quad Q^T Q = I$$

其中 $Q$ 的学习参数远少于 $W$。

#### 适用场景
- **多任务适配**：相比 LoRA，OFT 在某些任务上稳定性更好
- **微调微调**（fine-tuning the fine-tuning）：在已微调模型上再微调
- **对齐敏感任务**：正交约束减少灾难性遗忘

#### 注意点
- 计算正交矩阵的梯度略复杂，部分框架支持有限
- 超参调优空间相比 LoRA 小

---

### 4. 冻结策略 (Freezing Strategy)

#### 定义
在微调时，选择性冻结（不更新）模型的某些层，仅训练其他层。

#### 何时冻结哪些层

| 场景 | 冻结策略 | 理由 |
|------|---------|------|
| 预训练知识重要 | 冻结底层（embedding + 前 6-12 层） | 底层捕捉通用特征，防止遗忘 |
| 任务特定性强 | 仅微调顶层（最后 3-6 层 + head） | 快速适配，参数少 |
| 中等微调（LoRA） | 冻结大部分，LoRA 在 Q/V/K/O 层 | 平衡适配与稳定性 |
| 少样本学习（Few-shot） | 冻结除 head 和最后 1-2 层 | 极少数据无法训练全部权重 |

#### 实践示例
```python
import torch

# 冻结底层
for name, param in model.named_parameters():
    if 'transformer.h.0' in name or 'transformer.h.1' in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# 或使用 peft 库的自动冻结
from peft import LoraConfig
config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    modules_to_save=["embedding"]  # embedding 层不冻结但不加 LoRA
)
```

---

### 5. PPO (Proximal Policy Optimization | 近端策略优化)

#### 定义
一种强化学习算法，通过奖励模型指导 LLM 生成更符合人类偏好的输出。

#### 核心流程
1. **有监督微调** (SFT)：先在 (prompt, response) 数据上训练基础模型
2. **奖励建模**：收集人类标注的 response 偏好对，训练奖励模型 $R(x, y)$
3. **PPO 训练**：
   - 策略模型生成文本
   - 奖励模型评分
   - 计算 PPO 损失，更新策略模型参数
   - 保持与 SFT 模型的 KL 散度约束

#### 优点
- 显式优化人类偏好，效果可验证
- PPO 算法稳定、收敛快

#### 缺点
- 需要大规模人工标注数据（通常 10k+）
- 计算成本高（需同时运行策略模型、奖励模型、价值模型）
- 奖励模型本身可能有偏差

#### 实践命令
```bash
# 使用 TRL (Transformers Reinforcement Learning) 库
python examples/ppo_trainer_example.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name imdb \
  --batch_size 32 \
  --mini_batch_size 4 \
  --learning_rate 1e-5 \
  --output_dir ./ppo_output
```

---

### 6. DPO (Direct Preference Optimization | 直接偏好优化)

#### 定义
一种无需奖励模型的偏好优化方法。直接从偏好对数据中学习，通过最大化 preferred response 与 dispreferred response 的对数概率差。

#### 与 PPO 的对比

| 指标 | PPO | DPO |
|------|-----|-----|
| 需要奖励模型 | 是 | 否 |
| 训练步骤数 | 多（SFT → 奖励模型 → PPO） | 少（SFT → DPO） |
| 计算资源 | 高（需多个模型） | 中等 |
| 收敛速度 | 较慢 | 更快 |
| 数据需求 | 大（人工标注偏好对） | 中等（但需质量高） |

#### 核心公式
给定偏好对 $(y_w, y_l)$（preferred, dispreferred），DPO 损失为：
$$\mathcal{L}_{DPO} = -\log \sigma\left( \beta \log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$$

其中 $\beta$ 通常设为 0.5 ~ 1.0。

#### 实践示例
```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# DPO 配置
dpo_config = DPOConfig(
    beta=0.5,
    learning_rate=5e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    output_dir="./dpo_output"
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()
```

---

### 7. KTO (Kahneman-Tversky Optimization | 卡尼曼-特维斯基优化)

#### 定义
一种基于行为经济学理论的偏好优化方法。利用人类决策中的"损失厌恶"与"参考点依赖"特性，直接优化模型使其更符合人类偏好。

#### 与 DPO 的区别

| 特性 | DPO | KTO |
|------|-----|-----|
| 理论基础 | 信息论（Bradley-Terry 模型） | 行为经济学（前景理论） |
| 损失函数 | 对数概率比 | 加权偏好损失 |
| 数据形式 | 偏好对 $(y_w, y_l)$ | 单独反应 (prompt, response, 好/坏) |
| 灵活性 | 中等 | 高（可独立评价每个样本） |
| 效率 | 中等 | 通常更高效 |

#### 原理
KTO 为每个样本 $(x, y)$ 和标签 $z$ （好/坏）优化：
$$\mathcal{L}_{KTO} = \mathbb{E}_{z=1}\left[w_+ \cdot \ell(v(x,y))\right] + \mathbb{E}_{z=0}\left[w_- \cdot \ell(-v(x,y))\right]$$

其中 $w_+, w_-$ 是从人类数据推导的权重，反映损失厌恶。

#### 适用场景
- **不平衡数据**：好样本与坏样本数量差异大
- **多样性偏好**：用户对好/坏的定义差异大
- **实时反馈**：可增量更新单个样本标签

#### 实践提示
```python
# 假设数据格式：
# {"prompt": "...", "response": "...", "is_good": True/False}

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# 配置 KTO 训练（伪代码，实际需 KTO 库支持）
kto_config = {
    "beta": 0.5,
    "loss_weight_good": 1.0,   # w_+ 权重
    "loss_weight_bad": 1.0,    # w_- 权重，可调整处理不平衡
    "learning_rate": 5e-4
}

# 训练
trainer.train()
```

---

## 实用工具与框架 (Tools & Frameworks)

### 环境与库
- **HuggingFace Transformers**：基础模型加载与训练
- **PEFT**（peft）：LoRA、QLoRA、前缀微调等高效方法
- **TRL**（trl）：PPO、DPO 训练管道
- **Unsloth**：优化的 LoRA 训练（速度快 2-5 倍）
- **bitsandbytes**：8-bit 量化，进一步降低显存

### 集成框架
- **LLaMA-Factory**：一站式微调框架，支持 SFT、LoRA、DPO、PPO
- **Axolotl**：配置驱动的微调工具
- **H2O LLM Studio**：低代码平台，界面友好
- **AutoTrain**：Hugging Face 官方托管服务

### 安装示例
```bash
# 基础依赖
pip install transformers peft torch datasets

# 强化学习与优化
pip install trl

# 加速（可选）
pip install unsloth 
pip install bitsandbytes

# 量化（可选，用于 QLoRA）
pip install bitsandbytes
```

### 快速开始命令行
```bash
# 1. SFT 微调（使用 LLaMA-Factory）
llamafactory-cli train \
  --model_name_or_path meta-llama/Llama-2-7b \
  --data_path ./data/sft_data.json \
  --output_dir ./sft_output \
  --learning_rate 5e-5 \
  --num_train_epochs 3

# 2. LoRA 微调
llamafactory-cli train \
  --model_name_or_path meta-llama/Llama-2-7b \
  --data_path ./data/sft_data.json \
  --output_dir ./lora_output \
  --adapter_name_or_path lora \
  --learning_rate 5e-4 \
  --num_train_epochs 3

# 3. DPO 对齐
llamafactory-cli train \
  --model_name_or_path meta-llama/Llama-2-7b \
  --data_path ./data/dpo_data.json \
  --template dpo \
  --output_dir ./dpo_output \
  --learning_rate 5e-4
```

---

## 最佳实践总结 (Best Practices)

1. **数据质量第一**：高质量数据 > 大量低质数据
2. **循序渐进**：SFT → PEFT (LoRA/OFT) → 偏好优化 (DPO/KTO)
3. **参数选择**：
   - LoRA 秩：从 8 开始，逐步增加到 64；监控过拟合
   - 学习率：SFT 用 2e-5，LoRA 用 5e-4，DPO/KTO 用 5e-4
4. **硬件考虑**：
   - 8GB GPU：使用 LoRA + QLoRA + 梯度累积
   - 16GB GPU：LoRA + 批大小 8~16
   - 80GB GPU：全量 SFT 或多任务并行
5. **评估指标**：定义任务相关指标，分阶段验证效果
6. **防止灾难性遗忘**：始终在部分通用数据上混合训练

---

## 防止灾难性遗忘的完整策略

微调时模型会遗忘原始预训练知识。以下策略可有效防范：

### 核心策略
1. ✅ **使用 PEFT 方法**：LoRA/OFT 通过冻结原权重、仅在低秩空间微调，天然防止灾难性遗忘
2. ✅ **参数冻结策略**：冻结底层（embedding + 前 6-12 层）保留通用特征
3. ✅ **低学习率与正则化**：使用较小的学习率（2e-5 ~ 5e-5）、权重衰减、早停
4. ✅ **混合数据训练**：在微调数据中混入部分通用数据（原预训练数据的子集）
5. ✅ **KL 散度约束**：在 PPO、DPO、KTO 中约束新模型与参考模型的 KL 散度

### 衡量指标
- **在微调数据上的精度**：new task accuracy
- **在原始任务上的精度**：general task accuracy（如 MMLU、HellaSwag）
- **综合指标**：加权平均或 Pareto frontier

---

## 常见问题速查表

**Q: 我只有 8GB GPU，怎么微调 7B 模型？**
A: 使用 LoRA + QLoRA（量化）+ 梯度累积，成本可降低 10 倍。

**Q: 微调后模型变得容易遗忘预训练知识？**
A: 这是灾难性遗忘问题。使用 PEFT 或参数冻结策略。

**Q: 应该用 PPO 还是 DPO？**
A: DPO 成本更低，开源友好。有大规模标注数据用 PPO。

**Q: 数据严重不平衡怎么办？**
A: 用 KTO，理论上比 DPO 更适合处理不平衡数据。

**Q: 如何快速验证微调效果？**
A: 在验证集上测试，建议先用小数据集（1k 样本）快速迭代，后续扩大。

---

## 推荐学习与实践路径

### 初学者路线
1. 理解 SFT 基础（全量微调）
2. 学习 PEFT 概念，特别是 LoRA
3. 掌握参数冻结技巧
4. 理解灾难性遗忘现象

### 进阶路线
1. 对比 LoRA 和 OFT 的性能
2. 学习 PPO 原理（工业标准）
3. 理解 DPO 和 KTO 的理论差异
4. 在实际项目中应用多种方法组合

### 实践建议
- 从小数据集开始（1-5k 样本）快速测试各方法
- 逐步扩大数据规模，监控过拟合和灾难性遗忘
- 使用 LLaMA-Factory 或 Axolotl 加快原型迭代
- 定期在通用基准（MMLU、BLEU 等）上验证性能

---

**标签**：#LLM #微调 #PEFT #对齐  
**最后更新**：2025-11-23