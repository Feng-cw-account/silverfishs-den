---
tags:
  - type/permanent
  - attr/technique
  - cs/llm
  - ml/peft
---

# LoRA - 低秩适配 (Low-Rank Adaptation)

## 定义
一种 [[PEFT|参数高效微调]] 方法。通过在原权重旁添加可学习的低秩矩阵对 $\Delta W = AB^T$ 来快速适配任务，其中 $A, B$ 的秩 $r \ll d$（模型隐藏维度）。

## 核心原理
$$\text{新权重} = \text{原权重} + \frac{\alpha}{r} \times A_{r \times d} \times B_{d \times r}^T$$

其中 $\alpha$ 是缩放因子，通常设为学习率的倍数。

## 关键优点
- **参数高效**：仅需微调 0.5~3% 的参数（e.g. 7B 模型 → 4M 可训练参数）
- **速度快**：显存占用少，训练速度快 5-10 倍
- **模块化**：可为不同任务训练多个 LoRA 适配器，灵活切换
- **防灾难性遗忘**：原权重不变，微调隐藏在低秩空间

## 超参配置
- `rank (r)`: 通常 8 ~ 64，平衡性能与参数量
- `lora_alpha`: 缩放因子，通常为秩的 2 倍
- `target_modules`: 选择要添加 LoRA 的层（常见：q_proj, v_proj, k_proj, o_proj, up_proj, down_proj）
- `lora_dropout`: 0.05 ~ 0.1

## 与其他方法的比较
- 比 [[SFT|全量微调]] 参数量少 100 倍，速度快 10 倍
- 比 [[OFT|正交微调]] 更易实现，框架支持更广泛
- 与 [[参数冻结策略]] 天然兼容

## 适用场景
- 显存有限
- 需要快速原型迭代
- 多任务场景（每个任务一个 LoRA 适配器）

## 注记
LoRA 的秩越高，参数越多，但也更容易过拟合。建议从秩 8 开始逐步增加。
