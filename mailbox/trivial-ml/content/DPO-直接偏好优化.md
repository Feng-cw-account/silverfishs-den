---
tags:
  - type/permanent
  - attr/technique
  - cs/llm
  - ml/rl-alignment
---

# DPO - 直接偏好优化 (Direct Preference Optimization)

## 定义
一种无需奖励模型的偏好优化方法。直接从偏好对数据中学习，通过最大化 preferred response 与 dispreferred response 的对数概率差。

## 核心优点（相比 [[PPO]]）
- **无需奖励模型**：省去昂贵的奖励模型训练
- **训练步骤少**：SFT → DPO，而非 SFT → 奖励模型 → PPO
- **计算资源少**：仅需一个模型，不需多模型协同
- **收敛更快**：直接优化偏好信号

## 数学原理
给定偏好对 $(y_w, y_l)$（preferred, dispreferred），DPO 损失为：
$$\mathcal{L}_{DPO} = -\log \sigma\left( \beta \log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$$

其中 $\beta$ 是温度参数，通常设为 0.5 ~ 1.0。

## 与 [[PPO]] 的详细对比

| 指标 | PPO | DPO |
|------|-----|-----|
| 需要奖励模型 | 是 | 否 |
| 训练步骤数 | 多（SFT → 奖励 → PPO） | 少（SFT → DPO） |
| 计算资源 | 高（多模型） | 中等 |
| 收敛速度 | 较慢 | 更快 |
| 数据需求 | 大（10k+） | 中等 |
| 理论基础 | 强化学习 | 对比学习 |

## 数据格式
```
{
  "prompt": "...",
  "chosen": "...",      # preferred response
  "rejected": "..."     # dispreferred response
}
```

## 应用场景
- 数据充足但标注预算有限
- 需要快速迭代原型
- 对齐效果良好且成本低
- 与 [[SFT]] 结合效果显著

## 与其他方法的关系
- 改进于 [[PPO]]（去除奖励模型）
- 启发了 [[KTO]] 的设计
- 可与 [[LoRA]] 或 [[OFT]] 结合

## 注记
DPO 近年成为开源社区的标准对齐方法，相比 [[PPO]] 更易实现、成本更低。
