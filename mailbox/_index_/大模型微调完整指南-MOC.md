---
tags:
  - type/moc
  - cs/llm
  - ml/fine-tuning
---

# 大模型微调完整指南 (MOC)

这是关于 LLM 微调的知识索引与导航地图。详细内容见 [[LLM-fine-tuning]]。

## 快速导航

### 📌 核心微调方法（7 大技术）

#### 全量微调
- **[[SFT - 监督式微调|SFT]]**：最直接的方法，更新所有参数。性能最高，成本最高。见 [[LLM-fine-tuning#1-监督式微调-sft--supervised-fine-tuning]]

#### 参数高效微调 (PEFT)
- **[[PEFT - 参数高效微调|PEFT]]**：总体框架与分类
  - **[[LoRA - 低秩适配|LoRA]]**：添加低秩矩阵，最常用，易实现。见 [[LLM-fine-tuning#2-lora-low-rank-adaptation--低秩适配]]
  - **[[OFT - 正交微调|OFT]]**：正交约束，数值稳定，参数更少。见 [[LLM-fine-tuning#3-oft-orthogonal-finetuning--正交微调]]

#### 优化与策略
- **[[参数冻结策略|Freeze 参数冻结]]**：选择性冻结层，与各微调方法正交。与 SFT/LoRA/OFT 组合使用，防止灾难性遗忘。见 [[LLM-fine-tuning#4-冻结策略-freezing-strategy]]
  - *场景*：预训练知识重要、任务特异性强、少样本学习

#### 偏好对齐（3 大对齐方法）
- **[[PPO - 近端策略优化|PPO]]**：工业标准，需要奖励模型，成本高。见 [[LLM-fine-tuning#5-ppo-proximal-policy-optimization--近端策略优化]]
- **[[DPO - 直接偏好优化|DPO]]**：改进 PPO，无需奖励模型，开源流行。见 [[LLM-fine-tuning#6-dpo-direct-preference-optimization--直接偏好优化]]
- **[[KTO - 卡尼曼特维斯基优化|KTO]]**：理论改进 DPO，处理不平衡数据更好。见 [[LLM-fine-tuning#7-kto-kahneman-tversky-optimization--卡尼曼-特维斯基优化]]

---

## 微调流程推荐

```
第一阶段：任务适配
├─ 有充足数据 → SFT (全量)
└─ 数据有限/显存紧张 → PEFT (LoRA 或 OFT)

第二阶段：参数优化
├─ 与 SFT 结合 → 添加参数冻结
└─ 与 PEFT 结合 → 已内置高效性

第三阶段：对齐优化（可选）
├─ 有大规模标注数据 → PPO
├─ 中等数据 + 成本敏感 → DPO
└─ 不平衡数据 + 灵活需求 → KTO
```

---

## 关键概念

- **[[灾难性遗忘]]**：微调时遗忘原始知识的风险，各方法都需防范。防止策略见 [[LLM-fine-tuning#防止灾难性遗忘的完整策略]]
- **[[PEFT - 参数高效微调|PEFT]]**：参数占比 0.5~3%，速度快 5-10 倍

---

## 方法对比速查表

### 性能 vs 成本

| 方法 | 参数量 | 显存 | 时间 | 性能 | 学习曲线 |
|------|--------|------|------|------|---------|
| SFT | 100% | 高 | 慢 | 最高 | 最容易 |
| LoRA | 0.5~3% | 低 | 快 | 中等 | 易 |
| OFT | 1~2% | 低 | 快 | 中等 | 中 |
| Freeze | 变量 | 低 | 快 | 中 | 易 |
| PPO | 100% | 极高 | 极慢 | 很高 | 难 |
| DPO | 0~100% | 中 | 中 | 高 | 中 |
| KTO | 0~100% | 中 | 中 | 高 | 中 |

---

## 场景指南

### 数据量指南
- **> 100k 高质数据** → 全量 SFT（可选加 Freeze 保护底层）
- **10k ~ 100k 数据** → PEFT (LoRA/OFT) + **Freeze**（冻结底层）
- **< 10k 数据** → LoRA + **Freeze**（冻结大部分） + 偏好对齐 (DPO/KTO)

### 硬件指南
- **< 8GB GPU** → LoRA + QLoRA（量化）+ **Freeze**
- **8~16GB GPU** → LoRA + 梯度累积 + **Freeze**（冻结底层）
- **16~80GB GPU** → LoRA 或 OFT（可选 Freeze）
- **> 80GB GPU** → SFT 或多任务并行（Freeze 可选）

### Freeze 冻结层的最佳实践
- **冻结底层**（embedding + 前 6-12 层）：保护通用特征，适合任务差异大
- **冻结中层**（前 50%）：平衡性，适合中等微调
- **仅冻结 embedding**：轻量冻结，适合 LoRA+Freeze 组合
- **动态解冻**：先冻结后逐步解冻，适合复杂任务

---

## 防止灾难性遗忘的策略

1. ✅ 使用 PEFT → 原权重不变
2. ✅ **添加 Freeze 参数冻结** → 保护底层特征（最直接有效的方法）
3. ✅ 低学习率 (2e-5 ~ 5e-5) + 早停
4. ✅ 混合通用数据训练
5. ✅ 对齐时使用 KL 约束（PPO/DPO/KTO）

**Freeze 冻结详见** [[LLM-fine-tuning#4-冻结策略-freezing-strategy]]

---

## 常见问题快速查阅

详见 [[LLM-fine-tuning#常见问题速查表]]

**Q: 我只有 8GB GPU，怎么微调 7B 模型？**
A: 使用 LoRA + QLoRA（量化）+ 梯度累积，成本可降低 10 倍。

**Q: 微调后模型变得容易遗忘预训练知识？**
A: 这是灾难性遗忘问题。使用 PEFT 或参数冻结策略。

**Q: 应该用 PPO 还是 DPO？**
A: DPO 成本更低，开源友好。有大规模标注数据用 PPO。

**Q: 数据严重不平衡怎么办？**
A: 用 KTO，理论上比 DPO 更适合。

---

## 推荐学习与实践路径

### 初学者路线
1. 理解 SFT 基础
2. 学习 PEFT 概念，特别是 LoRA
3. 掌握 **Freeze 参数冻结** 技巧（与 SFT/LoRA 组合）
4. 理解灾难性遗忘现象
5. 简单对齐尝试（如 DPO）

### 进阶路线
1. 对比 LoRA 和 OFT 的性能与稳定性
2. 深入 Freeze 策略（何时冻结、冻结多少）
3. 学习 PPO 原理（工业标准）
4. 理解 DPO 和 KTO 的理论差异
5. 在实际项目中应用多种方法组合

### 实践建议
- 从小数据集开始（1-5k 样本）快速测试各方法
- 逐步扩大数据规模，监控过拟合和灾难性遗忘
- **优先尝试组合**：SFT + Freeze，或 LoRA + Freeze
- 使用 LLaMA-Factory 或 Axolotl 加快原型迭代
- 定期在通用基准（MMLU、BLEU 等）上验证性能

---

## 相关资源

- **原子笔记库**：`mailbox/trivial-ml/content/` 中有单个技术的详细解析
- **工具框架**：见 [[LLM-fine-tuning#实用工具与框架-tools--frameworks]]
- **最佳实践**：见 [[LLM-fine-tuning#最佳实践总结-best-practices]]

---

**标签**：#LLM #微调 #PEFT #对齐  
**最后更新**：2025-11-23
