# 标签系统

超简化的两层标签：

## 一级标签（笔记类型，灵活选择）

可根据笔记内容选择一个或多个：

| 标签 | 用途 | 示例 |
|-----|------|------|
| `#concept` | 概念/定义/理论 | Group, Process, Algorithm |
| `#principle` | 原理/定律/规则 | Law of Gauss, TCP/IP |
| `#algo` | 算法/步骤/方法 | Paging, FFT, Gradient Descent |
| `#technique` | 技术/实现/工程 | Attention, PEFT, RLHF |
| `#history` | 历史/演进/背景 | History of Attention |
| `#proof` | 证明/推导 | Mathematical Proof |

**灵活使用示例：**
- 既是概念又是技术？→ `#concept #technique`
- 既是算法又是原理？→ `#algo #principle`

## 二级标签（知识领域）

| 标签 | 领域 |
|-----|------|
| `#math` | 数学 |
| `#cs` | 计算机 |
| `#physics` | 物理 |
| `#ml` | 机器学习 |

---

## 快速示例

根据笔记内容选择合适的标签组合：

- 数学概念 → `#concept #math`
- 物理定律 → `#principle #physics`
- 计算机算法 → `#algo #cs`
- 机器学习技术 → `#technique #ml`
- 技术原理 → `#principle #technique #cs`
- 算法证明 → `#algo #proof #math`



## 添加标签

在文件顶部一行添加：

```
#concept #math
#algo #cs
#principle #physics
```

就这样。