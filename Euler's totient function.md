---
tags:
  - type/permanent
  - status/evergreen
  - attr/principle
  - topic/algorithm
---

## Totient
totient is the name of function counting in specific rules.

## define
$$ 
\phi(n) = 

\left \{ 
\begin{align}
& p - 1  &\text{for } &n = p&\\
& p^k - p^{k-1} & \text{for }& n = p^k &\\
& \frac{\phi(p_1) \phi(p_2) d}{\phi(d)} & \text{ for } &n = p_1 p_2, d = \text{gdc}(p_1, p_2)& \\
& \phi(p_1) \phi(p_2)  & \text{ for } &n = p_1 p_2, \text{ where relatively primetive}&

\end{align} 
\right . 

$$
$$  \text{where } p_i \text{ is prime number} $$

so:
$$
\begin{align}
&\phi(n) & = & \phi(p_1^{a_1}) \phi(p_2^{a_2}) \dots \phi(p_n^{a_n})& \\
&& = & n(1-\frac{1}{p_1}) (1 - \frac{1} {p_2})\dots p_n^{a_n} (1 - \frac{1} {p_k}) \\
\end{align}
$$

## property
1. [[Mutiplicative function]]
---
## **related**：
[[Code implement of Euler's totient function]]

[[code that should be print]]