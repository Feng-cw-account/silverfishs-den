#math/algebra/abstract 
## pre-define
$$ 
\mathbf{S} = \mathbf{Z}a + \mathbf{Z}b = \{n \in \mathbf{Z} \mid n = ra + sb \quad \text{for some integers r, s}\}
$$
## prove S is Subgroup of Z
1. $\mathbf{S} \in \mathbf{Z}$
2. Closure: $$ \left \{ \begin{align} q_1 = r_1a+s_1b \in \mathbf{S}, \\ q_2 = r_2a + s_2b \in \mathbf{S}  \ \end{align} \right .  \quad\to q = q_1 + q_2 = (r_1 + r_2)a + (s_1 + s_2)b \in \mathbf{S} $$
3. Identity: $0 \in \mathbf{S}$
4. Inverses: if $q = ra + sb , q \in \mathbf{S}, \text{then} -q = -ra -sb, -q \in \mathbf{S}$

## conclusion
**therefore**, 
based on [[The Cyclic Decomposition Theorem for Subgroups of Addition Group]],
we know that:
$$\exists d, s.t. \mathbf{Z}d = \mathbf{Z}a + \mathbf{Z}b$$
and:$$
	d = \text{gcd}(a,b) \quad \text{\\\\why d is the gdc?}
	$$
then:$$
	\begin{align}
	&\text{d divides a and b} \\
	&\text{if an integer e divides both a and b, it also divides d} \\
	&\text{There are integers r and s such that} \quad d = ra + sb
	\end{align}
	$$


## why d is the gcd
### d is divisor
$a = k_1 d$, $b = k_2 d$
## e divides d
$e$ is any divisor of $a,b$.
$d \in \mathbf{S}$, so $d=ra+sb$
so $e$ divides $d$
### conclusion
$e$ and $d$ are positive, so $e<d$
so d is gcd.

[[how to prove a element is greatest one in a set]]