#math/algebra/abstract 
## pre-define
$$ 
\mathbf{S} = \mathbf{Z}a + \mathbf{Z}b = \{n \in \mathbf{Z} \mid n = ra + sb \quad \text{for some integers r, s}\}
$$
## prove S is Subgroup of Z
1. $\mathbf{S} \in \mathbf{Z}$
2. Closure: $$ \left \{ \begin{align} q_1 = r_1a+s_1b \in \mathbf{S}, \\ q_2 = r_2a + s_2b \in \mathbf{S}  \ \end{align} \right .  \quad\to q = q_1 + q_2 = (r_1 + r_2)a + (s_1 + s_2)b \in \mathbf{S}$$
3. Identity: $0 \in \mathbf{S}$
4. Inverses: if $q = ra + sb , q \in \mathbf{S}, \text{then} -q = -ra -sb, -q \in \mathbf{S}$

## conclusion
**therefore**, 
based on [[The Cyclic Decomposition Theorem for Subgroups of (ℤ, +)]],
we know that:
$$\exists r, s.t. \mathbf{Z}r = \mathbf{Z}a + \mathbf{Z}b$$
and:$$
	r = \text{gcd}(a,b)
	$$
then:$$
	\begin{align}
	&\text{d divides a and b} \\
	&\text{if an integer e divides both a and b, it also divides d} \\
	&\text{There are integers r and s such that} \quad d = ra + sb
	\end{align}
	$$
	
	