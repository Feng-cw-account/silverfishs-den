## preparation
Let x be a element of **finite order** $n$, *which means $<x>$ has n distinct elements*.
Let $k \in \mathbf{Z}$ and be written as $k = nq + r, q,r \in \mathbf{Z}, \text{ and } r \in [0, n)$

## conclusion
1. $x^k = x^r$ 
2. $x^k = 1$ if and only if $r =0$
3. let $d = \text{gcd}(k,n), \text{order}(x^k) = n/d$

## proof
for **1** :
Obviously, $n$ divides $nq$, so  $x^{nq} = 1$
so $x^k = x^{nq} x^r = x^r$
for **2**:
based on conclusion 1, $x^k = 1 \to x^r = 1$
if $r=0$ then $x^k = 1$ is trivial
if $x^r = 1$ and for all $\mathbf{Z}^+$, the minimal r  let$x^r = 1$ is $n$.
and $r \in [0, n)$ so r is 0.
for **3**：
Let the possible order of $x^k$ is $m$
for $(x^k)^m = 1$, $x^{km} = 1$
so $n$ divides $km$
$n = n^{\prime}d, k = k^{\prime}d$, where $n^{\prime}, k^{\prime}$ is relatively prime
$n^{\prime} \mid k^{\prime}m \to n^{\prime} \mid m$
so, $m = \rho n^{\prime}$ and minimal $m^{opt}$ is $n^{\prime}=\frac{n}{\text{gcd}(k,n)}$
then, we should check $m^{opt}$ do be the minimal one.
let there is a $m^{\prime}$ that is smaller then $m^{opt}$, then
