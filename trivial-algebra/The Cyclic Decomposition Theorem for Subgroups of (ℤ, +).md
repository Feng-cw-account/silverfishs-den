#math/algebra/abstract
## conclusion
subgroup of $(\mathbf{Z},+), \quad \mathbf{S}$ is always in shape of $\mathbf{Z} a$   , if the subgroup is not the trivial subset $\{0\}$ .
## general strategy
let $S$ be a subgroup of $\mathbf{Z}$ , where $a$ is the smallest positive integer.
let $\mathbf{Z}a = \{n \in Z \mid n = ka \quad \text{for some k in }\mathbf{Z}\}$.

1. if $S$ is not **trivial**, then at least a positive integer must in $S$
	1. > make sure a exists.
2. shows $\mathbf{Z}a \in S$
	1. > all subsets of $\mathbf{Z}a$ is in $\mathbf{S}$
	2. for identity: $0a = 0 \in \mathbf{S}$
	3. for closure: $a \in \mathbf{S} \to (a + \dots + a) \in \mathbf{S}$
	4. $a \in \mathbf{S} \to -a \in \mathbf{S}$
4. shows $S \in \mathbf{Z}a$
	1. **sehr tricky** 
	2. each $n \in \mathbf{S}$ in $\mathbf{Z}a$
		1. generalization: let $n = qa + r$
		2. and then prove $r = 0$ and only $=0$
5. then we know a **not trivial** $S$ always equals to a $\mathbf{Z}a$
6. 

## provement:
1. [[normal form of provement of sets' equality]]