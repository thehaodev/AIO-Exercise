## Eigenvector and eigenvalues example
Input matrix A = 
$$ \begin{align}\begin{bmatrix} 0.9  & 0.2\\ 0.1 & 0.8 \end{bmatrix}\end{align} $$
### 1. Eigenvalue (λ): det($A$ - λ$I$) = 0
$$
det \begin{align}\begin{bmatrix} 0.9-λ  & 0.2\\ 0.1 & 0.8-λ \end{bmatrix}\end{align} = 0
$$

$$
(0.9-λ)\ast(0.8-λ) - (0.1-λ)\ast(0.2-λ)=0
$$

$$
0.7-1.7λ+λ^2=0 
$$

$$
λ=\begin{align}\begin{bmatrix} 1  & 0.7 \end{bmatrix}\end{align} 
$$

### 2. Eigenvector (v): $Av = λv$
$$
(A-λI)=0
$$

$$
\begin{align}\begin{bmatrix} 0.9-λ  & 0.2\\ 0.1 & 0.8-λ \end{bmatrix}\end{align}v = 0
$$
With: $ λ=\begin{align}\begin{bmatrix} 1  & 0.7 \end{bmatrix}\end{align} 
$

$$
\begin{align}\begin{bmatrix} -0.1  & 0.2\\ 0.1 & -0.2 \end{bmatrix}\end{align}v = 0
$$

or 

$$
\begin{align}\begin{bmatrix} 0.2  & 0.2\\ 0.1 & 0.1 \end{bmatrix}\end{align}v = 0
$$

so
$$
v =\begin{align}\begin{bmatrix} 2  & -1\\ 1 & 1 \end{bmatrix}\end{align}
$$

### 3. Normalize vector: 
$$
v_i=\frac{v_i}{\sqrt{\displaystyle\sum_{1}^nv_i^2}}
$$

$$ 
v=\begin{align}\begin{bmatrix} -0.894  & -0.707\\ 0.447 & 0.707 \end{bmatrix}\end{align}
$$



