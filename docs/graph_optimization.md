### Graph Optimization  
#### What is graph?  
A graph is a pair $G = (V, E)$,
where $V$ is a set of nodes, each of which contains some parameters to be optimized.  $E$ is a set of connected information, whose elements are denotes the constraint relationship between two nodes.  
Many robotics and computer vision problems can be represented by a graph problem.

#### How to solve graph problem?
A graph problem can be defined as a nonlinear least squares problems.
$f_{ij}(v_i, v_j; e_{ij})$ shows the constraint relationship between node $v_i$ and $v_j$
$e_{ij}$ is the prior error of $v_i$ and $v_j$.  
$$ 
F(V) = \sum_{\{i,j\} \in E}f_{ij}(v_i, v_j; e_{ij})^2 \quad
$$

We need to find a optimal set of nodes (i.e. $V$) to minimize the overall cost. 
According to [guass_newton_method.md](./guass_newton_method.md), 
as soon as we can compute the hessian matrix $H$ and gradient $g$, we can solve this graph optimization problem.

#### The hessian matrix $H$
We note that the size of the hessian matrix will be very large,
since there are many parameters for $F$.  
The hessian matrix of $f_{ij}$ can be show as:
$$H_{ij} = 
\begin{bmatrix}
... & ...       & ... & ...       & ... \\  
... & J_i^T J_i & ... & J_i^T J_j & ... \\  
... & ...       & ... & ...       & ... \\  
... & J_j^T J_i & ... & J_j^T J_j & ... \\  
... & ...       & ... & ...       & ... \\  
\end{bmatrix}$$

The $J_i^T J_i$ is located in row i column i of $H_{ij}$  
The $J_j^T J_j$ is located in row j column j of $H_{ij}$  
The $J_i^T J_j$ is located in row i column j of $H_{ij}$  
The $J_j^T J_i$ is located in row j column i of $H_{ij}$  

The overall hessian matrix of F is:

$$ H = \sum_{ \{i,j\} \in E}{H_{ij}} $$

#### The gradient $g$

The gradient vector of $f_{ij}$ can be show as:

$$g_{ij} = 
\begin{bmatrix}
... \\
J_i^T r_i \\
... \\
J_j^T r_n \\
... \\
\end{bmatrix}$$

The $J_i^T r_i$ is located in row i of $g_{ij}$  
The $J_j^T r_j$ is located in row j of $g_{ij}$  

The overall gradient vector of F is:

$$ g = \sum_{\{i,j\} \in E}{g_{ij}} $$

### Derivative of edge between two SO3
Suppose $\varphi$ is an smooth mapping between two Lie Groups,
we can define the derivative of $\varphi$ as $J$:

$$
    \exp(\widehat{J\delta}) = \varphi(x)^{-1}\varphi(x\oplus\delta)
$$

$x$ is a the parameter of $\varphi$, and $\delta$ is a small increment to $x$.

The SO3 edge can define as:
$$
    \varphi(A,B) = Z^{-1}A^{{-1}}B
$$

Where $A$ and $B$ are the two SO3 of node_a and node_b. The $Z$ represents the relative pose of $A$, $B$,
which usually measured by odometry or loop-closing.


$$
    \exp(\widehat{J_A\delta}) = (Z^{-1}A^{{-1}}B)^{-1}(Z^{-1}(A\exp(\hat{\delta}))^{{-1}}B) \\
    = B^{-1}AZ Z^{-1}\exp(-\hat{\delta})A^{{-1}}B \\
    = B^{-1}A\exp(-\hat{\delta})A^{{-1}}B \\
    = -\exp(B^{-1}A \hat{\delta} A^{{-1}}B) \\
    = -\exp(\widehat{B^{-1}A\delta})
$$

Hence:
$$
   J_A = -B^{-1}A
$$


$$
    \exp(\widehat{J_B\delta}) = (Z^{-1}A^{{-1}}B)^{-1}(Z^{-1}A B \exp{(\hat{\delta}})) \\
    = B^{-1}AZ Z^{-1}A B \exp{(\hat{\delta}}) \\
    = \exp(\hat{\delta})
$$

Hence:
$$
   J_B = I
$$
