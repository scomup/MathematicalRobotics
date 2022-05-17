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

The $ J_i^T J_i $ is located in row i column i of $H_{ij}$  
The $ J_j^T J_j $ is located in row j column j of $H_{ij}$  
The $ J_i^T J_j $ is located in row i column j of $H_{ij}$  
The $ J_j^T J_i $ is located in row j column i of $H_{ij}$  

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

The $ J_i^T r_i $ is located in row i of $g_{ij}$  
The $ J_j^T J_j $ is located in row j of $g_{ij}$  

The overall gradient vector of F is:

$$ g = \sum_{\{i,j\} \in E}{g_{ij}} $$
