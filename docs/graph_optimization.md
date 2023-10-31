## Graph Optimization  
### What is graph?  
A graph is a pair $G = (V, E)$,
where $V$ is a set of vertices, each of which contains some parameters to be optimized.  $E$ is a set of connected information, whose elements are denotes the constraint relationship between two vertices.  
Many robotics and computer vision problems can be represented by a graph problem.

### How to solve graph problem?
A graph problem can be defined as a nonlinear least squares problems. Here, $r_k$ and $\Sigma_k$ represent the residual vector and the covariance matrix of edge k, respectively.

$$
\argmin_x  F(x) = \frac{1}{2} \sum_{e_{k}\in E} r_{k}^T \Sigma_{k}^{-1} r_{k} 
\tag{1}
$$



We need to find an optimal set of vertices (i.e. $V$) to minimize the overall cost. According to [guass_newton_method.md](./guass_newton_method.md), once we can compute the Hessian matrix $H$ and gradient $g$, we can solve this problem.

### The hessian matrix $H$
Assuming the number of vertices in the graph is n and the number of edges is m, the block sizes of J, r, H, and g are m x n, m x 1, n x n, and n x 1, respectively. We noticed that the size of H and g is independent of m.

The hessian matrix can be show as:
$$ 
H  =  J^T \Sigma^{-1} J
= \begin{bmatrix}
\ddots & \vdots & \vdots \\\\ 
 \vdots & \sum_{e_k \in E} {J_{i}^k}^T \Sigma^{-1}_{k} J_j^k & \vdots \\\\ 
\vdots & \vdots & \ddots
\end{bmatrix}
 \tag{2}
$$

### The gradient $g$

The gradient vector can be show as:

$$ 
g  =  J^T \Sigma^{-1} r = 
\begin{bmatrix}
\vdots \\\\
 \sum_{e_k \in E} {J_{i}^k}^T \Sigma^{-1}_{k} r_k  \\\\
\vdots 
\end{bmatrix}
\tag{3}
$$

$i$ and $j$ are vertex numbers, and they also indicate the row and column numbers within the Hessian matrix. $k$ is the edge number. $J_{i}^k$ represents the partial derivative matrix of $r_k$ with respect to $x_i$.

## Derivative of edge between two lie groups
Suppose $\varphi$ is an smooth mapping between two lie groups,
we can define the derivative of $\varphi$ as $J$:

$$
    \exp(\widehat{J\delta}) = \varphi(x)^{-1}\varphi(x\oplus\delta)
    \tag{4}
$$

$x$ is a the parameter of $\varphi$, and $\delta$ is a small increment to $x$.

The the transfrom error of two lie groups can define as:
$$
    \varphi(A,B) = Z^{-1}A^{{-1}}B \tag{5}
$$

Where $A$ and $B$ are the two lie groups, which represent the poses of two vertices. The $Z$ represents the relative pose of $A$ nad $B$, which usually measured by odometry or loop-closing.

### If A and B are SO3

$$
\begin{aligned}
    \exp(\widehat{J_A\delta}) 
    &= (Z^{-1}A^{{-1}}B)^{-1}(Z^{-1}(A\exp(\hat{\delta}))^{{-1}}B) \\
    &= B^{-1}AZ Z^{-1}\exp(-\hat{\delta})A^{{-1}}B \\
    &= B^{-1}A\exp(-\hat{\delta})A^{{-1}}B \\
    &= -\exp(B^{-1}A \hat{\delta} A^{{-1}}B) \\
    &= -\exp(\widehat{B^{-1}A\delta})
\end{aligned}
\tag{6}
$$

Hence:
$$
   J_A = -B^{-1}A \tag{7}
$$


$$
\begin{aligned}
    \exp(\widehat{J_B\delta}) 
    &= (Z^{-1}A^{{-1}}B)^{-1}(Z^{-1}A^{-1} B \exp{(\hat{\delta}})) \\
    &= B^{-1}AZ Z^{-1}A^{-1} B \exp{(\hat{\delta}}) \\
    &= \exp(\hat{\delta}) 
\end{aligned}
\tag{8}
$$

Hence:
$$
   J_B = I \tag{9}
$$

### If A and B are SE2

The small incremental matrix of SE2 can be shown as follow: 
$$
  \hat{\delta} = 
  \begin{bmatrix}
[ \omega ]_+ & v \\
0 & 0 \\
\end{bmatrix}
\tag{10}
$$


Where $\delta = \begin{bmatrix} v \\ w \end{bmatrix} \in \mathfrak{se}(2) $
$\omega$: the parameter of rotation (is a scalar). $[w]_+ = \begin{bmatrix} 0 & -w \\ w & 0 \end{bmatrix} $

$v$: the parameters of translation (is a 2d vector).


We rewrite the $B^{-1}A$ as $T_{BA}$.
$$
    T_{BA} =       
    \begin{bmatrix}
         R & t \\
        0 & 1 \\
    \end{bmatrix}
    \tag{11}
$$

We substitute (10) and (11) into (6), we get:
$$
\begin{aligned}
    \exp(\widehat{J_A\delta}) 
    &= -\exp(T_{BA} \hat{\delta} T_{BA}^{{-1}}) \\
    &= -\exp(T_{BA} 
          \begin{bmatrix}
            [ \omega ]_+ & v \\
            0 & 0 \\
            \end{bmatrix}
        T_{BA}^{{-1}}) \\
    &= -\exp
        (\begin{bmatrix}
             R & t \\
            0 & 1 \\
        \end{bmatrix}
          \begin{bmatrix}
            [ \omega ]_+ & v \\
            0 & 0 \\
            \end{bmatrix}
        \begin{bmatrix}
             R^T & -R^Tt \\
            0 & 1 \\
        \end{bmatrix}
        ) \\
    &= -\exp
        (\begin{bmatrix}
             R[ \omega ]_+ & Rv \\
            0 & 0 \\
        \end{bmatrix}
        \begin{bmatrix}
             R^T & -R^Tt \\
            0 & 1 \\
        \end{bmatrix}
        ) \\
     &= -\exp
        (\begin{bmatrix}
             R[ \omega ]_+R^T & R[ \omega ]_+(-R^Tt)+Rv \\
            0 & 0 \\
        \end{bmatrix}
        ) \quad \\
     &= -\exp
        (\begin{bmatrix}
             [ \omega ]_+ & -[ \omega ]_+t+Rv \\
            0 & 0 \\
        \end{bmatrix}
        ) \quad \\
     &= -\exp
        (\begin{bmatrix}
             [ \omega ]_+ & -[ \omega ]_+t+Rv \\
            0 & 0 \\
        \end{bmatrix}
)
\end{aligned} 
\tag{12}
$$

According to (10), we can rewrite (12) as:
$$
\begin{aligned}
\exp(\widehat{J_A\delta}) 
&=-\exp(\widehat{\begin{bmatrix}  -[ \omega ]_+t+Rv \\ w 
    \end{bmatrix}}) \\
&=-\exp(\widehat{\begin{bmatrix}  -\omega t^{\perp} +Rv \\ w 
    \end{bmatrix}}) \\
&=-\exp(\widehat{
    \begin{bmatrix}  R & -t^{\perp}\\ 0 & 1 \end{bmatrix}
    \begin{bmatrix}  v\\ w \end{bmatrix}
    })
\end{aligned} 
$$

Where $t^{\perp} = [1]_+  t=\begin{bmatrix} -t_2 \\ t_1 \end{bmatrix}$ 

Hence: 
$$
   J_A = -\begin{bmatrix}  R & -t^{\perp}\\ 0 & 1 \end{bmatrix}
   =  -\begin{bmatrix}  R_{BA} & -t_{BA}^{\perp}\\ 0 & 1 \end{bmatrix} \tag{13}
$$

similer with (9):

$$
J_B = I \tag{14}
$$



### If A and B are SE3

The small incremental matrix of SE3 can be shown as follow: 
$$
  \hat{\delta} = 
  \begin{bmatrix}
[ \omega ]_{\times} & v \\
0 & 0 \\
\end{bmatrix}
\tag{15}
$$


Where $\delta = \begin{bmatrix} v \\ w \end{bmatrix} \in \mathfrak{se}(3) $
$\omega$: the parameters of rotation (is a 3d vector). $[w]_{\times}$ is the skew symmetric matrix of $w$. 

$v$: the parameters of translation (is a 3d vector).

Similar to (12), we get:
$$
\begin{aligned}
    \exp(\widehat{J_A\delta}) 
    &= -\exp(T_{BA} \hat{\delta} T_{BA}^{{-1}}) \\
    &= -\exp(T_{BA} 
          \begin{bmatrix}
            [ \omega ]_{\times} & v \\
            0 & 0 \\
            \end{bmatrix}
        T_{BA}^{{-1}}) \\
    &= -\exp
        (\begin{bmatrix}
             R & t \\
            0 & 1 \\
        \end{bmatrix}
          \begin{bmatrix}
            [ \omega ]_{\times} & v \\
            0 & 0 \\
            \end{bmatrix}
        \begin{bmatrix}
             R^T & -R^Tt \\
            0 & 1 \\
        \end{bmatrix}
        ) \\
    &= -\exp
        (\begin{bmatrix}
             R[ \omega ]_{\times} & Rv \\
            0 & 0 \\
        \end{bmatrix}
        \begin{bmatrix}
             R^T & -R^Tt \\
            0 & 1 \\
        \end{bmatrix}
        ) \\
     &= -\exp
        (\begin{bmatrix}
             R[ \omega ]_{\times} R^T & -R[ \omega ]_{\times}R^Tt+Rv \\
            0 & 0 \\
        \end{bmatrix}
        ) \\
     &= -\exp
        (\begin{bmatrix}
             [ R\omega ]_{\times} & -[ R\omega ]_{\times}t+Rv \\
            0 & 0 \\
        \end{bmatrix}
        ) 
\end{aligned}
\tag{16}
$$

According to (10), we can rewrite (16) as:
$$
\begin{aligned}
\exp(\widehat{J_A\delta}) 
&=-\exp(\widehat{
    \begin{bmatrix}
          -[R\omega]_{\times}t+Rv \\
           Rw 
    \end{bmatrix}}) \\
&=-\exp(\widehat{
    \begin{bmatrix} 
         [t]_{\times}R\omega  +Rv \\
          Rw 
    \end{bmatrix}}) \\
&=-\exp(\widehat{
    \begin{bmatrix}  
        R & [t]_{\times}R \\
        0 & R 
    \end{bmatrix}
    \begin{bmatrix} 
        v \\
        w
    \end{bmatrix}
    })
\end{aligned}
\tag{17}
$$

Hence: 
$$
   J_A = -\begin{bmatrix}  
        R_{BA} & [t_{BA}]_{\times}R_{BA}  \\
        0 & R_{BA} 
    \end{bmatrix} \tag{18} 
$$

similer with (9):

$$
J_B = I \tag{19}
$$






