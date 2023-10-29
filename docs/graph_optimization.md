## Graph Optimization  
### What is graph?  
A graph is a pair $G = (V, E)$,
where $V$ is a set of vertices, each of which contains some parameters to be optimized.  $E$ is a set of connected information, whose elements are denotes the constraint relationship between two vertices.  
Many robotics and computer vision problems can be represented by a graph problem.

### How to solve graph problem?
A graph problem can be defined as a nonlinear least squares problems.
$f_{ij}(v_i, v_j; e_{ij})$ shows the constraint relationship between vertex $v_i$ and $v_j$
$e_{ij}$ is the prior error of $v_i$ and $v_j$.  
$$ 
F(V) = \sum_{\{i,j\} \in E}f_{ij}(v_i, v_j; e_{ij})^2 \quad \tag{1}
$$

We need to find a optimal set of vertices (i.e. $V$) to minimize the overall cost. 
According to [guass_newton_method.md](./guass_newton_method.md), 
as soon as we can compute the hessian matrix $H$ and gradient $g$, we can solve this graph optimization problem.

### The hessian matrix $H$
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
\end{bmatrix} \tag{2}$$

The $J_i^T J_i$ is located in row i column i of $H_{ij}$  
The $J_j^T J_j$ is located in row j column j of $H_{ij}$  
The $J_i^T J_j$ is located in row i column j of $H_{ij}$  
The $J_j^T J_i$ is located in row j column i of $H_{ij}$  

The overall hessian matrix of F is:

$$ H = \sum_{ \{i,j\} \in E}{H_{ij}} \tag{3} $$

### The gradient $g$

The gradient vector of $f_{ij}$ can be show as:

$$g_{ij} = 
\begin{bmatrix}
... \\
J_i^T r_i \\
... \\
J_j^T r_n \\
... \\
\end{bmatrix} \tag{4}$$

The $J_i^T r_i$ is located in row i of $g_{ij}$  
The $J_j^T r_j$ is located in row j of $g_{ij}$  

The overall gradient vector of F is:

$$ g = \sum_{\{i,j\} \in E}{g_{ij}} \tag{5} $$

## Derivative of edge between two lie groups
Suppose $\varphi$ is an smooth mapping between two lie groups,
we can define the derivative of $\varphi$ as $J$:

$$
    \exp(\widehat{J\delta}) = \varphi(x)^{-1}\varphi(x\oplus\delta)
    \tag{6}
$$

$x$ is a the parameter of $\varphi$, and $\delta$ is a small increment to $x$.

The the transfrom error of two lie groups can define as:
$$
    \varphi(A,B) = Z^{-1}A^{{-1}}B \tag{7}
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
\tag{8}
$$

Hence:
$$
   J_A = -B^{-1}A \tag{9}
$$


$$
\begin{aligned}
    \exp(\widehat{J_B\delta}) 
    &= (Z^{-1}A^{{-1}}B)^{-1}(Z^{-1}A^{-1} B \exp{(\hat{\delta}})) \\
    &= B^{-1}AZ Z^{-1}A^{-1} B \exp{(\hat{\delta}}) \\
    &= \exp(\hat{\delta}) 
\end{aligned}
\tag{10}
$$

Hence:
$$
   J_B = I \tag{11}
$$

### If A and B are SE2

The small incremental matrix of SE2 can be shown as follow: 
$$
  \hat{\delta} = 
  \begin{bmatrix}
[ \omega ]_+ & v \\
0 & 0 \\
\end{bmatrix}
\tag{12}
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
    \tag{13}
$$

We substitute (12) and (13) into (8), we get:
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
\tag{14}
$$

According to (12), we can rewrite (14) as:
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
   =  -\begin{bmatrix}  R_{BA} & -t_{BA}^{\perp}\\ 0 & 1 \end{bmatrix} \tag{15}
$$

similer with (11):

$$
J_B = I \tag{16}
$$



### If A and B are SE3

The small incremental matrix of SE3 can be shown as follow: 
$$
  \hat{\delta} = 
  \begin{bmatrix}
[ \omega ]_{\times} & v \\
0 & 0 \\
\end{bmatrix}
\tag{17}
$$


Where $\delta = \begin{bmatrix} v \\ w \end{bmatrix} \in \mathfrak{se}(3) $
$\omega$: the parameters of rotation (is a 3d vector). $[w]_{\times}$ is the skew symmetric matrix of $w$. 

$v$: the parameters of translation (is a 3d vector).

Similar to (14), we get:
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
\tag{18}
$$

According to (12), we can rewrite (18) as:
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
\tag{19}
$$

Hence: 
$$
   J_A = -\begin{bmatrix}  
        R_{BA} & [t_{BA}]_{\times}R_{BA}  \\
        0 & R_{BA} 
    \end{bmatrix} \tag{20} 
$$

similer with (11):

$$
J_B = I \tag{21}
$$






