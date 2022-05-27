## Robust kernel Newton Methods.  

### Optimization Problem
Optimization problem tries to find a set of best parameters to minimize the overall cost. 
$$
F = \sum_{i=0}^{n} \rho( f(x) )
$$

* f is the objective function. 
* x is the parameter vector.
* $\rho$ is robust kernel.  

### Least-Squares Problem

When the objective function is presented by square form, we call this problem as least-squares problem.

$$ 
f =  \frac{r^T \Sigma r}{2} 
$$

### Newton's method
Newton's method is an iterative method for solving nonlinear optimization problems. $\Delta x$ is iterative step.

$$ 
\Delta x = -H^{-1}g
$$
here: g is the gradient vector; H is the hessian matrix.

#### No robust kernel.
$$ 
g =\dot{f} \\
$$

$$ 
H = \ddot{f}
$$

$\dot{f}$: Partial derivatives of objective function 
$\ddot{f}$: Second-order partial derivatives of objective function 

$$
\dot{f} = r^T \Sigma \dot{r} 
$$

$$
\ddot{f} = \dot{r}^T \Sigma \dot{r} + r^T \Sigma \ddot{r} 
$$

#### With robust kernel.

$$
H = \ddot{\rho} \dot{f} \dot{f}^T + \dot{\rho} \ddot{f} 
$$

$$
g = \dot{\rho} \dot{f}  
$$


* $\dot{\rho}$: Partial derivatives of robust kernel function 
* $\ddot{\rho}$: Second-order partial derivatives of robust kernel function 

### Gaussian-Newton method
Unlike Newton's method, gauss-newton methods are not necessary to calculate second derivatves, which may difficult to compute in some cases.  

$$ 
g =\dot{f} \\
$$

$$ 
H = \ddot{f}
$$
#### No robust kernel.
$$
g =\dot{f} = r^T \Sigma \dot{r} 
$$

$$
H = \ddot{f} = \dot{r}^T \Sigma \dot{r} 
$$
#### With robust kernel.
$$
g = \dot{\rho} \dot{f}  
$$

$$
H = \dot{\rho} \dot{r}^T \Sigma \dot{r}
$$


### The jacabian matrix of 3d rotation or 3d transform ($\dot{r}$) 

If we define the increment of SO3/SE3 as:

$$T(x_{0}\oplus\delta) \triangleq T(x_{0})\exp( \delta )$$

The $\delta \in \mathfrak{so}(3)$ or $\delta \in \mathfrak{se}(3)$

We use a first-order Taylor expansion to approximate the original equation:  

$$T(x_{0}\oplus\delta) = T_{0}\exp( \delta ) \cong T_{0} + T_{0}\widehat{\delta}$$

The f(x) is the objective function. 
We want to find the optimal parameters (x) that minimize the result of the objective function.
$$f(x) = T(x)a - b$$

a is the target point:
b is the reference point.

We can use gauss-newton method to solve this problem.
According to gauss-newton method, we need to find the Jacobian matrix
for f(x).

$$\dot{f} = \frac{T_{0}\exp\left( \delta \right)a - T_{0}a}{\delta}$$

$$\cong \frac{T_{0}a + T_{0}\widehat{\delta}a - T_{0}a}{\delta}$$

$$= \frac{T_{0}\widehat{\delta}a}{\delta}$$

$$= - \frac{T_{0}\delta\widehat{a}}{\delta}$$

$$= - T_{0}\widehat{a} $$

#### When $\delta \in \mathfrak{so}(3)$
$T_0$ is a 3d rotation matrix($R_0$),
and $\widehat{a}$ is defined as a skew symmetric matrix for vector $a$

$$\dot{f} = - R_{0}[ a ]_{\times}$$

####  When $\delta \in \mathfrak{se}(3)$

$$\delta = [\ v, \omega ]$$

$\omega$: the parameter of rotation.

$v$: the parameter of translation.

$$\widehat{\delta} = \begin{bmatrix}
[ \omega ]_{\times} & v \\
0 & 0 \\
\end{bmatrix}$$

$$\dot{f} = \frac{R_{0}\widehat{\delta}a}{\delta}$$

$$= \frac{T_{0}\begin{bmatrix}
[ \omega ]_{\times} & v \\
0 & 0 \\
\end{bmatrix}\begin{bmatrix}
a \\
1 \\
\end{bmatrix}}{[ \omega,\ v ]}$$

$$= \frac{T_{0}\begin{bmatrix}
[ - a ]_{\times} & I \\
0 & 0 \\
\end{bmatrix}\begin{bmatrix}
\omega \\
v \\
\end{bmatrix}}{[ \omega,\ v ]}$$

$$= T_{0}\begin{bmatrix}
[ - a ]_{\times} & I \\
0 & 0 \\
\end{bmatrix}$$

