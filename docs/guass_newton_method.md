## Solve Nonlinear Least-Squares Problem with the Gauss-Newton Methods.  
### What is linear/nonlinear?  
* Linear: A polynomial of degree 1.  
* Nonlinear: A function cannot be expressed as a polynomial of degree 1.

### What is Least-Squares Problem?  
r(x) is the residual function, x is the parameter vector.  
The least-squares problem tries to find optimal parameters to minimize the sum of squared residuals. 
$$ 
cost = \sum_{i=0}^{n} (\frac{r^T \Sigma r}{2}) \tag{1}
$$
Where $\Sigma$ is the information matrix for the measurement. In the simple case, it can be the identity matrix.

### What is Gauss-Newton Methods?  
The gauss–newton methods is used to solve nonlinear least-squares problems.  
Unlike Newton's method, gauss-newton methods are not necessary to calculate second derivatves of residual function, which may difficult to compute in some cases.  
Gauss–newton methods update x using iterative method, the update amount is Δx. 

$$ 
\Delta x = -H^{-1}g \tag{2}
$$
here: g is the gradient vector; H is the hessian matrix.
$$ 
g = \sum g_i= \sum J_i^T \Sigma r_i \tag{3}
$$
$$ 
H  = \sum H_i \approx J_i^T \Sigma J_i  \tag{4}
$$
* r is the residual vector. 
* J is the jacobian matrix of r.

### The problem of 3D points matching

If we define the increment of SO3/SE3 as:

$$T(x_{0}\oplus\delta) \triangleq T(x_{0})\exp( \delta ) \tag{5} $$ 

The $\delta \in \mathfrak{so}(3)$ or $\delta \in \mathfrak{se}(3)$

We use a first-order Taylor expansion to approximate the original equation:  

$$T(x_{0}\oplus\delta) = T_{0}\exp( \delta ) \cong T_{0} + T_{0}\widehat{\delta} \tag{6} $$

The the residual function of 3D points matching problem can be defined as: 
$$r(x) = T(x)a - b \tag{7} $$

a is the target point:
b is the reference point.

We can use gauss-newton method to solve this problem.
According to gauss-newton method, we need to find the Jacobian matrix of r


$$
\begin{aligned}
\dot{r} &= \frac{T_{0}\exp\left( \delta \right)a - T_{0}a}{\delta} \\
&\cong \frac{T_{0}a + T_{0}\widehat{\delta}a - T_{0}a}{\delta}  \\
&= \frac{T_{0}\widehat{\delta}a}{\delta}  \\
&= - \frac{T_{0}\delta\widehat{a}}{\delta}  \\
&= - T_{0}\widehat{a} 
\end{aligned}
\tag{8} 
$$

### When $\delta \in \mathfrak{so}(3)$
$T_0$ is a 3d rotation matrix($R_0$),
and $\widehat{a}$ is defined as a skew symmetric matrix for vector $a$

$$\dot{r} = - R_{0}[ a ]_{\times}  \tag{9}$$

###  When $\delta \in \mathfrak{se}(3)$

$$\delta = [\ v, \omega ] \tag{10}$$

$\omega$: the parameters of rotation.

$v$: the parameters of translation.

$$\widehat{\delta} = \begin{bmatrix}
[ \omega ]_{\times} & v \\
0 & 0 \\
\end{bmatrix} \tag{11}$$

$$
\begin{aligned}
\dot{r} 
&= \frac{R_{0}\widehat{\delta}a}{\delta} \\
&=\frac{T_{0}\begin{bmatrix}
[ \omega ]_{\times} & v \\
0 & 0 \\
\end{bmatrix}
\begin{bmatrix}
a \\
1 \\
\end{bmatrix}}
{[v,\omega ]} \\
&= \frac{T_{0}\begin{bmatrix}
I & [ - a ]_{\times} \\
0 & 0 \\
\end{bmatrix}\begin{bmatrix}
v \\
\omega \\
\end{bmatrix}}{[v,\omega ]} \\
&= T_{0}\begin{bmatrix}
I & [ - a ]_{\times} \\
0 & 0 \\
\end{bmatrix} 
\end{aligned}
\tag{12}
$$

