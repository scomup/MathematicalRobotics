### Solve Nonlinear Least-Squares Problem with the Gauss-Newton Methods.  
#### What is linear/nonlinear?  
* Linear: A polynomial of degree 1.  
* Nonlinear: A function cannot be expressed as a polynomial of degree 1.

#### What is Least-Squares Problem?  
f(x) is the objective function, x is the parameter vector.  
The least-squares problem tries to find optimal parameters to minimize the overall cost. 
$$ 
cost = \sum_{i=0}^{n}f(x)^2 \quad
$$

#### What is Gauss-Newton Methods?  
The gauss–newton methods is used to solve nonlinear least-squares problems.  
Unlike Newton's method, gauss-newton methods are not necessary to calculate second derivatves, which may difficult to compute in some cases.  
Gauss–newton methods update x using iterative method, the update amount is Δx. 

$$ 
\Delta x = -H^{-1}g
$$
here: g is the gradient vector; H is the hessian matrix.
$$ 
g = J^Te
$$
$$ 
H \approx J^TJ
$$
J is the jacobian matrix, e is the residual vector. 


#### The jacabian matrix of 3d rotation or 3d transform 

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

