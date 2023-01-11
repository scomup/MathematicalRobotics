# 3D rotation   

$\gdef\skew#1{[{#1}]_{\times}}$
$\gdef\norm#1{\|{#1}\|}$
$\gdef\so3{\mathfrak{so}(3)}$

## Euler angles

The Euler angles are three angles introduced by Leonhard Euler to describe the orientation of a rigid body.

* $\alpha$: Often be called as *Roll*, rotation around the x axis.
* $\beta$: Often be called as *Pitch*, rotation around the y axis.
* $\gamma$: Often be called as *Yaw*, rotation around the z axis.

Rotation around the x axis.
$$
R_x(\alpha) = 
\left[\begin{matrix} 1 & 0 & 0 \\
0 & cos\alpha & -sin\alpha \\
0 & sin\alpha & cos\alpha
\end{matrix}\right]
\tag{1}
$$

Rotation around the y axis.
$$
R_y(\beta) = \left[\begin{matrix} cos\beta & 0 & sin\beta \\
0 & 1 & 0 \\
-sin\beta & 0 & cos\beta \end{matrix}
\right]
\tag{2}
$$

Rotation around the z axis.
$$
R_z(\gamma) = 
\left[\begin{matrix} cos\gamma & -sin\gamma & 0 \\
sin\gamma & cos\gamma & 0 \\ 0 & 0 & 1 
\end{matrix}\right]
\tag{3}
$$

Any rotation can be composed by above three elemental rotation matrices.

### Rotation order of Euler angles

But since the multiplication of matrices does not satisfy the commutative law, different orders of x,y and z rotation generate different final rotation matrices.

$$
\begin{aligned} R_{xyz}(\alpha, \beta, \gamma) 
&=
  R_z (\gamma) R_y (\beta) R_x (\alpha) \\
&= 
\left[\begin{matrix} cos\gamma & -sin\gamma & 0 \\
sin\gamma & cos\gamma & 0 \\ 0 & 0 & 1 
\end{matrix}\right]
\left[\begin{matrix} cos\beta & 0 & sin\beta \\
0 & 1 & 0 \\
-sin\beta & 0 & cos\beta 
\end{matrix}\right] 
\left[\begin{matrix} 1 & 0 & 0 \\
0 & cos\alpha & -sin\alpha \\
0 & sin\alpha & cos\alpha
\end{matrix}\right] \\
&= 
\left[\begin{matrix} c_{\gamma} c_{\beta} & c_{\gamma} s_{\beta} s_{\alpha} - c_{\alpha} s_{\gamma} & s_{\gamma} s_{\alpha} + c_{\gamma} c_{\alpha} s_{\beta} \\
c_{\beta} s_{\gamma} & c_{\gamma} c_{\alpha} + s_{\gamma} s_{\beta} s_{\alpha} & c_{\alpha} s_{\gamma} s_{\beta}- c_{\gamma} s_{\alpha} \\
-s_{\beta} & c_{\beta} s_{\alpha} & c_{\beta} c_{\alpha} \\
\end{matrix}\right]
\end{aligned} \tag{4}
$$

$$
\begin{aligned} R_{zxy}(\alpha, \beta, \gamma) 
&=
R_y (\beta) R_x (\alpha) R_z (\gamma) \\
&= 
\left[\begin{matrix} cos\beta & 0 & sin\beta \\
0 & 1 & 0 \\
-sin\beta & 0 & cos\beta 
\end{matrix}\right] 
\left[\begin{matrix} 1 & 0 & 0 \\
0 & cos\alpha & -sin\alpha \\
0 & sin\alpha & cos\alpha
\end{matrix}\right]
\left[\begin{matrix} cos\gamma & -sin\gamma & 0 \\
sin\gamma & cos\gamma & 0 \\ 0 & 0 & 1 
\end{matrix}\right] \\ 
&= 
\left[\begin{matrix} c_{\beta} c_{\gamma}+s_{\beta} s_{\alpha} s_{\gamma} & c_{\gamma} s_{\beta} s_{\alpha}-c_{\beta} s_{\gamma} & c_{\alpha}s_{\beta} \\ 
c_{\alpha} s_{\gamma} & c_{\alpha} c_{\gamma} & -s_{\alpha} \\
c_{\beta} s_{\alpha} s_{\gamma}-s_{\beta} c_{\gamma} & s_{\beta} s_{\gamma}+c_{\beta} c_{\gamma} s_{\alpha} & c_{\beta} c_{\alpha}
\end{matrix}\right]
\end{aligned} 
\tag{5}
$$

Therefore it is very important to determine the order of rotation when using Euler angles. 

Euler angles are frequently used, however, they are not continuous because of the gimbal lock problem. And they can be very tricky to handle in many mathematical problems due to the non-commutative nature.

### Infinitesimal Rotation

If the angles are small enough, the following equations are true.

$cos(a) \approx 1$
$sin(a)=a$
$sin(a)sin(b) = 0$

If we substitute the above equations into (8) and (9), we can yield the same results.

$\mathfrak{so}(3)$
$$
\begin{aligned} 
R_{xyz} \approx R_{zxy} &\approx
\left[\begin{matrix} 1 &  -  \gamma & \beta \\
\gamma & 1 & - \alpha \\
-\beta & \alpha & 1 \\
\end{matrix}\right] \\
&= I + 
\left[\begin{matrix} 0 &  -  \gamma & \beta \\
\gamma & 0 & - \alpha \\
-\beta & \alpha & 0 \\
\end{matrix}\right] \\
&= I + \skew{\omega}
\end{aligned} 
\qquad if \quad \alpha, \beta, \gamma \ll 1
\tag{6}
$$


$\skew{\omega}$ is a [skew-symmetric matrix](https://en.wikipedia.org/wiki/Skew-symmetric_matrix), which is composed of $\alpha$, $\beta$ and $\gamma$.

Now that we can obtain a commutative represention of a small rotation by a 3d vector. the remaining question is how to represent a larger rotation?

If we want a larger rotation, We can simply split 3d vector $\omega$ into n pieces, and compose the as follows:
$$
R(\omega) =
\underbrace{(I+\frac{\skew{\omega}}{n}) \times ...  (I+\frac{\skew{\omega}}{n})}_\text{n factors}
=(I+\frac{\skew{\omega}}{n})^n
\tag{7}
$$

For real numbers, this series is very famous, shows a way to compute the exponential function.
similarly, we can extend the definition of exponential function to skew-symmetric matrix.

$$
R(\omega) 
=(I+\frac{\skew{\omega}}{n})^n = e^{\skew{\omega}}
\tag{8}
$$

The exponential sum formula is also applicable to skew-symmetric matrix. Now, the above exponential map can transform a 3d vector into a rotation matrix.

$$
R(\omega) 
= e^{\skew{\omega}}
=\sum_{k=0}^\infty \frac{\skew{\omega}^k}{k!}
\tag{9}
$$

In actually, some theories of Lie groups have been described in above. The 3D rotation space $R$ is called as *special orthogonal group* $SO(3)$. The 3d vector $\omega$ is called the Lie algebra $\so3$ associated with $SO(3)$ by exponential map.

##Group

A [Group](https://en.wikipedia.org/wiki/Group_(mathematics)
) satisfied following requirements, known as group axioms.


* Closure:
$ \quad \forall a_1, a_2, \quad a_1 \cdot a_2 \in A$
* Associativity:
 $ \quad \forall a_1, a_2, a_3, \quad (a_1 \cdot a_2) \cdot a_3 = a_1 \cdot ( a_2 \cdot a_3) $
 * Identity:
$ \quad \exists a_0 \in A, \quad s.t. \quad \forall a \in A, \quad a_0 \cdot a = a \cdot a_0 = a $
 * Inverse:
$ \quad \forall a \in A, \quad \exists a^{-1} \in A, \quad s.t. \quad a \cdot a^{-1} = a_0 $


For example, the Rubik's Cube group is a group, we can simply verify that the group axioms are satisfied for it.

![cube](https://upload.wikimedia.org/wikipedia/commons/a/a6/Rubik%27s_cube.svg)

##Lie group

A Lie group is a continuous group, This means that a Lie group can be infinitely differentiable.
Therefore, The Rubik's Cube group is a group, but not a Lie group. 3D rotation space is a group also is a Lie group. 

Because of the following advantages, Lie group and Lie algebra are often used to represent rotations in the latest SLAM studies.

* Lie algebra only use 3 values to represent a rotation.
* A Lie group and Lie algebra is smooth (differentiable).
* There is no gimbal lock problem in Lie group or Lie algebra.
* For small rotation, multiplication of Lie group is commutative.
* For small rotation, Lie group is easy to be linearized (6).

## Special orthogonal group $SO(3)$

### exponential map
We can map a $\so3$ to $SO(3)$ by (8) or (9), But the calculation is too complicated. Here We will try to simplify it.

We define $\omega = \theta * k$
k is a unit vector

$$
\exp ( {{\mathbf{\phi}^\wedge }} ) 
a[0]
= \exp ( {\theta {\mathbf{a}^ \wedge }} ) \\ 
= \sum\limits_{n = 0}^\infty 
{\frac{1}{{n!}}{{( {\theta {\mathbf{a}^ \wedge }} )}^n}} a[0] \\
= \mathbf{I} + \theta {\mathbf{a}^ \wedge } + \frac{1}{{2!}}{\theta ^2}{\mathbf{a}^ \wedge }{\mathbf{a}^\wedge } + \frac{1}{{3!}}{\theta ^3}
$$
 + \frac{1}{{3!}}{\theta ^3}{\mathbf{a}^
$$

$$


