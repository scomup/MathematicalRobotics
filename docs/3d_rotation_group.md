# 3D rotation and Lie group

## Euler angles

Euler angles are a set of three angles introduced by Leonhard Euler that are used to describe the orientation or rotation of a rigid body in three-dimensional space.


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

By combining these three angles, the orientation of the rigid body in three-dimensional space can be fully described. 


### Rotation order of Euler angles

But since the multiplication of matrices does not satisfy the commutative law, different orders of x,y and z rotation generate different final rotation matrices.

$$
\begin{aligned} R_{xyz}
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
\begin{aligned} R_{zxy}
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

Therefore, it is crucial to determine the rotation order when using Euler angles.

Euler angles are commonly used, however, they are not continuous due to the gimbal lock problem. Additionally, handling them in various mathematical problems can be challenging because of their non-commutative nature.

### Infinitesimal Rotation

If the angles are small enough, the following approximations hold true:

* $cos(a) \approx 1$
* $sin(a)\approx a$
* $sin(a)sin(b) \approx 0$

By substituting these approximations into equations (8) or (9), we can obtain the same results.

$$
\newcommand{\norm}[1]{\|{#1}\|} %norm 
$$

$$
\newcommand{\so}[1]{ \mathfrak{so}{(#1)} } %lie algebra so3
$$

$$
\newcommand{\skew}[1]{[{#1}]_{\times}} %skew matrix
$$

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


Here, $\skew{\omega}$ is a [skew-symmetric matrix](https://en.wikipedia.org/wiki/Skew-symmetric_matrix). A skew-symmetric matrix is a square matrix where the transpose of the matrix is equal to the negation of the matrix itself.

Now that we can obtain a commutative represention of a small rotation by a 3d vector. The remaining question is how to represent a larger rotation?

If we desire a larger rotation, we can simply divide the 3D vector $\omega$ into n pieces and compose them as follows:

$$
R(\omega) =
\underbrace{(I+\frac{\skew{\omega}}{n}) \times ...  (I+\frac{\skew{\omega}}{n})}_\text{n factors}
=(I+\frac{\skew{\omega}}{n})^n
\tag{7}
$$

For real numbers, this series is very famous, shows a way to compute the exponential function.
Similarly, we can extend the definition of exponential function to skew-symmetric matrix.

$$
R(\omega) 
=(I+\frac{\skew{\omega}}{n})^n = e^{\skew{\omega}}
\tag{8}
$$

The exponential sum formula is also applicable to skew-symmetric matrix. Now, the exponential map (8) or (9) can transform a 3d vector into a rotation matrix.

$$
R(\omega) 
= e^{\skew{\omega}}
=\sum_{k=0}^\infty \frac{\skew{\omega}^k}{k!}
\tag{9}
$$


In actually, some part of Lie group theories have been described in above. The 3D rotation space $R$ is called as *special orthogonal group* $SO(3)$. The 3d vector $\omega$ is called the Lie algebra $\so{3}$ associated with $SO(3)$ by the exponential map.

## Group

A [Group](https://en.wikipedia.org/wiki/Group_(mathematics)
) satisfied following requirements, known as group axioms.


* Closure:
 if $ a_1, a_2 \in G$, then $a_1 \cdot a_2 \in G$
* Associativity:
 if $ a_1, a_2, a_3 \in G$, then $(a_1 \cdot a_2) \cdot a_3 = a_1 \cdot ( a_2 \cdot a_3) $
* Identity:
 For every $ a \in G$, there exists a $a_0 \in G$, such that $ a_0 \cdot a = a \cdot a_0 = a $
* Invertibility:
 For every $ a \in G$, there exists a $a^{-1} \in G$, such that $ a \cdot a^{-1} = a_0 $


For example, the Rubik's Cube group is a group, we can simply verify that the group axioms are satisfied for it.

![cube](https://upload.wikimedia.org/wikipedia/commons/a/a6/Rubik%27s_cube.svg)

##  Lie group

A Lie group is a continuous group, which means a Lie group is infinitely differentiable (smooth).
Therefore, The Rubik's Cube group, on the other hand, is a group but not a Lie group. In contrast, 3D rotation space is both a group and a Lie group.

Due to several advantages, Lie groups and Lie algebras are commonly used to represent rotations in modern SLAM (Simultaneous Localization and Mapping) studies. These advantages include:

* Lie algebra only requires three values to represent a rotation..
* A Lie groups or Lie algebras is differentiable.
* Gimbal lock problems do not occur in Lie groups or Lie algebras.
* For small rotations, Lie groups are easily linearized (6).

## Special orthogonal group $SO(3)$

### Exponential map
We can map a $\so3$ to $SO(3)$ using equations (8) or (9), However, these calculations can be quite complex. To simplify the process, we introduce the following definitions:

We define $\omega = \theta r$
* $r$ is a unit vector of $\omega$, $ r =\frac{\omega}{\norm{\omega}} $.
* $\theta$ is the norm of a $\omega$, $ \theta = \norm{\omega} $  

The exponential map can be computed as follows:

$$
\begin{aligned} 
\exp ( \skew{\omega} ) 
&= \exp ( \theta \skew{r} ) \\ 
&= \sum\limits_{k = 0}^\infty 
\frac{1}{k!} (\theta \skew{r} )^n \\
&= I + \theta \skew{r} + 
\frac{1}{2!} \theta^2 \skew{r}^2 +
\frac{1}{3!} \theta^3 \skew{r}^3 +
\frac{1}{4!} \theta^4 \skew{r}^4 + ... \\ 
&= r^T r - 
\skew{r}^2 + \theta \skew{r} +
\frac{1}{2!} \theta^2 \skew{r}^2 +
\frac{1}{3!} \theta^3 \skew{r}^3 +
\frac{1}{4!} \theta^4 \skew{r}^4 + ... \\ 
&= r^T r - (\skew{r}^2 - 
\frac{1}{2!} \theta^2 \skew{r}^2 -
\frac{1}{4!} \theta^4 \skew{r}^4 - ...) + (\theta \skew{r} +\frac{1}{3!} \theta^3 \skew{r}^3 + ...) \\ 
&= r^T r - (1 - 
\frac{1}{2!} \theta^2  +
\frac{1}{4!} \theta^4  - ...)\skew{r}^2 + (\theta -\frac{1}{3!}  \skew{r}^3 + ...)\skew{r} \\ 
&= r^T r - cos\theta \skew{r}^2 + sin\theta\skew{r} \\ 
&= \skew{r}^2 + I - cos\theta \skew{r}^2 + sin\theta\skew{r} \\ 
&= I + (1- cos\theta) \skew{r}^2 + sin\theta\skew{r} 
\end{aligned} 
\tag{10}
$$

In (10) two properties of skew symmetric matrices are used.
If $r$ is a unit vector:
* $\skew{r}\skew{r}\skew{r} = -\skew{r} $
* $r^Tr = \skew{r}^2 + I $



The formula (10) represents a fast way to calculate the exponential map from $\so3$ to $SO(3)$, and is known as [Rodrigues' Formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula).

### Logarithm map

In contrast to exponential map, the logarithm map corresponds a Lie algebra to a Lie group.

The formula (10) can be wrtten as matrix:

$$
\begin{aligned} 
R(\theta, r) 
&= I + (1- cos\theta) \
\left[\begin{matrix} 
1-r_1^2  & r_1 r_2 & r_1 r_3 \\
r_1 r_2 & 1-r_1^2 & -r_2 r_3 \\
-r_1 r_3 & r_2 r_3 & 1-r_1^2 \\
\end{matrix}\right] + 
sin\theta
\left[\begin{matrix} 
0 & -r_3 & r_2 \\
r_3 & 0 & -r_1 \\
-r_2 & r_1 & 0 \\
\end{matrix}\right] \\
&= \left[\begin{matrix} 
r_1^2 (1-cos\theta) + cos\theta        & r_1 r_2 (1-cos\theta) - r_3 sin\theta & r_1 r_3 (1-cos\theta) + r_2 sin\theta\\
r_1 r_2 (1-cos\theta) +r_3 sin\theta   & r_2^2 (1-cos\theta) + cos\theta      & -r_2 r_3 (1-cos\theta) - r_1 sin\theta\\
-r_1 r_3 (1-cos\theta) -r_2 sin\theta  & r_2 r_3 (1-cos\theta) + r_1 sin\theta & r_3^2 (1-cos\theta) + cos\theta\\
\end{matrix}\right]
\end{aligned} 
\tag{11}
$$

From equation (11), we can derive the following formulas:

$$
\theta = arccos( \frac{1}{2}(R_{11} + R_{22} + R_{33} -1)) \\
= arccos( \frac{1}{2}(tr(r)) -1)
\tag{12}
$$

$$
r = [ R_{32} - R_{23}, R_{13} - R_{31}, R_{12} - R_{21}]/2 sin \theta 
\tag{13}
$$

From equations (12) and (13), the logarithm map can be implemented as follows:

$$
log(R)^{\vee} = \omega = \frac{\theta[ R_{32} - R_{23}, R_{13} - R_{31}, R_{12} - R_{21}]}{2 sin \theta}  
\tag{14}
$$
