# Rigid Body Transformation of Velocity

# Introduction

Consider two objects A and B moving in 3D space. The translational and angular velocities of A are known. Assuming that A and B are mounted on the same rigid body, we want to calculate the translational and angular velocities of B.

This type of problem is commonly encountered in the field of robotics. For example, when calculating the velocity of a LiDAR mounted on a vehicle given the velocity of the wheel centers obtained from odometry, or when calculating the velocity of the end effector given the joint velocities of a robot Arm.

However, deriving the solution to this problem requires knowledge of Lie groups and is not easy. This text uses the knowledge of Lie groups discussed in the first part to explain the answer and solution to this problem.

# Problem

$$
\newcommand{\skew}[1]{[{#1}]_{\times}} %skew matrix
\newcommand{\so}[1]{ \mathfrak{so}{(#1)} } %lie algebra so3
\newcommand{\se}[1]{ \mathfrak{se}{(#1)} } %lie algebra se3
\newcommand{\norm}[1]{\|{#1}\|} %norm 
$$


The following are known:

-  $T_{ba}$: Rigid body transformation matrix that transforms A to B.
- $\xi_{a}$: Translational and angular velocities of A in its local frame (6-dimensional vector).
    - $v_a$： Translational velocities in the x, y, and z directions (3-dimensional vector).
    - $\omega_a$： Angular velocities around the x, y, and z axes (3-dimensional vector)
$$　
 \xi_{a}=
\left[\begin{matrix} 
  v_a \\
  \omega_a  \\
\end{matrix}\right]
$$


## Pose Difference

### Pose Difference of A

Before solving for the velocities, let us consider the difference between A and A'. As explained in the previous　documents, if the motion is at a constant speed, $T_{aa'}$ can be calculated using the exponential mapping of $\xi_a\Delta{t}$. Furthermore, if $\Delta{t}$ is sufficiently small, second-order and higher terms in the exponential mapping can be omitted, and it can be expressed as shown in equation (1).

$$
T_{aa'} = \exp{(\widehat{\xi \Delta{t}})} 
=\left[\begin{matrix} 
  I + \skew{\omega} \Delta{t}  & v\Delta{t}\\\\
  \mathbf{0}^T & 1 
\end{matrix}\right]
\tag{1}
$$

### Pose Difference of B

Since A and B are mounted on the same rigid body, difference between B and B' can be calculated as  equation (2).

$$
\begin{aligned} 
T_{bb'} &= T_{ba} T_{aa'} T_{ba}^{-1} \\\\
&= 
\left[\begin{matrix} 
  R_{ba} & t_{ba} \\\\
  \mathbf{0}^{-1} & 1  \\
\end{matrix}\right]
\left[\begin{matrix} 
  R_{aa'} & t_{aa'} \\\\
  \mathbf{0}^{-1} & 1  \\
\end{matrix}\right]
\left[\begin{matrix} 
  R_{ba}^{-1} & -R_{ba}^{-1} t_{ba} \\\\
  \mathbf{0}^T & 1  \\
\end{matrix}\right]\\\\
&= 
\left[\begin{matrix} 
  R_{ba}R_{aa'}R_{ba}^{-1} 
  & R_{ba}R_{aa'}(-R_{ba}^Tt_{ba})+R_{ba}t_{aa'}+t_{ba} \\\\
  \mathbf{0}^T & 1  \\
\end{matrix}\right]
\end{aligned}
\tag{2}
$$

## Translational Velocities
Next, let's consider translational velocities. If we denote the change of localization as $\Delta{x}$, the translational velocities is defined as follows.

$$
v = \lim_{\Delta{t} \to \infty} \frac{\Delta{x}}{\Delta{t}}
\tag{4}
$$

Substituting the translational part of Equation (3) with $t_{bb'}$ in Equation (4), the translational velocity of B can be derived as shown in Equation (5).

$$
\begin{aligned} 
v_{bb'} 
&= \lim_{\Delta{t} \to \infty} \frac{t_{bb'}}{\Delta{t}} \\\\
&= \lim_{\Delta{t} \to \infty} \frac{R_{ba}(I + \skew{\omega}\Delta{t})(-R_{ba}^Tt_{ba})+R_{ba}v\Delta{t}+t_{ba}}{\Delta{t}} \\\\
&=  -R_{ba}(I + \skew{\omega}  )R_{ba}^Tt_{ba}+R_{ba}v+t_{ba} \\\\
&=  -R_{ba}\skew{\omega}  R_{ba}^T t_{ba}  + R_{ba} v \\\\
&=  R_{ba}\skew{  R_{ba}^T t_{ba}} \omega  + R_{ba} v \\\\
&=  \skew{t_{ba}}R_{ba} \omega  + R_{ba} v  \\\\
\end{aligned}
\tag{5}
$$

When $R \in SO(3)$, the skew-symmetric matrix has the following property. By utilizing this property, we can transform from the 5th to the 6th line above.

### Angular Velocities

If we denote the change in orientation as $\Delta{R}$, the angular Velocities $\omega$ is defined as follows:

$$
\omega = \lim_{\Delta{t} \to \infty} \frac{\log{(\Delta{R})}^{\vee}}{\Delta{t}}
\tag{6}
$$

Since we have calculated the change in orientation of B in equation (4), we can calculate the angular velocities of B as follows:

$$
\begin{aligned} 
\omega_b 
&= \lim_{\Delta{t} \to \infty} \frac{\log{(R_{bb'})}^{\vee}}{\Delta{t}} \\\\
&= \lim_{\Delta{t} \to \infty} \frac{\log{(R_{ba}(I + \skew{\omega} \Delta{t} )R_{ba}^T)}^{\vee}}{\Delta{t}} \\\\
&= \lim_{\Delta{t} \to \infty} \frac{\log{(I+R_{ba}\skew{\omega} R_{ba}^T  \Delta{t})}^{\vee}}{\Delta{t}} \\\\
&= \lim_{\Delta{t} \to \infty} \frac{\log{(I+\skew{R_{ba}\omega \Delta{t}} )}^{\vee}}{\Delta{t}} \\\\
&= \lim_{\Delta{t} \to \infty} \frac{R_{ba}\omega \Delta{t}}{\Delta{t}} \\\\
&=  R_{ba}\omega\\\\
\end{aligned}
\tag{7}
$$



## Translational and Angular Velocities

Combining the results from equations (5) and (7), we can calculate the translational and angular Velocities of B in B's local frame as follows:

$$
\begin{aligned} 
\xi_b =
\left[\begin{matrix} 
  v_b \\\\
  \omega_b
\end{matrix}\right]
&= 
\left[\begin{matrix} 
  R_{ba} & \skew{t_{ba}}R_{ba} \\\\
  \mathbf{0}_{3\times3} & R_{ba}  \\
\end{matrix}\right] 
\left[\begin{matrix} 
  v_a \\\\
  \omega_a 
\end{matrix}\right]
\end{aligned}
\tag{8}
$$