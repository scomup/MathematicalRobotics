## Imu preintegration.  

### Our problem
Suppose we know the state of the robot at time i , as well as the IMU measurements from the i time to j time. We want predict the state of robot at time j.
$$ 
s_j^* = \rho(d(\xi(\zeta, b), s_i),s_i) \\
$$
The state of robot combined by attitude $\theta$, position $p$ and velocity $v$.   
$$
s_i = (\theta_{nb}, p_{nb}, v_{nb}) \\
s_j = (\theta_{nc}, p_{nc}, v_{nc}) \\
$$
* n denotes navigation state frame.
* b denotes body frame in time i.
* c denotes current frame in time j.

Function $\rho$ which predict $s_j$ take 2 parameters, $s_i$ and $d$. The $d$ represents the difference between two $s_i$ and $s_j$.
$$
\rho = s_i \oplus d \\
d(\xi,s_i) = (\theta_{nc}, p_{nc}, v_{nc}) \\
$$
$\xi$ represents bias corrected PIM (preintegration measurement), which take 2 parameters, the PIM $\zeta$ and IMU bias b.

#### The Jacobian of $s_i$
$$
J^{s_j^*}_{s_i} = J^{\rho}_{s_i} + J^{\rho}_{d} J^{d}_{s_i}
$$ 
#### The Jacobian of $b$
$$
J^{s_j^*}_{b} = J^{\rho}_{d} J^{d}_{\xi} J^{\xi}_{b}
$$ 

### PIM (preintegration measurement)
The PIM $\zeta(\theta, p ,v)$ integrates all the IMU measurements  without considering the state of the bias and the gravity.
$\omega^b_k$,$a^b_k$ are the acceleration and angular velocity measured by IMU (accelerometer + gyroscope) respectively.
$$
\theta_{k+1} = \theta_k + H(\theta_k)^{-1}\omega^b_k\Delta{t} \\
p_{k+1} = p_k + v_k\Delta{t} +R_k a^b_k \frac{\Delta{t}^2}{2} \\
v_{k+1} = v_k + R_k a^b_k \Delta{t}
$$
$n$: navigation frame, $b$: body frame.
#### A:Derivative of old $\zeta$
$$
A = \frac{\partial{\zeta_{k+1}}}{\partial \zeta_{k}} = 
\begin{bmatrix}
 \frac{\partial{\theta_{k+1}}}{\partial{\theta_{k}}}  & \frac{\partial{\theta_{k+1}}}{\partial{p_{k}}} &  \frac{\partial{\theta_{k+1}}}{\partial{v_{k}}}\\  
 \frac{\partial{p_{k+1}}}{\partial{\theta_{k}}}  & \frac{\partial{p_{k+1}}}{\partial{p_{k}}} &  \frac{\partial{p_{k+1}}}{\partial{v_{k}}}\\  
 \frac{\partial{v_{k+1}}}{\partial{\theta_{k}}}  & \frac{\partial{v_{k+1}}}{\partial{p_{k}}} &  \frac{\partial{v_{k+1}}}{\partial{v_{k}}}\\   
\end{bmatrix} 
=\begin{bmatrix}
 \frac{\partial{\theta_{k+1}}}{\partial{\theta_{k}}}  & 0_{3\times3} & 0_{3\times3}\\  
 \frac{\partial{p_{k+1}}}{\partial{\theta_{k}}}  & I_{3\times3} &  I_{3\times3} \Delta{t}\\  
 \frac{\partial{v_{k+1}}}{\partial{\theta_{k}}}  &  0_{3\times3} &  I_{3\times3}\\   
\end{bmatrix} \\
=\begin{bmatrix}
 I_{3\times3}-\frac{\Delta_{t}}{2}\widehat{\omega_{k}^{b}}  & 0_{3\times3} & 0_{3\times3}\\  
 -R_{k}\widehat{a_{k}^{b}}H(\theta_{k})\frac{\Delta_{t}}{2}^{2} & I_{3\times3} &  I_{3\times3} \Delta{t}\\  
 -R_{k}\widehat{a_{k}^{b}}H(\theta_{k})\Delta_{t}  &  0_{3\times3} &  I_{3\times3}\\   
\end{bmatrix}
$$
#### B:Derivative of input $a$
$$
B = \frac{\partial{\zeta_{k+1}}}{\partial a^b_k} = 
\begin{bmatrix}
 \frac{\partial{\theta_{k+1}}}{\partial{a^b_kk}} \\  
 \frac{\partial{p_{k+1}}}{\partial{a^b_k}}  \\  
 \frac{\partial{v_{k+1}}}{\partial{a^b_k}}  \\   
\end{bmatrix} 
=\begin{bmatrix}
 0_{3\times3} \\  
 R_{k}\frac{\Delta{t}^2}{2}  \\  
 R_{k} \Delta{t}  \\   
\end{bmatrix}  \\
$$

#### C:Derivative of input $\omega$
$$
C = \frac{\partial{\zeta_{k+1}}}{\partial \omega^b_k} = 
\begin{bmatrix}
 \frac{\partial{\theta_{k+1}}}{\partial{\omega^b_k}} \\  
 \frac{\partial{p_{k+1}}}{\partial{\omega^b_k}}  \\  
 \frac{\partial{v_{k+1}}}{\partial{\omega^b_k}}  \\   
\end{bmatrix} 
=\begin{bmatrix}
 \frac{\partial{\theta_{k+1}}}{\partial{\omega^b_k}} \\  
 0_{3\times3}  \\  
 0_{3\times3}  \\   
\end{bmatrix}  
=\begin{bmatrix}
 H(\theta_{k})^{-1}{\Delta{t}} \\  
 0_{3\times3}  \\  
 0_{3\times3}  \\   
\end{bmatrix} 
$$
### Bias correct
We want correct $\zeta$ by a given accelerometer and gyroscope bias.   
$$
\xi(\zeta,b+\Delta{b}) = \zeta + \Delta b_{acc} \frac{\partial{\zeta}}{\partial b_{acc}} +
\Delta b_{\omega} \frac{\partial{\zeta}}{\partial b_{\omega}} 
$$
* $b_{acc}$ is bias for accelerometer.
* $b_{\omega}$ is bias for gyroscope.
#### The jocabian of bias for corrected PIM.
$$
J^{\xi}_{b} =  [ \frac{\partial{\zeta}}{\partial b_{acc}}, \frac{\partial{\zeta}}{\partial b_{\omega}}] 
$$

#### Find the partial derivatives of accelerometer's bias

The bias model for accelerometer.
$$
\tilde{a^b_k} = a^b_k - b_{acc}  \\
$$

$$
\frac{\partial{\zeta_{k+1}}}{\partial b_{acc}} =
\frac{\partial{\zeta_{k+1}}}{\partial \zeta_{k}} \frac{\partial{\zeta_{k}}}{\partial b_{acc}} +
\frac{\partial{\zeta_{k+1}}}{\partial \tilde{a^b_k}} \frac{\partial{\tilde{a^b_k}}}{\partial b_{acc}} \\
= A \frac{\partial{\zeta_{k}}}{\partial b_{acc}} - B
$$
#### Find the partial derivatives of gyroscope's bias

$$
\tilde{\omega^b_k} = \omega^b_k - b_{\omega}
$$

$$
\frac{\partial{\zeta_{k+1}}}{\partial b_{\omega}} =
\frac{\partial{\zeta_{k+1}}}{\partial \zeta_{k}} \frac{\partial{\zeta_{k}}}{\partial b_{\omega}} +
\frac{\partial{\zeta_{k+1}}}{\partial \tilde{\omega^b_k}} \frac{\partial{\tilde{\omega^b_k}}}{\partial b_{\omega}} \\
= A \frac{\partial{\zeta_{k}}}{\partial b_{\omega}} - C
$$

* ~ denotes the corrected measurement.


### Delta between two states
The $d$ represents the difference between two $s_i$ and $s_j$.   
$$
d = (\theta_{bc}, p_{bc}, v_{bc}) \\
$$
We can calculate $d$ from corrected PIM $\xi(\theta_{bc}^{\xi},p_{bc}^{\xi},v_{bc}^{\xi})$ and velocity, which is included in $s_i$.

$$
d(\xi,s_i)=\begin{bmatrix}
\theta_{bc}^{\xi}\\  
p_{bc}^{\xi} + R_{nb}^{-1} v_{nb} \Delta{t} + R_{nb}^{-1} g \frac{\Delta{t}^2}{2} \\  
v_{bc}^{\xi} + R_{nb}^{-1} g \Delta{t}\\   
\end{bmatrix}
$$

* $g$ is the gravity vector.
* $*$ denotes the predicted navigation state.


#### The jocabian matrix of navigation state
$$
J^{d}_{s_i}=\begin{bmatrix}
 0_{3\times3}  & 0_{3\times3} & 0_{3\times3}\\  
 \frac{\partial{p_{bc}}}{\partial \theta_{nb}} & 0_{3\times3} &  \frac{\partial p_{bc}}{\partial v_{nb}} \\  
\frac{\partial{v_{bc}}}{\partial \theta_{nb}}  &  0_{3\times3} &  0_{3\times3}\\   
\end{bmatrix} \\
$$

where:
$$
\frac{\partial{p_{bc}}}{\partial \theta_{nb}} = \widehat{R_{nb}^{-1}v_{nb}} \Delta{t} + \widehat{R_{nb}^{-1}g} \frac{\Delta{t}^2}{2}
$$

$$
\frac{\partial\ p_{bc}}{\partial v_{nb}} = R_{nb}^{-1}R_{nb} \Delta{t} 
$$

$$
\frac{\partial{ v_{bc}}}{\partial \theta_{nb}} = \widehat{R_{nb}^{-1}g} \Delta{t}
$$
#### The jocabian matrix of $\xi$
$$
J^{d}_{\xi}=\begin{bmatrix}
 I_{3\times3} & 0_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & I_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & 0_{3\times3} & I_{3\times3}\\   
\end{bmatrix} \\
$$


### Predict function $\rho$
Function $\rho$ which predict $s_j$ take 2 parameters, $s_i$ and $d$ to predict $s_j^*$

* $s_j^*$ is the predicted $s_j$.

$$
R_{nc}^{*} = R_{nb}R_{bc} \\
p_{nc}^{*} = p_{nb} + R_{nb} p_{bc} \\
v_{nc}^{*} = v_{nb} + R_{nb} v_{bc}
$$
#### Derivative of $s_i$
$$
J^{\rho}_{s_i}=\begin{bmatrix}
 R_{bc}^{-1} & 0_{3\times3} & 0_{3\times3}\\  
 R_{bc}^{-1} \widehat{-p_{bc}} & R_{bc}^{-1} & 0_{3\times3}\\  
 R_{bc}^{-1} \widehat{-v_{bc}} & 0_{3\times3} & R_{bc}^{-1}\\   
\end{bmatrix} \\
$$

#### Derivative of $d$
$$
J^{\rho}_{d}=\begin{bmatrix}
 H(\theta_{bc}) & 0_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & R_{bc}^{-1} & 0_{3\times3}\\  
 0_{3\times3} & 0_{3\times3} & R_{bc}^{-1}\\   
\end{bmatrix} \\
$$
Where H is the derivative of the exponential map in $\theta$.

### Derivative of an Inverse Action
$$g = T^{-1}(x)p $$

$$\frac{\partial{g}}{\partial x} = \frac{(T e^{\widehat{\delta{x}}})^{-1} p - T^{-1}p}{\delta{x}} \\ 
= \frac{ e^{-\widehat{\delta{x}}} T^{-1} p - T^{-1}p}{\delta{x}} \\
= \frac{ (I - \widehat{\delta{x}}) T^{-1} p - T^{-1}p}{\delta{x}} \\
= \frac{ -\widehat{\delta{x}} T^{-1} p}{\delta{x}} \\
= \frac{ \delta{x} \widehat{T^{-1} p}}{\delta{x}} \\
= \widehat{T^{-1} p}\\
$$


$$\frac{\partial{g}}{\partial p} = \frac{T^{-1} (p + \delta{p})-T^{-1}p }{\delta{p}} \\ 
= \frac{T^{-1} \delta{p}}{\delta{p}} \\
= T^{-1}
$$


$$
    \exp(\widehat{J\delta}) = \varphi(x)^{-1}\varphi(x\oplus\delta)
    \tag{6}
$$

$$
    \varphi(A,B) = AB \tag{7}
$$

Where $A$ and $B$ are the two lie groups, which represent the poses of two nodes. The $Z$ represents the relative pose of $A$ nad $B$, which usually measured by odometry or loop-closing.

### If A and B are SO3

$$
    \exp(\widehat{J_A\delta}) = (AB)^{-1}(A\exp(\hat{\delta})B) \\
    = B^{-1}A^{-1}A\exp(\hat{\delta})B \\
    = \exp(B^{-1}\hat{\delta} B) \\
    = \exp(\widehat{B^{-1}\delta})
     \tag{8}
$$

Hence:
$$
   J_A = B^{-1} \tag{9}
$$

$$
    \exp(\widehat{J_B\delta}) = (AB)^{-1}(AB\exp(\hat{\delta})) \\
    = B^{-1}A^{-1}AB\exp(\hat{\delta}) \\
    = \exp(\hat{\delta}) \\
     \tag{8}
$$

Hence:
$$
   J_B = I
$$

