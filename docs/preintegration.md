## Imu preintegration.  

### Our problem
Suppose we know the state of the robot at time i and j, as well as the IMU measurements between the 2 time points. We can then obtain the following imu constraints.
$$ 
r = F(s_i,\zeta)\ominus s_j
$$
We can build the optimization problem subject to the above constraints
* navigation state is combined by attitude $\theta_{nb}$, position $p_n$ and velocity $v_n$.   
* $\zeta$ is the preintegration imu measurement.


### PIM (preintegration measurement)
For the convenience, we integrate all the IMU measurements first, without considering the state of the robot and the gravity.
$$
\zeta = (\theta, p ,v)
$$
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
\tilde{\zeta_{k+1}} = \zeta_{k+1} + \Delta b_{acc} \frac{\partial{\zeta_{k+1}}}{\partial b_{acc}} +
\Delta b_{\omega} \frac{\partial{\zeta_{k+1}}}{\partial b_{\omega}} 
$$
* $b_{acc}$ is bias for accelerometer.
* $b_{\omega}$ is bias for gyroscope.
* ~ denotes the corrected measurement.
#### The jocabian matrix of bias for bias corrected mesurement.
$$
J^{\tilde{\zeta}}_{b_{all}} =  [ \frac{\partial{\zeta_{k+1}}}{\partial b_{acc}}, \frac{\partial{\zeta_{k+1}}}{\partial b_{\omega}}] 
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


### Navigation state correct

navigation state is combined by attitude $\theta_{nb}$, position $p_n$ and velocity $v_n$.   
$$
s = (\theta_{nb}, p_{n}, v_{n}) 
$$
We can correct $\zeta$ vector with given navigation state.   

$$
\bar{\theta_{b}} = \tilde{\theta_b} \\
\bar{p_{b}} = \tilde{p_b} + R_{nb}^{-1} v_n \Delta{t} + R_{nb}^{-1} g \frac{\Delta{t}^2}{2} \\
\bar{v_{b}} = \tilde{v_b} + R_{nb}^{-1} g \Delta{t} \\
$$
* $g$ is the gravity vector.
* $\bar{}$ denotes the bavigation state corrected measurement.


#### The jocabian matrix of navigation state
$$
J^{\bar{\zeta}}_{s}=\begin{bmatrix}
 0_{3\times3}  & 0_{3\times3} & 0_{3\times3}\\  
 \frac{\partial{\bar{p_{b}}}}{\partial \theta_b}   & 0_{3\times3} &  \frac{\partial\bar{{p_{b}}}}{\partial v_b} \\  
\frac{\partial{\bar{v_{b}}}}{\partial \theta_b}  &  0_{3\times3} &  0_{3\times3}\\   
\end{bmatrix} \\
$$
where:
$$
\frac{\partial{\bar{p_{b}}}}{\partial \theta_b} = \widehat{R_{nb}^{-1}v_n} \Delta{t} + \widehat{R_{nb}^{-1}g} \frac{\Delta{t}^2}{2}
$$

$$
\frac{\partial\bar{{p_{b}}}}{\partial v_b} = R_{nb}^{-1}R_{nb} \Delta{t} 
$$

$$
\frac{\partial{\bar{v_{b}}}}{\partial \theta_b} = \widehat{R_{nb}^{-1}g} \Delta{t}
$$
#### The jocabian matrix of bias
$$
J^{\bar{\zeta}}_{b_{all}}=\begin{bmatrix}
 I_{3\times3} & 0_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & I_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & 0_{3\times3} & I_{3\times3}\\   
\end{bmatrix} \\
$$

### Predict $s_j$
Up to then, we can use $\bar{\zeta}$ to predict $s_j$
$$s_j^* = s_j\oplus\bar{\zeta} $$
* $s_j^*$ is the predicted $s_j$.

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
