# Imu preintegration.  

## Predicting navigation state by IMU
Suppose we know the navigation state of the robot at time i ,as well as the IMU measurements from the i time to j time. We want predict the state of robot at time j.
$$ 
s_j^* = \mathscr{R}(s_i, \mathscr{D}(\xi(\zeta, b))) \\
$$
The navigation state combined by attitude $R(\theta)$, position $p$ and velocity $v$.   
$$
s_i = (R_{nb}, p_{nb}, v_{nb}) \\
s_j = (R_{nc}, p_{nc}, v_{nc}) \\
$$
* n denotes navigation state frame.
* b denotes body frame in time i.
* c denotes current frame in time j.
* $\theta$ is the lie algebra of R.

The retract action $\mathscr{R}$ which defined on navigation state  takes 2 parameters: $s_i$ and $\mathscr{D}$ to predict $s_j$. The $\mathscr{D}$ represents the difference between $s_i$ and $s_j$.
$$
\mathscr{R} = s_i \oplus d \\
d(\xi,s_i) = (\theta_{nc}, p_{nc}, v_{nc}) \\
$$
$\xi$ represents bias corrected preintegration measurement (PIM), which take 2 parameters, the PIM $\zeta$ and IMU bias b.

#### The Jacobian of $s_i$
$$
J^{s_j^*}_{s_i} = J^{\mathscr{R}}_{s_i} + J^{\mathscr{R}}_{\mathscr{D}} J^{\mathscr{D}}_{s_i}
$$ 
#### The Jacobian of $b$
$$
J^{s_j^*}_{b} = J^{\mathscr{R}}_{\mathscr{D}} J^{\mathscr{D}}_{\xi} J^{\xi}_{b}
$$ 

### Preintegration measurement (PIM)
The PIM $\zeta(R(\theta), p ,v)$ integrates all the IMU measurements  without considering the state of the bias and the gravity.
$\omega^b_k$,$a^b_k$ are the acceleration and angular velocity measured by IMU (accelerometer + gyroscope) respectively.
$$
R_{k+1} = R_k \exp(\omega^b_k\Delta{t}) \\
p_{k+1} = p_k + v_k\Delta{t} +R_k a^b_k \frac{\Delta{t}^2}{2} \\
v_{k+1} = v_k + R_k a^b_k \Delta{t}
$$
$n$: navigation frame, $b$: body frame.
#### A:Derivative of old $\zeta$
$$
A = \frac{\partial{\zeta_{k+1}}}{\partial \zeta_{k}} = 
\begin{bmatrix}
 \frac{\partial{R_{k+1}}}{\partial{R_{k}}}  & \frac{\partial{R_{k+1}}}{\partial{p_{k}}} &  \frac{\partial{R_{k+1}}}{\partial{v_{k}}}\\  
 \frac{\partial{p_{k+1}}}{\partial{R_{k}}}  & \frac{\partial{p_{k+1}}}{\partial{p_{k}}} &  \frac{\partial{p_{k+1}}}{\partial{v_{k}}}\\  
 \frac{\partial{v_{k+1}}}{\partial{R_{k}}}  & \frac{\partial{v_{k+1}}}{\partial{p_{k}}} &  \frac{\partial{v_{k+1}}}{\partial{v_{k}}}\\   
\end{bmatrix} 
=\begin{bmatrix}
 \frac{\partial{R_{k+1}}}{\partial{R_{k}}}  & 0_{3\times3} & 0_{3\times3}\\  
 \frac{\partial{p_{k+1}}}{\partial{R_{k}}}  & I_{3\times3} &  I_{3\times3} \Delta{t}\\  
 \frac{\partial{v_{k+1}}}{\partial{R_{k}}}  &  0_{3\times3} &  I_{3\times3}\\   
\end{bmatrix} \\
=\begin{bmatrix}
 I_{3\times3}-\Delta_{t}\widehat{\omega_{k}^{b}}  & 0_{3\times3} & 0_{3\times3}\\  
 -R_{k}\widehat{a_{k}^{b}}\frac{\Delta_{t}}{2}^{2} & I_{3\times3} &  I_{3\times3} \Delta{t}\\  
 -R_{k}\widehat{a_{k}^{b}}\Delta_{t}  &  0_{3\times3} &  I_{3\times3}\\   
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
 I_{3\times3}\Delta{t} \\  
 0_{3\times3}  \\  
 0_{3\times3}  \\   
\end{bmatrix} 
$$
### Bias correct
We want correct $\zeta$ by a given accelerometer and gyroscope bias.   
$$
\xi(b+\Delta{b}) = \zeta \oplus (\Delta b_{acc} \frac{\partial{\zeta}}{\partial b_{acc}} +
\Delta b_{\omega} \frac{\partial{\zeta}}{\partial b_{\omega}} )
$$
* $b_{acc}$ is bias for accelerometer.
* $b_{\omega}$ is bias for gyroscope.
* Because the parameter $\theta$ cannot be added directly, we define the combination of rotations with the symbol $\oplus$. 
$$
a\oplus b = [\log(\exp(\theta_a)\exp(\theta_b)), v_a+v_b, v_a+v_b]
$$

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
The $\mathscr{D}$ represents the difference between two $s_i$ and $s_j$.   
$$
\mathscr{D} = (R_{bc}, p_{bc}, v_{bc}) \\
$$
We can calculate $\mathscr{D}$ from corrected PIM $\xi(R_{bc}^{\xi},p_{bc}^{\xi},v_{bc}^{\xi})$ and velocity, which is included in $s_i$.

$$
\mathscr{D}(\xi,s_i)=\begin{bmatrix}
R_{bc}^{\xi}\\  
p_{bc}^{\xi} + R_{nb}^{-1} v_{nb} \Delta{t} + R_{nb}^{-1} g \frac{\Delta{t}^2}{2} \\  
v_{bc}^{\xi} + R_{nb}^{-1} g \Delta{t}\\   
\end{bmatrix}
$$

* $g$ is the gravity vector.
* $*$ denotes the predicted navigation state.


#### The jocabian matrix of navigation state
$$
J^{\mathscr{D}}_{s_i}=\begin{bmatrix}
 0_{3\times3}  & 0_{3\times3} & 0_{3\times3}\\  
 \frac{\partial{p_{bc}}}{\partial R_{nb}} & 0_{3\times3} &  \frac{\partial p_{bc}}{\partial v_{nb}} \\  
\frac{\partial{v_{bc}}}{\partial R_{nb}}  &  0_{3\times3} &  0_{3\times3}\\   
\end{bmatrix} \\
$$

where:
$$
\frac{\partial{p_{bc}}}{\partial R_{nb}} = \widehat{R_{nb}^{-1}v_{nb}} \Delta{t} + \widehat{R_{nb}^{-1}g} \frac{\Delta{t}^2}{2}
$$

$$
\frac{\partial\ p_{bc}}{\partial v_{nb}} = R_{nb}^{-1}R_{nb} \Delta{t} = I_{3\times3}\Delta{t}
$$

$$
\frac{\partial{ v_{bc}}}{\partial R_{nb}} = \widehat{R_{nb}^{-1}g} \Delta{t}
$$
#### The jocabian matrix of $\xi$
$$
J^{\mathscr{D}}_{\xi}=\begin{bmatrix}
 I_{3\times3} & 0_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & I_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & 0_{3\times3} & I_{3\times3}\\   
\end{bmatrix} \\
$$


### Retraction $\mathscr{R}$
The retract action $\mathscr{R}$ which defined on navigation state  takes 2 parameters: $s_i$ and $\mathscr{D}$ to predict $s_j$.
* $s_j^*$ is the predicted $s_j$.

$$
R_{nc}^{*} = R_{nb}R_{bc} \\
p_{nc}^{*} = p_{nb} + R_{nb} p_{bc} \\
v_{nc}^{*} = v_{nb} + R_{nb} v_{bc}
$$
#### Derivative of $s_i$
$$
J^{\mathscr{R}}_{s_i}=\begin{bmatrix}
 R_{bc}^{-1} & 0_{3\times3} & 0_{3\times3}\\  
 -R_{bc}^{-1} \widehat{p_{bc}} & R_{bc}^{-1} & 0_{3\times3}\\  
 -R_{bc}^{-1} \widehat{v_{bc}} & 0_{3\times3} & R_{bc}^{-1}\\   
\end{bmatrix} \\
$$

#### Derivative of $d$
$$
J^{\mathscr{R}}_{d}=\begin{bmatrix}
 I_{3\times3} & 0_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & R_{bc}^{-1} & 0_{3\times3}\\  
 0_{3\times3} & 0_{3\times3} & R_{bc}^{-1}\\   
\end{bmatrix} \\
$$

## Navigation state prediction error (residual function)

If navigtion $state_j$ is measured by sensors, we can calculate the error between $state_j$ and $state_j^*$.

$$
r_{jj^*}=\mathscr{L}(s_j,s_j^*) =
\begin{bmatrix}
 \Delta{R} \\  
 \Delta{p}  \\  
 \Delta{v} \\   
\end{bmatrix} =
\begin{bmatrix}
 R_j^{-1} R_j^* \\  
 R_j^{-1} (p_j^* - p_j)  \\  
 R_j^{-1} (v_j^* - v_j) \\   
\end{bmatrix} \\
$$
Local $\mathscr{L}$  is the inverse function of $\mathscr{R}$, which takes two navigation states, and get the delta between the two states in tangent vector space
#### Derivative of an $s_j$
$$
J^{\mathscr{L}}_{s_j}=\begin{bmatrix}
 -\Delta{R}  & 0_{3\times3} & 0_{3\times3}\\  
 \widehat{\Delta{p}} & -I_{3\times3} & 0_{3\times3}\\  
 \widehat{\Delta{v}} & 0_{3\times3} &-I_{3\times3}\\   
\end{bmatrix} \\
$$
#### Derivative of an $s_j^*$
$$
J^{\mathscr{L}}_{s_j^*}=\begin{bmatrix}
 I_{3\times3}& 0_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & \Delta{R} & 0_{3\times3}\\  
 0_{3\times3} & 0_{3\times3} &\Delta{R}\\   
\end{bmatrix} \\
$$

### Overall Jaccobian for prediction error

To summarize, The prediction error $r$ takes 3 parameters $s_i$, $s_j$ and $b$. According to the chain rule, their Jaccobian can be written in the following form.
$$
J^r_{s_j} = J^{\mathscr{L}}_{s_j}
$$
$$
J^r_{s_i} = J^{\mathscr{L}}_{s_j^*} J^{s_j^*}_{s_i} = 
J^{\mathscr{L}}_{s_j^*}(J^{\mathscr{R}}_{s_i} + J^{\mathscr{R}}_{\mathscr{D}} J^{\mathscr{D}}_{s_i})
$$
$$
J^r_{b} = J^{\mathscr{L}}_{s_j^*} J^{s_j^*}_{b} = 
J^{\mathscr{L}}_{s_j^*}J^{\mathscr{R}}_{\mathscr{D}} J^{\mathscr{D}}_{\xi} J^{\xi}_{b}
$$

---

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

