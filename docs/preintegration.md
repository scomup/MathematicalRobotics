# Imu preintegration.  

## Predicting navigation state by IMU
Suppose we know the navigation state of the robot at time i ,as well as the IMU measurements from the i time to j time. We want predict the state of robot at time j.

$$ 
s_j^* = \mathscr{R}(s_i, \mathscr{D}(\xi(\zeta, b))) 
\tag{1}
$$

The navigation state combined by attitude $R(\theta)$, position $p$ and velocity $v$.   

$$
s_i = (R_{nb}, p_{nb}, v_{nb}) \\
s_j = (R_{nc}, p_{nc}, v_{nc}) 
\tag{2}
$$

* n denotes navigation state frame.
* b denotes body frame in time i.
* c denotes current frame in time j.
* $\theta$ is the lie algebra of R.

The retract action $\mathscr{R}$ which defined on navigation state  takes 2 parameters: $s_i$ and $\mathscr{D}$ to predict $s_j$. The $\mathscr{D}$ represents the difference between $s_i$ and $s_j$.

$$
d(\xi,s_i) = (R_{nc}, p_{nc}, v_{nc}) 
\tag{3}
$$

$\xi$ represents bias corrected preintegration measurement (PIM), which take 2 parameters, the PIM $\zeta$ and IMU bias b.

#### The Jacobian of $s_i$

$$
J^{s_j^*}_{s_i} = J^{\mathscr{R}}_{s_i} + J^{\mathscr{R}}_{\mathscr{D}} J^{\mathscr{D}}_{s_i} 
\tag{4}
$$ 

#### The Jacobian of $b$

$$
J^{s_j^*}_{b} = J^{\mathscr{R}}_{\mathscr{D}} J^{\mathscr{D}}_{\xi} J^{\xi}_{b}  
\tag{5}
$$ 

### Preintegration measurement (PIM)
The PIM $\zeta(R(\theta), p ,v)$ integrates all the IMU measurements  without considering the IMU bias and the gravity.
$\omega^b_k$,$a^b_k$ are the acceleration and angular velocity measured by IMU (accelerometer + gyroscope) respectively.

$$
\begin{aligned}
R_{k+1} &= R_k \exp(\omega^b_k\Delta{t}) \\
p_{k+1} &= p_k + v_k\Delta{t} +R_k a^b_k \frac{\Delta{t}^2}{2} \\
v_{k+1} &= v_k + R_k a^b_k \Delta{t}
\end{aligned} 
\tag{7}
$$

$n$: navigation frame, $b$: body frame.
#### A:Derivative of old $\zeta$

$$
\begin{aligned}
A &= \frac{\partial{\zeta_{k+1}}}{\partial \zeta_{k}}  \\
&=\begin{bmatrix}
 \frac{\partial{R_{k+1}}}{\partial{R_{k}}}  & \frac{\partial{R_{k+1}}}{\partial{p_{k}}} &  \frac{\partial{R_{k+1}}}{\partial{v_{k}}}\\  
 \frac{\partial{p_{k+1}}}{\partial{R_{k}}}  & \frac{\partial{p_{k+1}}}{\partial{p_{k}}} &  \frac{\partial{p_{k+1}}}{\partial{v_{k}}}\\  
 \frac{\partial{v_{k+1}}}{\partial{R_{k}}}  & \frac{\partial{v_{k+1}}}{\partial{p_{k}}} &  \frac{\partial{v_{k+1}}}{\partial{v_{k}}}\\   
\end{bmatrix} \\
&=\begin{bmatrix}
 \exp{(-\omega^b_k\Delta{t})}  & 0_{3\times3} & 0_{3\times3}\\  
 -R_{k}\widehat{a_{k}^{b}}\frac{\Delta{t}}{2}^{2} & I_{3\times3} &  I_{3\times3} \Delta{t}\\  
 -R_{k}\widehat{a_{k}^{b}}\Delta{t}  &  0_{3\times3} &  I_{3\times3}\\   
\end{bmatrix}
\end{aligned} 
\tag{8}
$$

#### B:Derivative of input $a$

$$
B = \frac{\partial{\zeta_{k+1}}}{\partial a^b_k} = 
\begin{bmatrix}
 \frac{\partial{R_{k+1}}}{\partial{a^b_k}} \\  
 \frac{\partial{p_{k+1}}}{\partial{a^b_k}}  \\  
 \frac{\partial{v_{k+1}}}{\partial{a^b_k}}  \\   
\end{bmatrix} 
=\begin{bmatrix}
 0_{3\times3} \\  
 R_{k}\frac{\Delta{t}^2}{2}  \\  
 R_{k} \Delta{t}  \\   
\end{bmatrix}  
\tag{9} 
$$

#### C:Derivative of input $\omega$

$$
C = \frac{\partial{\zeta_{k+1}}}{\partial \omega^b_k} = 
\begin{bmatrix}
 \frac{\partial{R_{k+1}}}{\partial{\omega^b_k}} \\  
 \frac{\partial{p_{k+1}}}{\partial{\omega^b_k}}  \\  
 \frac{\partial{v_{k+1}}}{\partial{\omega^b_k}}  \\   
\end{bmatrix}  
=\begin{bmatrix}
 H(\omega^b_k)\Delta{t} \\  
 0_{3\times3}  \\  
 0_{3\times3}  \\   
\end{bmatrix} 
\tag{10}
$$

Where $H$ is the Jocabian for $\exp$: $\exp(a+\delta{x}) = \exp(a) + H(a)\delta{x}$

### Bias correct
We want correct $\zeta$ by a given accelerometer and gyroscope bias.   

$$
\xi(b+\Delta{b}) = \zeta \oplus (\Delta b_{acc} \frac{\partial{\zeta}}{\partial b_{acc}} +
\Delta b_{\omega} \frac{\partial{\zeta}}{\partial b_{\omega}} ) 
\tag{11}
$$

* $b_{acc}$ is bias for accelerometer.
* $b_{\omega}$ is bias for gyroscope.
* Because the parameter $\theta$ cannot be added directly, we define the combination of $\zeta$ with the symbol $\oplus$. 

$$
a\oplus b = [\log(\exp(\theta_a)\exp(\theta_b)), p_a+p_b, v_a+v_b]
\tag{12}
$$

#### The jocabian of bias for corrected PIM.

$$
J^{\xi}_{b} =  [ \frac{\partial{\zeta}}{\partial b_{acc}}, \frac{\partial{\zeta}}{\partial b_{\omega}}] 
\tag{13}
$$

#### Find the partial derivatives of accelerometer's bias

The bias model for accelerometer.

$$
\tilde{a^b_k} = a^b_k - b_{acc}  
\tag{14}
$$

$$
\frac{\partial{\zeta_{k+1}}}{\partial b_{acc}} =
\frac{\partial{\zeta_{k+1}}}{\partial \zeta_{k}} \frac{\partial{\zeta_{k}}}{\partial b_{acc}} +
\frac{\partial{\zeta_{k+1}}}{\partial \tilde{a^b_k}} \frac{\partial{\tilde{a^b_k}}}{\partial b_{acc}} \\
= A \frac{\partial{\zeta_{k}}}{\partial b_{acc}} - B 
\tag{15}
$$

#### Find the partial derivatives of gyroscope's bias

$$
\tilde{\omega^b_k} = \omega^b_k - b_{\omega} 
\tag{16}
$$

$$
\frac{\partial{\zeta_{k+1}}}{\partial b_{\omega}} =
\frac{\partial{\zeta_{k+1}}}{\partial \zeta_{k}} \frac{\partial{\zeta_{k}}}{\partial b_{\omega}} +
\frac{\partial{\zeta_{k+1}}}{\partial \tilde{\omega^b_k}} \frac{\partial{\tilde{\omega^b_k}}}{\partial b_{\omega}} \\
= A \frac{\partial{\zeta_{k}}}{\partial b_{\omega}} - C 
\tag{17}
$$

* ~ denotes the corrected measurement.


### Delta between two states
The $\mathscr{D}$ represents the difference between two $s_i$ and $s_j$.   

$$
\mathscr{D} = (R_{bc}, p_{bc}, v_{bc}) \tag{18}
$$

We can calculate $\mathscr{D}$ from corrected PIM $\xi(R_{bc}^{\xi},p_{bc}^{\xi},v_{bc}^{\xi})$ and velocity, which is included in $s_i$.

$$
\mathscr{D}(\xi,s_i)=\begin{bmatrix}
R_{bc}^{\xi}\\  
p_{bc}^{\xi} + R_{nb}^{-1} v_{nb} \Delta{t} + R_{nb}^{-1} g \frac{\Delta{t}^2}{2} \\  
v_{bc}^{\xi} + R_{nb}^{-1} g \Delta{t}\\   
\end{bmatrix} 
\tag{19}
$$

* $g$ is the gravity vector.
* $*$ denotes the predicted navigation state.


#### The jocabian matrix of navigation state

$$
\begin{aligned}
J^{\mathscr{D}}_{s_i}
&=
\begin{bmatrix}
 0_{3\times3}  & 0_{3\times3} & 0_{3\times3}\\  
 \frac{\partial{p_{bc}}}{\partial R_{nb}} & 0_{3\times3} &  \frac{\partial p_{bc}}{\partial v_{nb}} \\  
\frac{\partial{v_{bc}}}{\partial R_{nb}}  &  0_{3\times3} &  0_{3\times3}\\   
\end{bmatrix} \\
&=
\begin{bmatrix}
 0_{3\times3}  & 0_{3\times3} & 0_{3\times3}\\  
 \widehat{R_{nb}^{-1}v_{nb}} \Delta{t} + \widehat{R_{nb}^{-1}g} \frac{\Delta{t}^2}{2} & 0_{3\times3} &  I_{3\times3}\Delta{t} \\  
\widehat{R_{nb}^{-1}g} \Delta{t}   &  0_{3\times3} &  0_{3\times3}\\   
\end{bmatrix}
\end{aligned} 
\tag{20}
$$

#### The jocabian matrix of $\xi$

$$
J^{\mathscr{D}}_{\xi}=\begin{bmatrix}
 I_{3\times3} & 0_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & I_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & 0_{3\times3} & I_{3\times3}\\   
\end{bmatrix} 
\tag{21}
$$


### Retraction $\mathscr{R}$
The retract action $\mathscr{R}$ which defined on navigation state  takes 2 parameters: $s_i$ and $\mathscr{D}$ to predict $s_j$.
* $s_j^*$ is the predicted $s_j$.

$$
\begin{aligned}
R_{nc}^{*} &= R_{nb}R_{bc} \\
p_{nc}^{*} &= p_{nb} + R_{nb} p_{bc} \\
v_{nc}^{*} &= v_{nb} + R_{nb} v_{bc} 
\end{aligned}
\tag{22}
$$

#### Derivative of $s_i$

$$
J^{\mathscr{R}}_{s_i}=\begin{bmatrix}
 R_{bc}^{-1} & 0_{3\times3} & 0_{3\times3}\\  
 -R_{bc}^{-1} \widehat{p_{bc}} & R_{bc}^{-1} & 0_{3\times3}\\  
 -R_{bc}^{-1} \widehat{v_{bc}} & 0_{3\times3} & R_{bc}^{-1}\\   
\end{bmatrix} 
\tag{23}
$$

#### Derivative of $d$

$$
J^{\mathscr{R}}_{d}=\begin{bmatrix}
 I_{3\times3} & 0_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & R_{bc}^{-1} & 0_{3\times3}\\  
 0_{3\times3} & 0_{3\times3} & R_{bc}^{-1}\\   
\end{bmatrix} 
\tag{24}
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
\end{bmatrix} 
\tag{25}
$$

Local $\mathscr{L}$  is the inverse function of $\mathscr{R}$, which takes two navigation states, and get the delta between the two states in tangent vector space
#### Derivative of an $s_j$

$$
J^{\mathscr{L}}_{s_j}=\begin{bmatrix}
 -\Delta{R}^{-1}  & 0_{3\times3} & 0_{3\times3}\\  
 \widehat{\Delta{p}} & -I_{3\times3} & 0_{3\times3}\\  
 \widehat{\Delta{v}} & 0_{3\times3} &-I_{3\times3}\\   
\end{bmatrix} 
\tag{26}
$$

#### Derivative of an $s_j^*$

$$
J^{\mathscr{L}}_{s_j^*}=\begin{bmatrix}
 I_{3\times3}& 0_{3\times3} & 0_{3\times3}\\  
 0_{3\times3} & \Delta{R} & 0_{3\times3}\\  
 0_{3\times3} & 0_{3\times3} &\Delta{R}\\   
\end{bmatrix} 
\tag{27}
$$

### Overall Jaccobian for prediction error

To summarize, The prediction error $r$ takes 3 parameters $s_i$, $s_j$ and $b$. According to the chain rule, their Jaccobian can be written in the following form.

$$
J^r_{s_j} = J^{\mathscr{L}}_{s_j}
\tag{28}
$$

$$
J^r_{s_i} = J^{\mathscr{L}}_{s_j^*} J^{s_j^*}_{s_i} = 
J^{\mathscr{L}}_{s_j^*}(J^{\mathscr{R}}_{s_i} + J^{\mathscr{R}}_{\mathscr{D}} J^{\mathscr{D}}_{s_i})
\tag{29}
$$

$$
J^r_{b} = J^{\mathscr{L}}_{s_j^*} J^{s_j^*}_{b} = 
J^{\mathscr{L}}_{s_j^*}J^{\mathscr{R}}_{\mathscr{D}} J^{\mathscr{D}}_{\xi} J^{\xi}_{b}
\tag{30}
$$

---

# Appendix

### A-1. Proof of [Preintegration measurement (PIM)] (8)(9)(10)
$A$ and $B$ are the two lie groups: $\varphi(A,B) = AB $

$$
\begin{aligned}
    \exp(\widehat{J_A\delta}) 
    &= (AB)^{-1}(A\exp(\hat{\delta})B) \\
    &= B^{-1}A^{-1}A\exp(\hat{\delta})B \\
    &= \exp(B^{-1}\hat{\delta} B) \\
    &= \exp(\widehat{B^{-1}\delta})
\end{aligned} 
$$

$$
\begin{aligned}
    \exp(\widehat{J_B\delta}) 
    &= (AB)^{-1}(AB\exp(\hat{\delta})) \\
    &= B^{-1}A^{-1}AB\exp(\hat{\delta}) \\
    &= \exp(\hat{\delta}) \\
\end{aligned}
$$

Hence:

$$
   J_A = B^{-1} 
   \tag{A1-1}
$$

$$
   J_B = I 
   \tag{A1-2}
$$

#### Proof of (8) $J_{\zeta_k}^{\zeta_{k+1}}$:
According to A1-1:

$$
\frac{\partial{R_{k+1}}}{\partial{R_{k}}} 
= \exp(-\omega^b_k \Delta t) \\
= I_{3\times3}-\Delta{t}\widehat{\omega_{k}^{b}}
$$

$A$ is a lie group, p is a vector: $\varphi(A,p) = Ap$


$$
\begin{aligned}
J_A 
&= \frac{A\exp\left( \delta \right)p - Ap}{\delta} \\
&\cong \frac{Aa + A\widehat{\delta}p - Ap}{\delta}  \\
&= \frac{A\widehat{\delta}p}{\delta}  \\
&= - \frac{A\delta\widehat{p}}{\delta}  \\
&= - A\widehat{p} 
\end{aligned}
\tag{A1-3}
$$

$$
\begin{aligned}
J_p &= \frac{A (p + \delta) - Ap}{\delta} \\
&= A
\end{aligned}
\tag{A1-4}
$$

According to A1-3:

$$
\frac{\partial{p_{k+1}}}{\partial{R_{k}}} 
= -R_{k}\widehat{a_{k}^{b}}\frac{\Delta{t}}{2}^{2}
$$

$$
\frac{\partial{v_{k+1}}}{\partial{R_{k}}} 
= -R_{k}\widehat{a_{k}^{b}}\Delta{t} 
$$

#### Proof of (9) $J_{a^b_k}^{\zeta_{k+1}}$:
According to A1-4:

$$
\frac{\partial{p_{k+1}}}{\partial{a^b_k}} 
= R_{k}\frac{\Delta{t}}{2}^{2}
$$

$$
\frac{\partial{v_{k+1}}}{\partial{a^b_k}} 
= R_{k}\Delta{t}
$$

#### Proof of (10) $J_{\omega^b_k}^{\zeta_{k+1}}$:
According to A1-2:

$$
\frac{\partial{R_{k+1}}}{\partial{\omega^b_k}} 
= I_{3 \times 3}\Delta{t}
$$

 ### A-2. Proof of [Delta between two states] (20)

$A$ is a lie group, p is a vector: $\varphi(A,p) = A^{-1}p$

$$
\begin{aligned}
J_A 
&= \frac{(A \exp(\hat{\delta}))^{-1} p - A^{-1}p}{\delta} \\ 
&= \frac{ \exp(\widehat{-\delta}) A^{-1} p - A^{-1}p}{\delta} \\
&= \frac{ (I - \hat{\delta}) A^{-1} p - A^{-1}p}{\delta} \\
&= \frac{ -\hat{\delta} A^{-1} p}{\delta} \\
&= \frac{ \delta{x} \widehat{A^{-1} p}}{\delta} \\
&= \widehat{A^{-1} p}
\end{aligned}
\tag{A2-1}
$$

$$
\begin{aligned}
J_p 
&= \frac{T^{-1} (p + \delta)-T^{-1}p }{\delta} \\ 
&= \frac{T^{-1} \delta}{\delta} \\
&= T^{-1}
\end{aligned}
\tag{A2-2}
$$

#### Proof of (20) $J_{s_i}^{\mathscr{D}}$
The $\mathscr{D}$ function:

$$
\mathscr{D}(\xi,s_i)=\begin{bmatrix}
R_{bc}^{\xi}\\  
p_{bc}^{\xi} + R_{nb}^{-1} v_{nb} \Delta{t} + R_{nb}^{-1} g \frac{\Delta{t}^2}{2} \\  
v_{bc}^{\xi} + R_{nb}^{-1} g \Delta{t}\\   
\end{bmatrix} 
$$

According to A2-1: 

$$
\frac{\partial{p_{bc}}}{\partial R_{nb}} =
\widehat{R_{nb}^{-1}v_{nb}} \Delta{t} + \widehat{R_{nb}^{-1}g} \frac{\Delta{t}^2}{2}
$$

$$
\frac{\partial{v_{bc}}}{\partial R_{nb}} =
\widehat{R_{nb}^{-1}g} \Delta{t}
$$

According to A2-2 and (22): 

$$
 \frac{\partial p_{bc}}{\partial v_{nb}} =
 \frac{ R_{nb}^{-1}(v_{nb} + R_{nb}\delta v_b)- R_{nb}^{-1}v_{nb}}{\delta v_b} \\
 = I_{3\times3}\Delta{t}
$$

### A-3. Proof of Retraction $\mathscr{R}$ (23)(24)
The $\mathscr{R}$ function:
 
$$
\begin{aligned}
R_{nc}^{*} &= R_{nb}R_{bc} \\
p_{nc}^{*} &= p_{nb} + R_{nb} p_{bc} \\
v_{nc}^{*} &= v_{nb} + R_{nb} v_{bc} 
\end{aligned}
$$

The Jacobian of x for F:

$$
J^F_x
=\frac{\mathscr{L}(F(x),F(\mathscr{R(x,\delta{x})}))}{\delta x}
\tag{A3-1}
$$

#### Proof of (23) $J^\mathscr{R}_{s_i}$:
According to A1-1: 

$$
\frac{\partial{R^*_{nc}}}{\partial R_{nb}} = R_{bc}^{-1}
$$

According to A2-2 and A3-1: 

$$
\frac{\partial{p_{nc}^*}}{\partial R_{nb}} 
=\frac{ R_{nc}^{-1}( R_{nb}\exp(\widehat{\delta \theta_{b}})p_{bc}- R_{nb}p_{bc})}{\delta \theta_{b}} \\
= -R^{-1}_{bc} \widehat{p_{bc}}
$$

$$
\frac{\partial{v_{nc}^*}}{\partial R_{nb}} =
\frac{ R_{nc}^{-1}( R_{nb}\exp(\widehat{\delta \theta_{b}})v_{bc}- R_{nb}v_{bc})}{\delta \theta_{b}} \\
= -R^{-1}_{bc} \widehat{v_{bc}}
$$


According to A1-3 and (22)(25): 

$$
\frac{\partial{p_{bc}^*}}{\partial p_{nb}} =
\frac{ R_{nc}^{-1}( p_{nb} + R_{nb}\delta{p_{b}} -  p_{nb})}{\delta{p_{b}}} \\
= R^{-1}_{bc}
$$

$$
\frac{\partial{v_{bc}^*}}{\partial v_{nb}} =
\frac{ R_{nc}^{-1}( v_{nb} + R_{nb}\delta v_{b} -  v_{nb})}{\delta v_{b}} \\
= R^{-1}_{bc}
$$

 #### Proof of (24) $J^\mathscr{R}_{d}$

According to A2-2 and A3-1: 

$$
\frac{\partial{p_{nc}^*}}{\partial p_{bc}} =
\frac{ R_{nc}^{-1}( R_{nb} (p_{bc} + \delta p_{b})- R_{nb}p_{bc})}{\delta p_{b}} \\
= R^{-1}_{bc}
$$

$$
\frac{\partial{v_{nc}^*}}{\partial v_{b}} =
\frac{ R_{nc}^{-1}( R_{nb} (v_{bc} + \delta v_{b})- R_{nb}v_{bc})}{\delta v_{b}} \\
= R^{-1}_{bc}
$$

### A-4. Proof of Local $\mathscr{L}$ (23)(24)
The $\mathscr{L}$ function:

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
\end{bmatrix} 
$$




