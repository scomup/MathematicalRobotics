# Filter

## Bayesian Filter

A Bayesian filter estimates the uncertain state of a system. It operates sequentially by predicting the next state from the past state distribution and control input, and then correcting that prediction using new observation data.

**Prediction Step**  
In this step, the “past state” and “control input” are used to predict the “next state” probabilistically:

$$ \bar{p}(x_t) = \int p(x_t \mid u, x_{t-1}) p(x_{t-1}) \, dx_{t-1} \tag{1} $$

- **Motion model** $p(x_t \mid u, x_{t-1})$: The probability of reaching state $x_t$ from $x_{t-1}$ with control input $u$.
- **Past state distribution** $p(x_{t-1})$: The probability distribution of the previous state, derived from the last update.

This step uses the Markov property, assuming $x_t$ depends only on $x_{t-1}$ and $u$.

**Correction Step**  
This step updates the “predicted state distribution” using new observation data to improve the current state estimate:

$$ p(x_t \mid z_t) = \eta \, p(z_t \mid x_t) \, \bar{p}(x_t) \tag{2} $$

- **Predicted distribution** $\bar{p}(x_t)$: The prior distribution from the prediction step.
- **Observation model** $p(z_t \mid x_t)$: The probability of observations $z_t$ given state $x_t$, incorporating sensor data.
- **Normalization constant** $\eta$: Ensures the total probability is 1.

Using Bayes' theorem, this step refines the prior with new observation $z_t$, yielding the posterior $p(x_t \mid z_t)$. This updated state passes forward as the prior in the next cycle, continuously refining the system's state estimate in real time.

# Features of Each Filtering Technique

In a Bayesian filter, distributions $p(x)$, $p(x_t \mid x_{t-1}, u)$, and $p(z_t \mid x_t)$ are all probabilities. You must model them accurately for proper computation.

Differences in how these distributions are modeled lead to two main filtering approaches:

- **Extended Kalman Filter (EKF)**:
  - Assumes state uncertainty follows a Gaussian (normal) distribution.
  - Cannot handle cases where the Gaussian assumption significantly fails.

- **Particle Filter (PF)**:
  - Does not assume a Gaussian form; instead, uses many sampled points (“particles”) to represent the uncertainty.
  - Flexible and can handle nonlinear or complex distributions but is computationally more expensive since processing increases with the number of samples.

## Extended Kalman Filter (EKF)

In many systems, using a Gaussian to represent uncertainty works well.

### Prediction Step  

Assume at time $t-1$ the state’s probability distribution is Gaussian:

$$ p(x_{t-1}) = \mathcal{N}(\bar{x}_{t-1}, P_{t-1}) \tag{3} $$

Here $\bar{x}_{t-1}$ and $P_{t-1}$ are the mean and covariance matrix. The overbar on $x$ ($\bar{x}$) represents the mean or predicted value.

Covariance $P_{t-1}$ indicates how uncertain you are about the state. Formally:

$$ P_{t-1} = \mathbb{E}\bigl((x_{t-1} - \bar{x}_{t-1})(x_{t-1} - \bar{x}_{t-1})^T\bigr) = \mathbb{E}(\Delta x_{t-1}, \Delta x_{t-1}^T) \tag{4} $$

where $\Delta x_{t-1} = x_{t-1} - \bar{x}_{t-1}$.

The robot’s motion is described by the motion equation $f$, which computes the next state $x_t$ from $x_{t-1}$ and a control $u$:

$$ x_t = f(x_{t-1}, u) \tag{5} $$

Control $u$ also has uncertainty, often modeled by a Gaussian:

$$ p(u) = \mathcal{N}(\bar{u}, Q) \tag{6} $$

Here $Q$ is the covariance of the control input $u$:

$$ Q = \mathbb{E}((u - \bar{u})(u - \bar{u})^T) = \mathbb{E}(\Delta u, \Delta u^T) \tag{7} $$

Ignoring error, the predicted mean is:

$$ \bar{x}_t = f(\bar{x}_{t-1}, \bar{u}) \tag{8} $$

We want to quantify this spread. From equation (5), $x_{t-1}$ and $u$ are uncertain, so the resulting distribution of next state ($\bar{x}_t$) might not strictly be Gaussian. But EKF assumes it remains Gaussian, with covariance $\bar{P}_t$. The steps to compute $\bar{P}_t$ follow.

Using (4):

$$ \bar{P}_t = \mathbb{E}\bigl((x_t - \bar{x}_t)(x_t - \bar{x}_t)^T\bigr) \tag{9} $$

Substituting (5) into (9):

$$ \bar{P}_t = \mathbb{E}\bigl(\bigl(f(x_{t-1}, u) - f(\bar{x}_{t-1}, \bar{u})\bigr), \bigl(f(x_{t-1}, u) - f(\bar{x}_{t-1}, \bar{u})\bigr)^T\bigr) \tag{10} $$

Letting $\Delta x = x_{t-1} - \bar{x}_{t-1}$ and $\Delta u = u - \bar{u}$, and assuming these are small, we linearize $f$ with a first-order Taylor expansion:

$$ 
f(x_{t-1}, u) \approx f(\bar{x}_{t-1}, \bar{u}) + F_x \Delta x + F_u \Delta u \tag{11} 
$$

where

$$
F_x = \frac{\partial f}{\partial x}\bigg|_{x = \bar{x}_{t-1}, u = \bar{u}}, 
\quad F_u = \frac{\partial f}{\partial u}\bigg|_{x = \bar{x}_{t-1}, u = \bar{u}} \tag{12} 
$$

Substituting this into equation (10) yields:

$$
\bar{P} _t = F_x \mathrm{E}(\Delta \mathbf{x} \Delta \mathbf{x}^T) F_x^T + 2 F_x \mathrm{E}(\Delta \mathbf{x} \Delta \mathbf{u}^T) F_u^T + F_u \mathrm{E}(\Delta \mathbf{u} \Delta \mathbf{u}^T) F_u^T
$$

Since \( x \) and \( u \) are independent, \( \mathrm{E}(\Delta \mathbf{x} \Delta \mathbf{u}^T) = 0 \). Furthermore, substituting equations (4) and (7) yields:

$$
\bar{P}_t = F_x P _{t-1} F_x^T + F_u Q F_u^T
\tag{14}
$$

### Correction Step

In the correction step, the predicted state $\bar{x}_t$ is adjusted using the observation data to estimate a more accurate state $\hat{x}_t$. Here, the hat symbol ($\hat{\quad}$) denotes the corrected value.

To obtain the corrected state $\hat{x}_t$, the update $\Delta{x}_t$ is first defined based on the predicted state $\bar{x}_t$ from the prediction step:

$$
\hat{x}_t = \bar{x}_t + \Delta{x}_t
\tag{15}
$$

The correction step calculation requires two main components:

1. The formula for the update $\Delta{x}_t$ to compute $\hat{x}_t$.
2. The formula for the corrected covariance matrix $\hat{P}_t$ (which represents the uncertainty in $\hat{x}_t$).

In this section, we will derive these two equations from theory.

Firstly, let's define the sensor observation model. The sensor observation is based on the true state \( x_t \), and the observation equation \( h \) gives the observation \( z \). This observation includes an error \( v \), represented by a covariance \( R \).

$$
p(z \mid x_t) = \mathcal{N}(h(x_t), R)
\tag{16}
$$

$$
z = h(x_t) + v
\tag{17}
$$

From the predicted state \( \bar{x}_t \), the corresponding observation can also be predicted. The difference between the actual observation \( z \) and the predicted observation \( h(\bar{x}_t) \) is defined as the observation residual \( y \).

$$
y \equiv z - h(\bar{x}_t)
\tag{18}
$$

The covariance matrix of the observation residual \( y \) is defined as \( S \) and can be derived as follows:

$$
\begin{aligned}
S &= E(yy^T) \\
&= E((H\Delta{x}_t + v)(H\Delta{x}_t + v)^T) \\
&= E(H\Delta{x}_t \Delta{x}_t^T H^T) + E(vv^T) \\
&= H\bar{P}_t H^T + R
\end{aligned}
\tag{19}
$$

The above calculation uses the linearization of \( h \) (i.e., \( h(x_t) \approx h(\bar{x}_t) + H \Delta{x}_t \)), where \( H \) is the Jacobian matrix of the observation model.

The ideal \( \Delta{x}_t \) is obtained by minimizing the discrepancy between the observation residual \( y \) and the prediction error \( \Delta{x}_t \). Therefore, an objective function can be formulated to minimize the total error with respect to \( \Delta{x}_t \), as shown in Equation (20):

$$
\begin{aligned}
\text{argmin} \quad F(\bar{x} + \Delta{x}_t) &= \lVert y \rVert^2_S + \lVert \Delta{x}_t \rVert^2_P \\
&= \lVert z - h(\bar{x}_t) \rVert^2_S + \lVert x - \bar{x}_t \rVert^2_P \\
&= (z - h(\bar{x}_t)) S^{-1} (z - h(\bar{x}_t))^T + (x - \bar{x}_t) P^{-1} (x - \bar{x}_t)^T
\end{aligned}
\tag{20}
$$

By differentiating the objective function with respect to \( \Delta{x}_t \) and setting it to zero, the optimal \( \Delta{x}_t \) can be found:

$$
\begin{aligned}
0 &= 2 H^T S^{-1} (z - h(\bar{x}_t)) + 2 P^{-1} \Delta{x}_t \\
\Delta{x}_t &= P H^T S^{-1} (z - h(\bar{x}_t)) \\
\Delta{x}_t &= P H^T S^{-1} y
\end{aligned}
\tag{21}
$$

The term \( P H^T S^{-1} \) is defined as the **Kalman Gain** \( K \):

$$
K \equiv P H^T S^{-1}
\tag{22}
$$

By substituting Equations (21) and (22) into Equation (15), the corrected state \( \hat{x}_t \) can be calculated as follows:

$$
\hat{x}_t = \bar{x}_t + K y
\tag{23}
$$

Next, we evaluate the corrected covariance matrix \( \hat{P}_t \). The corrected covariance matrix is computed based on the mean square error between the true state \( x_t \) and the corrected state \( \hat{x}_t \):

$$
\hat{P}_t = E((x_t - \hat{x}_t) (x_t - \hat{x}_t)^T)
\tag{24}
$$

Expanding this equation:

$$
\begin{aligned}
\hat{P}_t &= E((x_t - \bar{x}_t - K y) (x_t - \bar{x}_t - K y)^T) \\
&= E\left[(x_t - \bar{x}_t) (x_t - \bar{x}_t)^T - (x_t - \bar{x}_t) (K y)^T - (K y) (x_t - \bar{x}_t)^T + (K y) (K y)^T\right]
\end{aligned}
\tag{25}
$$

* First term: \( E[(x_t - \bar{x}_t) (x_t - \bar{x}_t)^T] = \bar{P}_t \)

* Second and third terms:
$$
\begin{aligned}
&E[(x_t - \bar{x}_t) (h(x_t) + v - h(\bar{x}_t))^T K^T] \\
=& E[(x_t - \bar{x}_t) (h(\bar{x}_t) + H (x_t - \bar{x}_t) + v - h(\bar{x}_t))^T K^T] \\
=& E[(x_t - \bar{x}_t) (H (x_t - \bar{x}_t) + v)^T K^T] \\
=& E[(x_t - \bar{x}_t) (x_t - \bar{x}_t)^T H^T K^T] \\
=& \bar{P}_t H^T K^T \\
=& K H \bar{P}_t
\end{aligned}
\tag{26}
$$

* Fourth term:

$$
E((K y) (K y)^T) = K E(yy^T) K^T = K S K^T
\tag{27}
$$

Thus, the corrected covariance matrix is calculated as:

$$
\begin{aligned}
\hat{P}_t &= \bar{P}_t - 2 \bar{P}_t H^T K^T + K S K^T \\
&= \bar{P}_t - 2 \bar{P}_t H^T K^T + P H^T S^{-1} S K^T \\
&= \bar{P}_t - \bar{P}_t H^T K^T \\
&= (I - K H) \bar{P}_t
\end{aligned}
\tag{28}
$$

## Summary of EKF

The calculation of the Extended Kalman Filter (EKF) is now fully derived. In summary, the EKF calculation involves using the following six equations.

**Prediction Step**

* Equation (8): Predicting the state: \( \bar{x}_t = f(\bar{x}_{t-1}, \bar{u}) \)

* Equation (14): Predicting the covariance: \( \bar{P}_t = F_x P_{t-1} F_x^T + F_u Q F_u^T \)

**Correction Step**

* Equation (18): Observation residual: \( y = z - h(\bar{x}_t) \)

* Equation (22): Kalman Gain: \( K = P H^T (H \bar{P}_t H^T + R)^{-1} \)

* Equation (23): Correcting the state: \( \hat{x}_t = \bar{x}_t + K y \)

* Equation (28): Correcting the covariance: \( \hat{P}_t = (I - K H) \bar{P}_t \)
