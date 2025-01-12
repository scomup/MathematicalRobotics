# Bayesian Filter
A Bayesian filter represents the uncertain state of a system (for example, a robot’s position or velocity) as a probability distribution and estimates it sequentially. First, it predicts the next state based on the past state distribution and control input, and then corrects that prediction using new observation data.

**Prediction Step**  
In the prediction step of a Bayesian filter, the “past state” and the associated “control input or operation” are used to probabilistically predict the “next state.” This process is expressed by:

$$ \bar{p}(x_t) = \int p(x_t \mid u, x_{t-1}) p(x_{t-1}) \, dx_{t-1} \tag{1} $$

- **Motion model** $p(x_t \mid u, x_{t-1})$: This represents the probability of arriving at state $x_t$ if the old state was $x_{t-1}$ and you applied control input $u$. It’s designed based on the physical properties and principles of the system.
- **Past state distribution** $p(x_{t-1})$: The probability distribution that the old state was actually $x_{t-1}$. It is obtained from the previous step’s filtering update.

The integral in (1) can be seen as an application of the law of total probability, meaning you consider all possible $x_{t-1}$ to find $\bar{p}(x_t)$.

Also, the prediction step of a Bayesian filter relies on the assumption that knowing the current state $x_{t-1}$ and control $u$ is sufficient to predict the next state $x_t$, without needing earlier states ($x_{t-2}, x_{t-3}, \dots$). This is the Markov property.

**Correction Step**  
In the correction step of a Bayesian filter, the “predicted state distribution” is updated with the “observed data” to compute a more reliable probability distribution for the current state. This process is expressed by:

$$ p(x_t \mid z_t) = \eta \, p(z_t \mid x_t) \, \bar{p}(x_t) \tag{2} $$

Breaking this down:

- **Predicted distribution** $\bar{p}(x_t)$: The prior distribution over $x_t$ from the prediction step.
- **Observation model** $p(z_t \mid x_t)$: The probability of observations $z_t$ given state $x_t$. This incorporates sensor data.
- **Normalization constant** $\eta$: A factor ensuring the total probability integrates to 1.

This follows Bayes' theorem, using the new observation $z_t$ to update the prior distribution $\bar{p}(x_t)$ and yield the posterior $p(x_t \mid z_t)$. By integrating the observation data with the prior belief, the uncertainty around the current state is reduced, yielding a more accurate estimation.

After correction, $p(x_t \mid z_t)$ is passed forward as the prior $\bar{p}(x_{t+1})$ in the next prediction step. Repeating the prediction and correction steps continuously updates the system’s state in real time.

**Features of Each Filtering Technique**  
In a Bayesian filter, distributions $p(x)$, $p(x_t \mid x_{t-1}, u)$, and $p(z_t \mid x_t)$ are all probabilities. You must model them accurately for proper computation.

Differences in how these distributions are modeled lead to two main filtering approaches:

- **Extended Kalman Filter (EKF)**:
  - Assumes state uncertainty follows a Gaussian (normal) distribution.
  - Suitable for linear or slightly nonlinear systems, with efficient computation and low processing overhead.
  - Cannot handle cases where the Gaussian assumption significantly fails.

- **Particle Filter (PF)**:
  - Does not assume a Gaussian form; instead, uses many sampled points (“particles”) to represent the uncertainty.
  - Flexible and can handle nonlinear or complex distributions.
  - Computationally more expensive, since processing increases with the number of samples.

### Extended Kalman Filter (EKF)
We now dive into the details of the Extended Kalman Filter. In many systems, using a Gaussian to represent uncertainty works well, and the lower computation overhead is crucial in fields like robotics and control systems.

We'll examine EKF in detail, with visuals and equations to build up a thorough understanding.

#### Prediction Step  
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

$$ Q = \mathbb{E}((u-\bar{u})(u-\bar{u})^T) = \mathbb{E}(\Delta u, \Delta u^T) \tag{7} $$

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

$$ \bar{P}_t = F_x P_{t-1} F_x^T + F_u Q F_u^T \tag{13} $$

### Correction Step:

In the Extended Kalman Filter (EKF) correction step, the predicted state $\bar{x}_t$ is adjusted using the observation data to estimate a more accurate state $\hat{x}_t$. Here, the hat symbol ($\hat{\quad}$) denotes the corrected value.

To obtain the corrected state $\hat{x}_t$, the update $\Delta{x}_t$ is first defined based on the predicted state $\bar{x}_t$ from the prediction step:

$$
\hat{x}_t = \bar{x}_t + \Delta{x}_t
\tag{15}
$$

The correction step calculation requires two main components:

1. The formula for the update $\Delta{x}_t$ to compute $\hat{x}_t$.
2. The formula for the corrected covariance matrix $\hat{P}_t$ (which represents the uncertainty in $\hat{x}_t$).

We will derive these two formulas from theory in this section.

#### Step 1: Define the Observation Model

The sensor observation is based on the true state $x_t$, and the observation equation $h$ provides the observation $z$. This observation includes some error $v$, represented by a covariance matrix $R$.

$$
p(z|x_t) = \mathcal{N}(h(x_t), R)
\tag{16}
$$

$$
z = h(x_t) + v
\tag{17}
$$

From the predicted state $\bar{x}_t$, the corresponding observation can be predicted. The difference between the actual observation $z$ and the predicted observation $h(\bar{x}_t)$ is referred to as the observation residual $y$:

$$
y \equiv z - h(\bar{x}_t)
\tag{18}
$$

#### Step 2: Calculate the Covariance of the Observation Residual $y$

The covariance matrix of the residual $y$ is defined as $S$, and it is derived as follows:

$$
\begin{aligned}
S &= E(yy^T) \\\\
&= E((H\Delta{x}_t + v)(H\Delta{x}_t + v)^T) \\\\
&= E(H\Delta{x}_t\Delta{x}_t^TH^T) + E(vv^T) \\\\
&= H\bar{P}_tH^T + R
\end{aligned}
\tag{19}
$$

The calculation above uses the linear approximation of $h$ (i.e., $h(x_t) \approx h(\bar{x}_t) + H \Delta{x}_t$), where $H$ is the Jacobian matrix of the observation model.

#### Step 3: Minimize the Overall Error

The ideal $\Delta{x}_t$ is obtained by minimizing the discrepancy between the observation residual $y$ and the prediction error $\Delta{x}_t$. Therefore, an objective function can be formulated to minimize the total error with respect to $\Delta{x}_t$ as shown in Equation (20):

$$
\begin{aligned}
\text{argmin} \quad F(\bar{x}+\Delta{x}_t) &= \lVert y \rVert ^2_S + \lVert \Delta{x}_t \rVert^2_P \\\\ 
&= \lVert z - h(\bar{x}_t) \rVert^2_S + \lVert x - \bar{x}_t \rVert^2_P \\\\ 
&= ( z - h(\bar{x}_t))S^{-1}( z - h(\bar{x}_t))^T + ( x - \bar{x}_t )P^{-1}( x - \bar{x}_t )^T
\end{aligned}
\tag{20}
$$

By differentiating the objective function with respect to $\Delta{x}_t$ and setting it to zero, the optimal $\Delta{x}_t$ can be found:

$$
\begin{aligned}
0 &= 2 H^TS^{-1}(z - h(\bar{x}_t)) + 2 P^{-1}\Delta{x}_t \\\\
\Delta{x}_t &= PH^TS^{-1}(z - h(\bar{x}_t)) \\\\
\Delta{x}_t &= PH^TS^{-1}y
\end{aligned}
\tag{21}
$$

The term $PH^TS^{-1}$ is defined as the **Kalman Gain** $K$:

$$
K \equiv PH^TS^{-1}
\tag{22}
$$

By substituting Equation (21) and (22) into Equation (15), the corrected state $\hat{x}_t$ can be computed as:

$$
\hat{x}_t = \bar{x}_t + Ky
\tag{23}
$$

#### Step 4: Update the Covariance Matrix

Next, we evaluate the corrected covariance matrix $\hat{P}_t$. The corrected covariance is computed based on the mean square error between the true state $x_t$ and the corrected state $\hat{x}_t$:

$$
\hat{P}_t = E((x_t-\hat{x}_t)(x_t-\hat{x}_t)^T)
\tag{24}
$$

Expanding this equation:

$$
\begin{aligned}
\hat{P}_t &= E((x_t-\bar{x}_t-Ky)(x_t-\bar{x}_t-Ky)^T) \\\\
&= E\left[(x_t - \bar{x}_t)(x_t - \bar{x}_t)^T - (x_t - \bar{x}_t)(Ky)^T - (Ky)(x_t - \bar{x}_t)^T + (Ky)(Ky)^T\right]
\end{aligned}
\tag{25}
$$

- **First term**: $ E[(x_t - \bar{x}_t)(x_t - \bar{x}_t)^T] = \bar{P}_t $
- **Second and third terms**: 
$$
\begin{aligned}
&E[(x_t−\bar{x}_t)(h(x_t)+v−h(\bar{x}_t))^TK^T] \\\\
=&E[(x_t−\bar{x}_t)(h(\bar{x}_t) +H(x_t−\bar{x}_t) +v−h(\bar{x}_t))^TK^T] \\\\
=&E[(x_t−\bar{x}_t)(H(x_t−\bar{x}_t) +v)^TK^T] \\\\
=&E[(x_t - \bar{x}_t)(x_t - \bar{x}_t)^T H^TK^T] \\\\
=& \bar{P}_t H^TK^T \\\\
=& KH\bar{P}_t
\end{aligned}
\tag{26}
$$

- **Fourth term**:
$$
E((Ky)(Ky)^T) = KE(yy^T)K^T = KSK^T
\tag{27}
$$

Thus, the updated covariance matrix is calculated as:

$$
\begin{aligned}
\hat{P}_t &= \bar{P}_t - 2 \bar{P}_t H^TK^T + KSK^T \\\\
&= \bar{P}_t - 2 \bar{P}_t H^TK^T + PH^TS^{-1} SK^T \\\\
&= \bar{P}_t - \bar{P}_t H^TK^T \\\\
&= (I-K H)\bar{P}_t
\end{aligned}
\tag{28}
$$

