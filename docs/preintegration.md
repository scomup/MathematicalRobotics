## Imu preintegration.  

### PIM (preintegration measurement)

### Correct PIM

Correct PIM vector with given velocity and gravity.   

$$
\theta^\prime = \theta \\
p^\prime = p + R_{nb}^{-1} v_n \Delta{t} + R_{nb}^{-1} g \frac{\Delta{t}^2}{2} \\
v^\prime = v + R_{nb}^{-1} g \Delta{t} \\
$$
$n$ navigation frame, $b$ body frame


### The jocabian for Corrected_PIM

#### Derivative of P
$$
\frac{\partial{p^\prime}}{\partial r} = \widehat{R_{nb}^{-1}v_n} \Delta{t} + \widehat{R_{nb}^{-1}g} \frac{\Delta{t}^2}{2}
$$

$$
\frac{\partial{p^\prime}}{\partial v} = R_{nb}^{-1} \Delta{t} 
$$

$$
\frac{\partial{v^\prime}}{\partial r} = \widehat{R_{nb}^{-1}g} \Delta{t}
$$

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
