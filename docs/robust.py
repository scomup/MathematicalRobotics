import sympy
from sympy import diff, Matrix, Array, symbols

a, b, f , x, x1, x2, d, w = symbols('a, b, f , x, x1, x2, d, w')
# def kernel(x):
#        return 1-sympy.exp(-w*x)
def kernel(x):
        return sympy.sqrt(x)

f = x*x/2

rho = kernel(f) 

d = diff(rho, x)
sqrt(2)/(2)
print(d)
d2 = diff(rho, x, x)


print(d2)
