import sympy
from sympy import diff, Matrix, Array,symbols

a,b,f ,x1,x2 = symbols('a,b,f ,x1,x2' )
def rho(x):
        aux =  x + 1.0
        return sympy.log(aux)

r = a*x1+x2-b

f = r*r/2

ro = rho(f) 

dro = Matrix([diff(ro,x1),diff(ro,x2)])
print(dro)

d2ro = Matrix([[diff(ro,x1,x1),diff(ro,x2,x1)],[diff(ro,x1,x2),diff(ro,x2,x2)]])

print(d2ro)
