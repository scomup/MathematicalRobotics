import sympy
from sympy import diff, Matrix, Array,symbols


a1,a2,a3 = symbols('a1,a2,a3')
b1,b2,b3 = symbols('b1,b2,b3')
a = Matrix([a1,a2,a3,1])
b = Matrix([b1,b2,b3,1])
I=Matrix.eye(4)
t11,t12,t13,t14,t21,t22,t23,t24,t31,t32,t33,t34 = symbols('t11,t12,t13,t14,t21,t22,t23,t24,t31,t32,t33,t34')
T0 = Matrix([[t11,t12,t13,t14],
                   [t21,t22,t23,t24],
                   [t31,t32,t33,t34],[0,0,0,1]])
R0 = Matrix([[t11,t12,t13],
                   [t21,t22,t23],
                   [t31,t32,t33]])
# the parameters of SE3
x1,x2,x3,x4,x5,x6 = symbols('x1,x2,x3,x4,x5,x6')
x = Matrix([x1,x2,x3,x4,x5,x6])

def skew(x):
    return Matrix([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])

def hat(x):
    return Matrix([[0,-x[5],x[4],x[0]],[x[5],0,-x[3],x[1]],[-x[4],x[3],0,x[2]],[0,0,0,0]])

expx = I + hat(x)

# guass_newton_method.md (7)
r = T0*expx*a-b
r = Matrix(r[0:3])

dr_real = Matrix([[diff(r[0],x1),diff(r[0],x2),diff(r[0],x3),diff(r[0],x4),diff(r[0],x5),diff(r[0],x6)],
                  [diff(r[1],x1),diff(r[1],x2),diff(r[1],x3),diff(r[1],x4),diff(r[1],x5),diff(r[1],x6)],
                  [diff(r[2],x1),diff(r[2],x2),diff(r[2],x3),diff(r[2],x4),diff(r[2],x5),diff(r[2],x6)]])

# guass_newton_method.md (12)
dr = Matrix.hstack(R0,-R0*skew(a))

d2r1 = Matrix([[diff(r[0],x1,x1),diff(r[0],x1,x2),diff(r[0],x1,x3),diff(r[0],x1,x4),diff(r[0],x1,x5),diff(r[0],x1,x6)],
                     [diff(r[0],x2,x1),diff(r[0],x2,x2),diff(r[0],x2,x3),diff(r[0],x2,x4),diff(r[0],x2,x5),diff(r[0],x2,x6)],
                     [diff(r[0],x3,x1),diff(r[0],x3,x2),diff(r[0],x3,x3),diff(r[0],x3,x4),diff(r[0],x3,x5),diff(r[0],x3,x6)],
                     [diff(r[0],x4,x1),diff(r[0],x4,x2),diff(r[0],x4,x3),diff(r[0],x4,x4),diff(r[0],x4,x5),diff(r[0],x4,x6)],
                     [diff(r[0],x5,x1),diff(r[0],x5,x2),diff(r[0],x5,x3),diff(r[0],x5,x4),diff(r[0],x5,x5),diff(r[0],x5,x6)],
                     [diff(r[0],x6,x1),diff(r[0],x6,x2),diff(r[0],x6,x3),diff(r[0],x6,x4),diff(r[0],x6,x5),diff(r[0],x6,x6)]])
d2r2 = Matrix([[diff(r[1],x1,x1),diff(r[1],x1,x2),diff(r[1],x1,x3),diff(r[1],x1,x4),diff(r[1],x1,x5),diff(r[1],x1,x6)],
                     [diff(r[1],x2,x1),diff(r[1],x2,x2),diff(r[1],x2,x3),diff(r[1],x2,x4),diff(r[1],x2,x5),diff(r[1],x2,x6)],
                     [diff(r[1],x3,x1),diff(r[1],x3,x2),diff(r[1],x3,x3),diff(r[1],x3,x4),diff(r[1],x3,x5),diff(r[1],x3,x6)],
                     [diff(r[1],x4,x1),diff(r[1],x4,x2),diff(r[1],x4,x3),diff(r[1],x4,x4),diff(r[1],x4,x5),diff(r[1],x4,x6)],
                     [diff(r[1],x5,x1),diff(r[1],x5,x2),diff(r[1],x5,x3),diff(r[1],x5,x4),diff(r[1],x5,x5),diff(r[1],x5,x6)],
                     [diff(r[1],x6,x1),diff(r[1],x6,x2),diff(r[1],x6,x3),diff(r[1],x6,x4),diff(r[1],x6,x5),diff(r[1],x6,x6)]])
d2r3 = Matrix([[diff(r[2],x1,x1),diff(r[2],x1,x2),diff(r[2],x1,x3),diff(r[2],x1,x4),diff(r[2],x1,x5),diff(r[2],x1,x6)],
                     [diff(r[2],x2,x1),diff(r[2],x2,x2),diff(r[2],x2,x3),diff(r[2],x2,x4),diff(r[2],x2,x5),diff(r[2],x2,x6)],
                     [diff(r[2],x3,x1),diff(r[2],x3,x2),diff(r[2],x3,x3),diff(r[2],x3,x4),diff(r[2],x3,x5),diff(r[2],x3,x6)],
                     [diff(r[2],x4,x1),diff(r[2],x4,x2),diff(r[2],x4,x3),diff(r[2],x4,x4),diff(r[2],x4,x5),diff(r[2],x4,x6)],
                     [diff(r[2],x5,x1),diff(r[2],x5,x2),diff(r[2],x5,x3),diff(r[2],x5,x4),diff(r[2],x5,x5),diff(r[2],x5,x6)],
                     [diff(r[2],x6,x1),diff(r[2],x6,x2),diff(r[2],x6,x3),diff(r[2],x6,x4),diff(r[2],x6,x5),diff(r[2],x6,x6)]])

d2r = Array([d2r1.tolist(),d2r2.tolist(),d2r3.tolist()])

s11,s12,s13,s22,s33,s23 = symbols('s11,s12,s13,s22,s33,s23')

# Sigma is a diagonal matrix
Sigma = Matrix([[s11,s12,s13],
                [s12,s22,s23],
                [s13,s23,s33]])

#newton_method.md (2)
f=r.T*Sigma*r/2
df_real = Matrix([diff(f[0],x1),diff(f[0],x2),diff(f[0],x3),diff(f[0],x4),diff(f[0],x5),diff(f[0],x6)]).T

#newton_method.md (6)
df = r.T*Sigma*dr

if(dr.expand()==dr_real.expand()):
    print("The dr is correct!")
else:
    print("The dr is wrong!")

if(df.expand()==df_real.expand()):
    print("The df is correct!")
else:
    print("The df is wrong!")

d2f_real = Matrix([[diff(f[0],x1,x1),diff(f[0],x1,x2),diff(f[0],x1,x3),diff(f[0],x1,x4),diff(f[0],x1,x5),diff(f[0],x1,x6)],
                   [diff(f[0],x2,x1),diff(f[0],x2,x2),diff(f[0],x2,x3),diff(f[0],x2,x4),diff(f[0],x2,x5),diff(f[0],x2,x6)],
                   [diff(f[0],x3,x1),diff(f[0],x3,x2),diff(f[0],x3,x3),diff(f[0],x3,x4),diff(f[0],x3,x5),diff(f[0],x3,x6)],
                   [diff(f[0],x4,x1),diff(f[0],x4,x2),diff(f[0],x4,x3),diff(f[0],x4,x4),diff(f[0],x4,x5),diff(f[0],x4,x6)],
                   [diff(f[0],x5,x1),diff(f[0],x5,x2),diff(f[0],x5,x3),diff(f[0],x5,x4),diff(f[0],x5,x5),diff(f[0],x5,x6)],
                   [diff(f[0],x6,x1),diff(f[0],x6,x2),diff(f[0],x6,x3),diff(f[0],x6,x4),diff(f[0],x6,x5),diff(f[0],x6,x6)]])

d2f = dr.T * Sigma * dr 
print(d2f)

if(d2f.expand()==d2f_real.expand()):
    print("The d2f is correct!")
else:
    print("The d2f is wrong!")

