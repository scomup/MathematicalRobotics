import sympy
from sympy import diff

"""
r1,r2 = sympy.symbols('r1,r2')
r = sympy.Matrix([r1,r2])

dr1,dr2 = sympy.symbols('dr1,dr2')
dr = sympy.Matrix([dr1,dr2])


s1,s2,s3 = sympy.symbols('s1,s2,s3')
Sigma = sympy.Matrix([[s1,s3],[s3,s2]])

f = r.T * Sigma * r/2
df_ = sympy.Matrix(sympy.diff(f,r).rhape(1,2))*dr

df =r.T * Sigma * dr

print(df_)
print(df)

ddf_ = sympy.Matrix(sympy.diff(df,r).rhape(1,2))*dr

print(ddf_)
"""
a1,a2,a3 = sympy.symbols('a1,a2,a3')
b1,b2,b3 = sympy.symbols('b1,b2,b3')
a = sympy.Matrix([a1,a2,a3,1])
b = sympy.Matrix([b1,b2,b3,1])
I=sympy.Matrix.eye(4)
t11,t12,t13,t14,t21,t22,t23,t24,t31,t32,t33,t34 = sympy.symbols('t11,t12,t13,t14,t21,t22,t23,t24,t31,t32,t33,t34')
T0 = sympy.Matrix([[t11,t12,t13,t14],
                   [t21,t22,t23,t24],
                   [t31,t32,t33,t34],[0,0,0,1]])
R0 = sympy.Matrix([[t11,t12,t13],
                   [t21,t22,t23],
                   [t31,t32,t33]])

x1,x2,x3,x4,x5,x6 = sympy.symbols('x1,x2,x3,x4,x5,x6')
x = sympy.Matrix([x1,x2,x3,x4,x5,x6])

def skew(x):
    return sympy.Matrix([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])

def hat(x):
    return sympy.Matrix([[0,-x[5],x[4],x[0]],[x[5],0,-x[3],x[1]],[-x[4],x[3],0,x[2]],[0,0,0,0]])

exp2 = I + hat(x)
exp3 = I + hat(x) + hat(x) * hat(x)/2
r = T0*exp2*a-b

# [R0,-R0*skew(a)]
dr = sympy.Matrix([[diff(r[0],x1),diff(r[0],x2),diff(r[0],x3),diff(r[0],x4),diff(r[0],x5),diff(r[0],x6)],
                   [diff(r[1],x1),diff(r[1],x2),diff(r[1],x3),diff(r[1],x4),diff(r[1],x5),diff(r[1],x6)],
                   [diff(r[2],x1),diff(r[2],x2),diff(r[2],x3),diff(r[2],x4),diff(r[2],x5),diff(r[2],x6)]])

d2r1 = sympy.Matrix([[diff(r[0],x1,x1),diff(r[0],x1,x2),diff(r[0],x1,x3),diff(r[0],x1,x4),diff(r[0],x1,x5),diff(r[0],x1,x6)],
                     [diff(r[0],x2,x1),diff(r[0],x2,x2),diff(r[0],x2,x3),diff(r[0],x2,x4),diff(r[0],x2,x5),diff(r[0],x2,x6)],
                     [diff(r[0],x3,x1),diff(r[0],x3,x2),diff(r[0],x3,x3),diff(r[0],x3,x4),diff(r[0],x3,x5),diff(r[0],x3,x6)],
                     [diff(r[0],x4,x1),diff(r[0],x4,x2),diff(r[0],x4,x3),diff(r[0],x4,x4),diff(r[0],x4,x5),diff(r[0],x4,x6)],
                     [diff(r[0],x5,x1),diff(r[0],x5,x2),diff(r[0],x5,x3),diff(r[0],x5,x4),diff(r[0],x5,x5),diff(r[0],x5,x6)],
                     [diff(r[0],x6,x1),diff(r[0],x6,x2),diff(r[0],x6,x3),diff(r[0],x6,x4),diff(r[0],x6,x5),diff(r[0],x6,x6)]])
d2r2 = sympy.Matrix([[diff(r[1],x1,x1),diff(r[1],x1,x2),diff(r[1],x1,x3),diff(r[1],x1,x4),diff(r[1],x1,x5),diff(r[1],x1,x6)],
                     [diff(r[1],x2,x1),diff(r[1],x2,x2),diff(r[1],x2,x3),diff(r[1],x2,x4),diff(r[1],x2,x5),diff(r[1],x2,x6)],
                     [diff(r[1],x3,x1),diff(r[1],x3,x2),diff(r[1],x3,x3),diff(r[1],x3,x4),diff(r[1],x3,x5),diff(r[1],x3,x6)],
                     [diff(r[1],x4,x1),diff(r[1],x4,x2),diff(r[1],x4,x3),diff(r[1],x4,x4),diff(r[1],x4,x5),diff(r[1],x4,x6)],
                     [diff(r[1],x5,x1),diff(r[1],x5,x2),diff(r[1],x5,x3),diff(r[1],x5,x4),diff(r[1],x5,x5),diff(r[1],x5,x6)],
                     [diff(r[1],x6,x1),diff(r[1],x6,x2),diff(r[1],x6,x3),diff(r[1],x6,x4),diff(r[1],x6,x5),diff(r[1],x6,x6)]])
d2r3 = sympy.Matrix([[diff(r[2],x1,x1),diff(r[2],x1,x2),diff(r[2],x1,x3),diff(r[2],x1,x4),diff(r[2],x1,x5),diff(r[2],x1,x6)],
                     [diff(r[2],x2,x1),diff(r[2],x2,x2),diff(r[2],x2,x3),diff(r[2],x2,x4),diff(r[2],x2,x5),diff(r[2],x2,x6)],
                     [diff(r[2],x3,x1),diff(r[2],x3,x2),diff(r[2],x3,x3),diff(r[2],x3,x4),diff(r[2],x3,x5),diff(r[2],x3,x6)],
                     [diff(r[2],x4,x1),diff(r[2],x4,x2),diff(r[2],x4,x3),diff(r[2],x4,x4),diff(r[2],x4,x5),diff(r[2],x4,x6)],
                     [diff(r[2],x5,x1),diff(r[2],x5,x2),diff(r[2],x5,x3),diff(r[2],x5,x4),diff(r[2],x5,x5),diff(r[2],x5,x6)],
                     [diff(r[2],x6,x1),diff(r[2],x6,x2),diff(r[2],x6,x3),diff(r[2],x6,x4),diff(r[2],x6,x5),diff(r[2],x6,x6)]])

d2r = sympy.Array([d2r1.tolist(),d2r2.tolist(),d2r3.tolist()])
#r = sympy.Matrix(r[0:3]) 

#H = sympy.Matrix.zeros(6,6)

#for i in range(6):
#    for j in range(6):
#        H[i,j] = r.T * sympy.Matrix(d2r[:,i,j])
#
#print(H)




