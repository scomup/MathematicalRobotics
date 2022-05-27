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
df_ = sympy.Matrix(sympy.diff(f,r).reshape(1,2))*dr

df =r.T * Sigma * dr

print(df_)
print(df)

ddf_ = sympy.Matrix(sympy.diff(df,r).reshape(1,2))*dr

print(ddf_)
"""
a1,a2,a3 = sympy.symbols('a1,a2,a3')
b1,b2,b3 = sympy.symbols('b1,b2,b3')
a = sympy.Matrix([a1,a2,a3,1])
b = sympy.Matrix([b1,b2,b3,1])
I=sympy.Matrix.eye(3)
r11,r12,r13,r14,r21,r22,r23,r24,r31,r32,r33,r34 = sympy.symbols('r11,r12,r13,r14,r21,r22,r23,r24,r31,r32,r33,r34')
T0 = sympy.Matrix([[r11,r12,r13,r14],[r21,r22,r23,r24],[r31,r32,r33,r34],[0,0,0,1]])
R0 = sympy.Matrix([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])

w1,w2,w3 = sympy.symbols('w1,w2,w3')
w = sympy.Matrix([w1,w2,w3])
t1,t2,t3 = sympy.symbols('t1,t2,t3')
t = sympy.Matrix([t1,t2,t3])

delta = sympy.Matrix([[1,-w3,w2,t1],[w3,1,-w1,t2],[-w2,w1,1,t3],[0,0,0,1]])

def skew(w):
    return sympy.Matrix([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])

def hat(w,tt):
    return sympy.Matrix([[0,-w[2],w[1],tt[1] ],
    [w[2],0,-w[0],t[2]],
    [-w[1],w[0],0,,t[2]],
    [0,0,0,1]])


res = T0*(I+hat(w,t))*a-b

df = sympy.Matrix([[diff(res[0],t1),diff(res[0],t2),diff(res[0],t3),diff(res[0],w1),diff(res[0],w2),diff(res[0],w3)],
                   [diff(res[1],t1),diff(res[1],t2),diff(res[1],t3),diff(res[1],w1),diff(res[1],w2),diff(res[1],w3)],
                   [diff(res[2],t1),diff(res[2],t2),diff(res[2],t3),diff(res[2],w1),diff(res[2],w2),diff(res[2],w3)]])

d2f1 = sympy.Matrix([[diff(res[0],t1,t1),diff(res[0],t1,t2),diff(res[0],t1,t3),diff(res[0],t1,w1),diff(res[0],t1,w2),diff(res[0],t1,w3)],
                     [diff(res[0],t2,t1),diff(res[0],t2,t2),diff(res[0],t2,t3),diff(res[0],t2,w1),diff(res[0],t2,w2),diff(res[0],t2,w3)],
                     [diff(res[0],t3,t1),diff(res[0],t3,t2),diff(res[0],t3,t3),diff(res[0],t3,w1),diff(res[0],t3,w2),diff(res[0],t3,w3)],
                     [diff(res[0],w1,t1),diff(res[0],w1,t2),diff(res[0],w1,t3),diff(res[0],w1,w1),diff(res[0],w1,w2),diff(res[0],w1,w3)],
                     [diff(res[0],w2,t1),diff(res[0],w2,t2),diff(res[0],w2,t3),diff(res[0],w2,w1),diff(res[0],w2,w2),diff(res[0],w2,w3)],
                     [diff(res[0],w3,t1),diff(res[0],w3,t2),diff(res[0],w3,t3),diff(res[0],w3,w1),diff(res[0],w3,w2),diff(res[0],w3,w3)]])
d2f2 = sympy.Matrix([[diff(res[1],t1,t1),diff(res[1],t1,t2),diff(res[1],t1,t3),diff(res[1],t1,w1),diff(res[1],t1,w2),diff(res[1],t1,w3)],
                     [diff(res[1],t2,t1),diff(res[1],t2,t2),diff(res[1],t2,t3),diff(res[1],t2,w1),diff(res[1],t2,w2),diff(res[1],t2,w3)],
                     [diff(res[1],t3,t1),diff(res[1],t3,t2),diff(res[1],t3,t3),diff(res[1],t3,w1),diff(res[1],t3,w2),diff(res[1],t3,w3)],
                     [diff(res[1],w1,t1),diff(res[1],w1,t2),diff(res[1],w1,t3),diff(res[1],w1,w1),diff(res[1],w1,w2),diff(res[1],w1,w3)],
                     [diff(res[1],w2,t1),diff(res[1],w2,t2),diff(res[1],w2,t3),diff(res[1],w2,w1),diff(res[1],w2,w2),diff(res[1],w2,w3)],
                     [diff(res[1],w3,t1),diff(res[1],w3,t2),diff(res[1],w3,t3),diff(res[1],w3,w1),diff(res[1],w3,w2),diff(res[1],w3,w3)]])
d2f3 = sympy.Matrix([[diff(res[2],t1,t1),diff(res[2],t1,t2),diff(res[2],t1,t3),diff(res[2],t1,w1),diff(res[2],t1,w2),diff(res[2],t1,w3)],
                     [diff(res[2],t2,t1),diff(res[2],t2,t2),diff(res[2],t2,t3),diff(res[2],t2,w1),diff(res[2],t2,w2),diff(res[2],t2,w3)],
                     [diff(res[2],t3,t1),diff(res[2],t3,t2),diff(res[2],t3,t3),diff(res[2],t3,w1),diff(res[2],t3,w2),diff(res[2],t3,w3)],
                     [diff(res[2],w1,t1),diff(res[2],w1,t2),diff(res[2],w1,t3),diff(res[2],w1,w1),diff(res[2],w1,w2),diff(res[2],w1,w3)],
                     [diff(res[2],w2,t1),diff(res[2],w2,t2),diff(res[2],w2,t3),diff(res[2],w2,w1),diff(res[2],w2,w2),diff(res[2],w2,w3)],
                     [diff(res[2],w3,t1),diff(res[2],w3,t2),diff(res[2],w3,t3),diff(res[2],w3,w1),diff(res[2],w3,w2),diff(res[2],w3,w3)]])


print(f)



