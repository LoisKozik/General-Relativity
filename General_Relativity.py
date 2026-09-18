import numpy as np
import sympy as smp
from sympy import *
from IPython.display import display, Latex
import matplotlib.pyplot as plt
#import plotly.graph_objs as go

plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor":  '#373e4b',
    "axes.facecolor": '#373e4b',
})

G, M, t, r, theta, phi, k, c = smp.symbols('G M t r θ Φ k c', nonzero=True, real=True, positive=True)
a = smp.Function('a')
a_dot = smp.Function('ȧ')
a_dot = smp.Function('ȧ')
a_Dot = smp.Function('ä')

Gamma = IndexedBase("Gamma")
Riemann = IndexedBase("R")

Var = {
    t : 0,
    r : 1,
    theta : 2,
    phi : 3
}

Rav = {
    0 : t,
    1 : r,
    2 : theta,
    3 : phi
}

c_constant = 299792458  #m/s
G_constant = 6.67408*1E-11 #m^3/kg/s^2
H0_constant = 73,3 #km/s/Mpc
#Cosmological_constant = 
k_constant = 8*np.pi*G_constant/c_constant**4 #m/J

def R_s(m):
    return 2*G*m/c_constant**2

def Subs(expr):
    return smp.simplify(expr).subs([ (smp.diff(a(t),t,2), a_Dot(t)) , (smp.diff(a(t),t), a_dot(t)) ])

#########################################################################################################################################
#                                                           METRICS                                                                     #
#########################################################################################################################################

def Metric(dimensions):
    return np.zeros(dimensions, dtype='object')

def Display(Metric):
    return Matrix(Metric)

# Minkowkis Metric, Convention (-+++)
Mink_Metric = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
Mink = smp.Matrix(Mink_Metric)

# Swcharzschil Metric, Convention (-+++)
S_Metric = np.zeros((4,4))
S = smp.Matrix(S_Metric)

S[0,0] = -(1-2*G*M/r)
S[1,1] = 1/(1-2*G*M/r)
S[2,2] = r**2
S[3,3] = r**2*smp.sin(theta)**2

S = smp.nsimplify(S)


# FLRW Metric, Convention (+---)
FLRW_Metric = np.zeros((4,4))
FLRW = smp.Matrix(FLRW_Metric)

FLRW[0,0] = c**2
FLRW[1,1] = -a(t)**2/(1-k*r**2)
FLRW[2,2] = -a(t)**2*r**2
FLRW[3,3] = -a(t)**2*r**2*smp.sin(theta)**2

FLRW = smp.nsimplify(FLRW)

#########################################################################################################################################
#                                                           COMPUTE                                                                     #
#########################################################################################################################################

# Compute Christoffel Symbols

def Christoffel_Symbols(Metric):
    inv_Metric = Metric.inv()
    Gamma = np.zeros((4,4,4), dtype=object)
    for k in range(4):
        for i in range(4):
            for j in range(4):
                S = 0
                for l in range(4):
                    partial1 = smp.diff(Metric[i,l], Rav[j])
                    partial2 = smp.diff(Metric[l,j], Rav[i])
                    partial3 = smp.diff(Metric[j,i], Rav[l])
                    G = 1/2 * inv_Metric[k,l] * (partial1 + partial2 - partial3)
                    S += G
                Gamma[k,i,j] = smp.nsimplify(S)
    return Gamma

# Compute the Riemann Curvature Tensor

def Riemann_Tensor(Metric, cs=None):
    R = np.zeros((4,4,4,4), dtype=object)
    if cs is not None:
        Gamma = cs
    else:
        Gamma = Christoffel_Symbols(Metric)
    
    for i in range(4):
        for j in range(4):
            for l in range(4):
                for m in range(4):
                    s = 0
                    partial1 = smp.diff(Gamma[i,j,m],Rav[l])
                    partial2 = smp.diff(Gamma[i,j,l], Rav[m])
                    for k in range(4):
                        s += smp.nsimplify(Gamma[i,l,k]*Gamma[k,j,m] - Gamma[i,m,k]*Gamma[k,j,l])
                    R[i,j,l,m] = smp.simplify(partial1 - partial2 + s)
    return R

# Compute the Ricci Tensor

def Ricci_Tensor(Metric, R=None):
    Ric = np.zeros((4,4), dtype=object)
    if R is not None:
        R_tensor = R
    else:
        R_tensor = Riemann_Tensor(Metric)
        
    for i in range(4):
        for j in range(4):
            s = 0
            for k in range(4):
                s += R_tensor[k,i,k,j]
            Ric[i,j] = smp.simplify(s)
    return Ric

# Compute the Ricci Scalar

def Ricci_Scalar(Metric, R=None):
    if R is not None:
        R_tensor = R
    else:
        R_tensor = Riemann_Tensor(Metric)
    Ric = Ricci_Tensor(Metric,R_tensor)
    
    Scal = 0
    for i in range(4):
        for j in range(4):
            Scal += (Metric.inv())[i,j] * Ric[i,j]
    return Scal


#########################################################################################################################################
#                                                            DIPLAY                                                                     #
#########################################################################################################################################

def Display_Christoffel_Symbols(Metric, cs=None):
    if cs is not None:
        G = cs
    else:
        G = Christoffel_Symbols(Metric)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                g = smp.nsimplify(G[i,j,k])
                if g != 0:
                    #print(g)
                    display(Latex('$' + latex(Gamma[Rav[i],Rav[j]]**Rav[k]) + ' = ' + latex(Subs(g)) + '$'))


def Display_Riemann_Tensor(Metric, R=None):
    if R is not None:
        R_tensor = R
    else:
        R_tensor = Riemann_Tensor(Metric)
    
    for i in range(4):
        for j in range(4):
            for l in range(4):
                for m in range(4):
                    r = smp.simplify(R_tensor[i,j,l,m])
                    if r != 0 :
                        display(Latex('$' + latex(Riemann[Rav[j],Rav[l],Rav[m]]**Rav[i]) + ' = ' + latex(Subs(r)) + '$'))


def Display_Ricci_Tensor(Metric, R=None):
    if R is not None:
        R_tensor = R
    else:
        R_tensor = Riemann_Tensor(Metric)
    Ric = Ricci_Tensor(Metric,R_tensor)
    
    for i in range(4):
        for j in range(4):
            ric = smp.simplify(Ric[i,j])
            if ric != 0:
                display(Latex('$' + latex(Riemann[Rav[i],Rav[j]]) + ' = ' + latex(Subs(ric)) + '$'))
 

def Display_Ricci_Scalar(Metric, R=None):
    if R is not None:
        R_tensor = R
    else:
        R_tensor = Riemann_Tensor(Metric)
    Scal = Ricci_Scalar(Metric,R_tensor)
    display(Latex('$' + latex(Riemann) + ' = ' + latex(Subs(Scal)) + '$'))

#########################################################################################################################################
#                                                          Trajectory                                                                   #
#########################################################################################################################################

def f(t,u,G,Mass,L):
    return 3*G**2*Mass**2/L**2*u**2-u+1

# Dispaly attractor
def sphere(radius):
    theta = np.linspace(0,2*np.pi)
    phi = np.linspace(0,2*np.pi)

    theta,phi = np.meshgrid(theta,phi)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return x,y,z

def trajectory(Mass,N,L,E,u0,uh0,theta0,dtheta):
    theta = theta0
    u = u0
    uh = uh0
    r = L**2/(G_constant*Mass*u)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    t = 0

    X = []
    Y = []
    T = []

    for i in range(1,N):
        u_o = u
        uh_o = uh
        h = dtheta

        # Heun's Methode
        uh_hat = uh_o + h*f(theta,u_o,G_constant,Mass,L)
        u = u_o + 0.5*h*(uh_o + uh_hat)

        u_hat = u_o + h*uh_o
        uh = uh_o + 0.5*h*(f(theta, u_o, G_constant, Mass, L) + f(theta, u_hat, G_constant, Mass, L))

        r = L**2/(G_constant*Mass*u)

        d_tau = r**2 * dtheta / L
        dt = E * d_tau / (1.0 - 2 * G_constant * Mass / r)
        t = t + dt

        theta = theta + dtheta
        x = r*np.cos(theta)
        y = r*np.sin(theta)

        X.append(x)
        Y.append(y)
        T.append(t)

    return X,Y,T
    

def trajectory_2d(Mass,N,L,E,u0=0.5,uh0=0,theta0=0,dtheta=0.01):
    X,Y,T = trajectory(Mass,N,L,E,u0,uh0,theta0,dtheta)
    
    plt.plot(X,Y, label='Trajectory')
    plt.scatter(0,0, c='r', label='Attractor')
    plt.legend(loc="upper right")
    plt.show()

def trajectory_3d(Mass,radius,N,L,E,u0=0.5,uh0=0,theta0=0,dtheta=0.01):
    X,Y,T = trajectory(Mass,N,L,E,u0,uh0,theta0,dtheta)

    a,b,c = sphere(radius)

    Scale = 1/5e15

    x = np.array(X) * Scale
    y = np.array(Y) * Scale
    z = np.zeros(len(X)-4)

    N = max(max(x),max(y))

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y,z=z, mode='lines', name='Trajectory'), go.Scatter3d(x=a.flatten(), y=b.flatten(),z=c.flatten(), mode='lines', name='Attractor')])

    fig.update_layout(
            title = dict(text="TITLE"), 
            template= "plotly_dark", 
            margin=dict(l=0, r=0, b=0, t=0), 
            scene=dict(camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)), #the default values are 1.25, 1.25, 1.25
            xaxis=dict(range=[-N,N]),
            yaxis=dict(range=[-N,N]),
            zaxis=dict(range=[-N,N]),
            aspectmode='cube',
                    ))

    fig.layout.scene.camera.projection.type = "orthographic"
    fig.show()