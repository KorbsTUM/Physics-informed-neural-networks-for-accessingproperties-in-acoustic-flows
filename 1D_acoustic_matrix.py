import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.linalg as la

L = 2 # Length of domain in x-direction
dimX = 50 # Number of steps in x-direction for discretization

T = 1 # Simulated total time
dimT = 2000 # Number of steps of time discretization

c = 343 # Definition of constant speed of sound 

Solver = 'Theta' # Choice of the temporal solver: Explicit, Implicit, Theta

Theta = 0.7 # Theta-Parameter for the Theta-Solver

x,dx = np.linspace(0,L,dimX,retstep=True) # Discretization of space

t,dt = np.linspace(0,T,dimT,retstep=True) # Discretization of time

C = c*dt/dx # Courant number

P = np.zeros((dimX,dimT)) # Initialize pressure tensor

def I(x,L): # Define the Starting condition for t = 0
    p0 = np.sin((np.pi/L)*x)
    return p0

P[:,0] = I(x,L) # Include starting condition into pressure tensor
P[0,0] = P[-1,0] = 0

A = np.zeros((dimX,dimX))

A[0,0] = A[-1,-1] = 0

for i in range(1,dimX-1):
    A[i,i] = -2
    A[i,i-1] = A[i,i+1] = 1

if Solver == 'Explicit':
    P[:,1] = P[:,0] + 0.5*C**2*A@P[:,0]
    P[0,1] = P[-1,1] = 0
    for n in range(1,dimT-2):
        P[:,n+1] = C**2*A@P[:,n] + 2*P[:,n] - P[:,n-1]
        P[0,n+1] = P[-1,n+1] = 0
        
elif Solver == 'Implicit':
    Astar = np.eye(dimX) - 0.5*C**2*A
    P[:,1] = la.solve(Astar,P[:,0])
    P[0,1] = P[-1,1] = 0
    
    Astar2 = np.eye(dimX) - C**2*A
    
    for n in range(1,dimT-2):
        bstar2 = 2*P[:,n] - P[:,n-1]
        P[:,n+1] = la.solve(Astar2,bstar2)
        P[0,n+1] = P[-1,n+1] = 0
        
elif Solver == 'Theta':
    Astar = np.eye(dimX) - 0.5*C**2*Theta*A
    bstar = P[:,0] + 0.5*(1-Theta)*C**2*A@P[:,0]
    P[:,1] = la.solve(Astar,bstar)
    P[0,1] = P[-1,1] = 0
    
    Astar2 = np.eye(dimX) - C**2*Theta*A
    
    for n in range(1,dimT-2):
        bstar2 = (1-Theta)*C**2*A@P[:,n] + 2*P[:,n] - P[:,n-1]
        P[:,n+1] = la.solve(Astar2,bstar2)
        P[0,n+1] = P[-1,n+1] = 0
        

Figure = plt.figure()

lines_plotted = plt.plot([])

line_plotted = lines_plotted[0]

plt.xlim(-0.1,1.1*L)

plt.ylim(-1.1,1.1)

def AnimationFunction(frame):
    y = P[:,frame]
    line_plotted.set_data((x,y))

anim = FuncAnimation(Figure, AnimationFunction, frames=200, interval=25)

plt.show()

anim.save('1D_acosutic_matrix.gif', dpi=80, writer='imagemagick')