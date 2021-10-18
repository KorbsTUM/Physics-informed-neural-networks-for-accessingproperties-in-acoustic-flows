import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 2 # Length of domain in x-direction
dimX = 50 # Number of steps in x-direction for discretization

T = 0.02 # Simulated total time
dimT = 200 # Number of steps of time discretization

c = 343 # Definition of constant speed of sound 

x,dx = np.linspace(0,L,dimX,retstep=True) # Discretization of space

t,dt = np.linspace(0,T,dimT,retstep=True) # Discretization of time

C = c*dt/dx # Courant number

P = np.zeros((dimX,dimT)) # Initialize pressure tensor

def I(x,L): # Define the Starting condition for t = 0
    p0 = np.sin((np.pi/L)*x)
    return p0

P[:,0] = I(x,L) # Include starting condition into pressure tensor
P[0,0] = P[-1,0] = 0

for i in range(1,dimX-1):
    P[i,1] = P[i,0] - 0.5*C**2*(P[i+1,0] - 2*P[i,0] + P[i-1,0])
P[0,1] = P[-1,1] = 0

for n in range(2,dimT-1):
    for i in range(1,dimX-1):
        P[i,n] = 2*P[i,n-1] - P[i,n-2] + C**2*(P[i+1,n-1] - 2*P[i,n-1] \
                                               + P[i-1,n-1])
    P[0,n] = P[-1,n] = 0
   
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

anim.save('1D_acosutic.gif', dpi=80, writer='imagemagick')
