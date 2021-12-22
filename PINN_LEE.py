import tensorflow as tf
import numpy as np
tf.keras.backend.set_floatx('float64')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 7]
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import h5py
import os

# --- Disable eager execution
tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = ""



# Read the data from reference file
hf = h5py.File('LEE_Data','r')
t = np.array(hf.get('t')).T
x = np.array(hf.get('x')).T
Exact_p = np.array(hf.get('p')).T
Exact_v = np.array(hf.get('v')).T
hf.close()
c = 1
rho = 1.225

X, T = np.meshgrid(x,t)

X_all = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
Pexact_all = Exact_p.flatten()[:,None]
Vexact_all = Exact_v.flatten()[:,None]     

# Domain bounds
lb = X_all.min(0)
ub = X_all.max(0)
print(lb)
print(ub)


# matplotlib inline
plt.figure()
plt.contourf(Exact_p)

plt.figure()
plt.contour(Exact_v)


# initial condition
plt.figure()
plt.plot(x,Exact_p[0,:])

# start the sequential definition of the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer((2,)) )

# multiple feedforward layers
model.add(tf.keras.layers.Dense(20, activation='tanh'))
model.add(tf.keras.layers.Dense(20, activation='tanh'))
model.add(tf.keras.layers.Dense(20, activation='tanh'))
model.add(tf.keras.layers.Dense(20, activation='tanh'))
model.add(tf.keras.layers.Dense(20, activation='tanh'))
model.add(tf.keras.layers.Dense(20, activation='tanh'))
model.add(tf.keras.layers.Dense(20, activation='tanh'))
model.add(tf.keras.layers.Dense(20, activation='tanh'))
model.add(tf.keras.layers.Dense(20, activation='tanh'))

# output layer with a linear activation
model.add(tf.keras.layers.Dense(1, activation='linear'))


# First, we define the points associated with the initial condition, which is known
# Uniformly distributed points in x-domain [-1,1] for the initial condition
# u = -sin(pi*x)
Nx_init = 200
x_init = np.linspace(lb[0],ub[0],Nx_init)
p_x_init = 0.2*np.sin(np.pi*x_init)
X_init = np.hstack((x_init[:,None],np.zeros(len(x_init))[:,None]) )

# Second, we define the points associated with the left and right Dirichlet boundary conditions (u=0)
# Uniformly distributed points in t-domain [0,1] for the left BC
Nt_BC = 200
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
p_train = np.zeros(len(t_BC))
T_init = np.hstack((np.zeros(len(t_BC))[:,None],t_BC[:,None]) )

X_trainIBC = np.vstack((X_init,T_init)) #  we combine the initial condition with the left BC into a single array
p_trainIBC = np.vstack((p_x_init[:,None],np.zeros(len(t_BC))[:,None]) )


Nt_BC = 200
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
p_train = np.interp(t_BC,t,Exact_p[:,int(0.1*len(x))])
T_init = np.hstack((x[int(0.1*len(x))]*np.ones(len(t_BC))[:,None],t_BC[:,None]) )
X_trainIBC = np.vstack((X_trainIBC,T_init)) #  we combine the initial condition with the left BC into a single array
p_trainIBC = np.vstack((p_trainIBC,p_train[:,None]) )


Nt_BC = 200
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
p_train = np.interp(t_BC,t,Exact_p[:,int(0.2*len(x))])
T_init = np.hstack((x[int(0.2*len(x))]*np.ones(len(t_BC))[:,None],t_BC[:,None]) )
X_trainIBC = np.vstack((X_trainIBC,T_init)) #  we combine the initial condition with the left BC into a single array
p_trainIBC = np.vstack((p_trainIBC,p_train[:,None]) )


Nt_BC = 200
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
p_train = np.interp(t_BC,t,Exact_p[:,int(0.3*len(x))])
T_init = np.hstack((x[int(0.3*len(x))]*np.ones(len(t_BC))[:,None],t_BC[:,None]) )
X_trainIBC = np.vstack((X_trainIBC,T_init)) #  we combine the initial condition with the left BC into a single array
p_trainIBC = np.vstack((p_trainIBC,p_train[:,None]) )


Nt_BC = 200
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
p_train = np.interp(t_BC,t,Exact_p[:,int(0.4*len(x))])
T_init = np.hstack((x[int(0.4*len(x))]*np.ones(len(t_BC))[:,None],t_BC[:,None]) )
X_trainIBC = np.vstack((X_trainIBC,T_init)) #  we combine the initial condition with the left BC into a single array
p_trainIBC = np.vstack((p_trainIBC,p_train[:,None]) )


Nt_BC = 200
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
p_train = np.interp(t_BC,t,Exact_p[:,int(0.5*len(x))])
T_init = np.hstack((x[int(0.5*len(x))]*np.ones(len(t_BC))[:,None],t_BC[:,None]) )
X_trainIBC = np.vstack((X_trainIBC,T_init)) #  we combine the initial condition with the left BC into a single array
p_trainIBC = np.vstack((p_trainIBC,p_train[:,None]) )


Nt_BC = 200
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
p_train = np.interp(t_BC,t,Exact_p[:,int(0.6*len(x))])
T_init = np.hstack((x[int(0.6*len(x))]*np.ones(len(t_BC))[:,None],t_BC[:,None]) )
X_trainIBC = np.vstack((X_trainIBC,T_init)) #  we combine the initial condition with the left BC into a single array
p_trainIBC = np.vstack((p_trainIBC,p_train[:,None]) )


Nt_BC = 200
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
p_train = np.interp(t_BC,t,Exact_p[:,int(0.7*len(x))])
T_init = np.hstack((x[int(0.7*len(x))]*np.ones(len(t_BC))[:,None],t_BC[:,None]) )
X_trainIBC = np.vstack((X_trainIBC,T_init)) #  we combine the initial condition with the left BC into a single array
p_trainIBC = np.vstack((p_trainIBC,p_train[:,None]) )

Nt_BC = 200
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
p_train = np.interp(t_BC,t,Exact_p[:,int(0.8*len(x))])
T_init = np.hstack((x[int(0.8*len(x))]*np.ones(len(t_BC))[:,None],t_BC[:,None]) )
X_trainIBC = np.vstack((X_trainIBC,T_init)) #  we combine the initial condition with the left BC into a single array
p_trainIBC = np.vstack((p_trainIBC,p_train[:,None]) )

Nt_BC = 200
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
p_train = np.interp(t_BC,t,Exact_p[:,int(0.9*len(x))])
T_init = np.hstack((x[int(0.9*len(x))]*np.ones(len(t_BC))[:,None],t_BC[:,None]) )
X_trainIBC = np.vstack((X_trainIBC,T_init)) #  we combine the initial condition with the left BC into a single array
p_trainIBC = np.vstack((p_trainIBC,p_train[:,None]) )

# Uniformly distributed points in t-domain [0,1] for the right BC
Nt_BC = 200
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
p_train = np.zeros(len(t_BC))
T_init = np.hstack((np.ones(len(t_BC))[:,None],t_BC[:,None]) )
X_trainIBC = np.vstack((X_trainIBC,T_init))
p_trainIBC = np.vstack((p_trainIBC,np.zeros(len(t_BC))[:,None]) )


# WE DO NOT DISTRIBUTE UNIFORMLY THE POINTS (this creates an issue because of the symetry of the problem). Instead
# we are specifying a distribution of points using a "Latin Hypercube distribution"
# This ensures a near-random distribution that still fills our space-time domain
N_coloc = 80000
X_coloc_train = lb + (ub-lb)*lhs(2, N_coloc)


# This function allows us to compute the residual of the physical equation.
# To do that we use the built-in tf.gradients function that computes the gradient of a tensorflow function
@tf.function
def net_f(coloc_tensor):
    p = model(coloc_tensor)
    dp = tf.gradients(p, coloc_tensor)[0] # tf.gradients provides the gradient of u with respect to coloc_tensor and we select the 1st element    u_t = du[:,1:]
    p_x = dp[:,0:1] # given that coloc_tensor is a 2-dimensional array, u_x is the first element, u_t is the second_one
    p_t = dp[:,1:2]
    p_xx = tf.gradients(p_x, coloc_tensor)[0][:,0:1]
    p_tt = tf.gradients(p_t, coloc_tensor)[0][:,1:2]
    f = p_tt - c*p_xx
    return f

def custom_loss_wrapper(coloc_tensor):
    
    def custom_loss(y_true, y_pred):
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        f = net_f(coloc_tensor)
        derivative_loss = tf.reduce_mean( tf.square( f) )
        return mse_loss + derivative_loss

    return custom_loss



model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=custom_loss_wrapper(X_coloc_train))


checkpoint_filepath = './tmp2/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)



hist = model.fit(X_trainIBC, p_trainIBC, batch_size=150, epochs=10000, callbacks=[early_stop_callback,model_checkpoint_callback])
model.load_weights(checkpoint_filepath)

loss_history = hist.history['loss']
#matplotlib inline
plt.plot(loss_history)
plt.title("Loss History")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()


p_pred = model.predict(X_all)
p_nn = np.reshape(p_pred,[5000,200])
plt.contourf(x,t,p_nn)