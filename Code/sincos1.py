"""
Sin - cos example
"""

import numpy as np
import pandas as pd
import pylab as pb
import GPy
import scipy as sp

#This part is to import the GCode
import sys
path_ref = '../../'
sys.path.append(path_ref)
import GCode

#Plots parameters
ticksize = 20

plot_signals = True
plot_derivatives = False
joint_model = True
signal1_model = True
signal2_model = True
noise = True

def arrow(centre,slope,radius):
    x0,y0 = centre
    _s = np.sign(slope) if slope != 0 else 1
    theta = np.arccos( _s*1./np.sqrt(1. + slope**2.))
    x1 = radius*np.cos(theta) + x0
    y1 = radius*np.sin(theta) + y0
    x2 = radius*np.cos(np.pi + theta) + x0
    y2 = radius*np.sin(np.pi + theta) + y0
    _x = np.array([x2,x1])
    _y = np.array([y2,y1])
    return _x,_y


#Signals and derivatives
def fs1(x):
    return 2.*np.sin(x) + 3

def fs2(x):
    return ((x-15)/5.)**2 + 6

def dfs1(x):
    return 2.*np.cos(x)

def dfs2(x):
    return 2 * ((x-15)/5.) /5.

def f(X):
    return fs1(X) + fs2(X)

def df(X):
    return dfs1(X) + dfs2(X)

#Data points
num_data = 100
X = np.linspace(0,30,num_data)[:,None]
X_star = np.linspace(3,27,8.)[:,None]

E = np.random.normal(0,1.1,num_data)[:,None] if noise else np.zeros((num_data,1))

Y1 = fs1(X)
Y2 = fs2(X)
Y = f(X)

Y1_star = fs1(X_star)
Y2_star = fs2(X_star)
Y_star = f(X_star)

dY1_star = dfs1(X_star)
dY2_star = dfs2(X_star)
dY_star = df(X_star)


#Plot signals
if plot_signals:
    fig = pb.figure(figsize=(11,5))
    ax1 = fig.add_subplot(111,aspect='equal')
    ax1.plot(X,fs1(X),'r--',lw=3)
    ax1.plot(X,fs2(X),'r--',lw=3)
    #ax1.plot(X,f(X),'b-')
    ax1.set_ylim(0,16)
    ax1.tick_params(labelsize=ticksize)

    pb.tight_layout()
    pb.grid()
    fig.savefig('../Figs/indep_signals.pdf',dpi=100.)

    fig = pb.figure(figsize=(11,5))
    ax1 = fig.add_subplot(111,aspect='equal')
    ax1.plot(X,f(X)+E,'gray',lw=3)
    ax1.plot(X,f(X),'b--',lw=3)
    ax1.set_ylim(4,20)
    ax1.tick_params(labelsize=ticksize)
    pb.tight_layout()
    pb.grid()
    fig.savefig('../Figs/out_signal.pdf',dpi=100.)

#Derivatives
if plot_derivatives:

    #Both signals
    fig0 = pb.figure(figsize=(11,5))
    ax1 = fig0.add_subplot(111,aspect='equal')
    ax1.plot(X,Y,'b-')

    for x0,y0,dy in zip(X_star,Y_star,dY_star):
        _s = np.sign(dy)
        theta = np.arccos( _s*1./np.sqrt(1. + dy**2.))
        radius = 1.
        x1 = radius*np.cos(theta) + x0
        y1 = radius*np.sin(theta) + y0
        x2 = radius*np.cos(np.pi + theta) + x0
        y2 = radius*np.sin(np.pi + theta) + y0
        _x = np.array([x2,x1])
        _y = np.array([y2,y1])
        ax1.plot(_x,_y,'r-',lw=2)
    ax1.plot(X_star,Y_star,'rx',mew=1.5)
    pb.tight_layout()
    pb.grid()

    #Signals 1 and 2
    fig1 = pb.figure(figsize=(11,6.5))
    ax1 = fig1.add_subplot(212,aspect='equal')
    ax1.plot(X,Y1,'b-')

    for x0,y0,dy in zip(X_star,Y1_star,dY1_star):
        _s = np.sign(dy)
        theta = np.arccos( _s*1./np.sqrt(1. + dy**2.))
        radius = 1.
        x1 = radius*np.cos(theta) + x0
        y1 = radius*np.sin(theta) + y0
        x2 = radius*np.cos(np.pi + theta) + x0
        y2 = radius*np.sin(np.pi + theta) + y0
        _x = np.array([x2,x1])
        _y = np.array([y2,y1])
        ax1.plot(_x,_y,'r-',lw=2)
    ax1.plot(X_star,Y1_star,'rx',mew=1.5)
    pb.tight_layout()
    pb.grid()

    ax2 = fig1.add_subplot(211,aspect='equal')
    ax2.plot(X,Y2,'b-')

    for x0,y0,dy in zip(X_star,Y2_star,dY2_star):
        _s = np.sign(dy)
        theta = np.arccos( _s*1./np.sqrt(1. + dy**2.))
        radius = 1.
        x1 = radius*np.cos(theta) + x0
        y1 = radius*np.sin(theta) + y0
        x2 = radius*np.cos(np.pi + theta) + x0
        y2 = radius*np.sin(np.pi + theta) + y0
        _x = np.array([x2,x1])
        _y = np.array([y2,y1])
        ax2.plot(_x,_y,'r-',lw=2)
    ax2.plot(X_star,Y2_star,'rx',mew=1.5)
    pb.tight_layout()
    pb.grid()


#Model both signals together
if joint_model:

    K1 = GPy.kern.RBF(1,lengthscale=45./360.,variance=.5,active_dims=[0])
    K2 = GPy.kern.RBF(1,lengthscale=180./360.,variance=.5,active_dims=[0])
    K1.lengthscale.constrain_bounded(0,1.)
    K2.lengthscale.constrain_bounded(10,30)
    G = GPy.kern.DiffRBF2S(K1,K2,beta=1.)
    G.beta.constrain_fixed(0)
    G.alpha.constrain_fixed(1.)
    m = GPy.models.GPSPDE([X,np.array([]).reshape(0,X.shape[1])],[Y+E,np.array([]).reshape(0,1)],G)
    if not noise:
        m.mixed_noise.Gaussian_noise_0.variance.constrain_fixed(1e-4)
    m.checkgrad(verbose=True)
    m.optimize()

    Ysamps = m.posterior_samples_f(m.X,7)
    #Ysamps = m.posterior_samples(m.X,Y_metadata={'output_index':np.zeros_like(Y).astype(int)})
    Ys1, Vs1 =m.predict(m.X,Y_metadata={'output_index':np.zeros_like(Y).astype(int)},kern=m.kern.base_kernel)
    Ys2, Vs2 =m.predict(m.X,Y_metadata={'output_index':np.zeros_like(Y).astype(int)},kern=m.kern.adj_kernel)

    fig = pb.figure(figsize=(11,5))
    ax1 = fig.add_subplot(111,aspect='equal')
    for yi in Ysamps.T:
        ax1.plot(X,yi,'Teal',lw=1)
    ax1.plot(X,f(X)+E,'gray',lw=3)
    ax1.set_ylim(4,20)
    ax1.tick_params(labelsize=ticksize)
    pb.tight_layout()
    pb.grid()
    fig.savefig('../Figs/combined_samples.pdf',dpi=100.)

    fig2 = pb.figure(figsize=(11,5))
    ax2 = fig2.add_subplot(111,aspect='equal')
    ax2.plot(X,fs1(X),'r--',lw=3)
    ax2.plot(X,Ys1-Ys1.mean() + fs1(X).mean(),'Teal',lw=3)

    ax2.plot(X,fs2(X),'r--',lw=3)
    ax2.plot(X,Ys2-Ys2.mean() + fs2(X).mean(),'Teal',lw=3)
    ax2.set_ylim(0,16)
    ax2.tick_params(labelsize=ticksize)
    pb.tight_layout()
    pb.grid()
    fig2.savefig('../Figs/estimated_components.pdf',dpi=100.)

#Model both signals by separate
if signal1_model:

    K = GPy.kern.RBF(1,lengthscale=.9,variance=.3,active_dims=[0])
    K.lengthscale.constrain_bounded(0,1.)
    K2 = GPy.kern.RBF(1,lengthscale=.5,variance=.8,active_dims=[1])
    K2.lengthscale.constrain_bounded(10,30)

    G1 = GPy.kern.DiffRBF2S(K,K2,beta=1.)
    G1.beta.constrain_fixed(0)
    G1.alpha.constrain_fixed(1.)

    XX = np.hstack([X,X])
    XX_star = np.hstack([X_star,X_star])
    m = GPy.models.GPSPDE([XX,np.array([]).reshape(0,XX.shape[1])],[Y+E,np.array([]).reshape(0,1)],G1)
    if not noise:
        m.mixed_noise.Gaussian_noise_0.variance.constrain_fixed(1e-4)
    m.checkgrad(verbose=True)
    m.optimize()

    Y_pred, V_pred =m.predict(np.hstack([XX,np.zeros_like(X)]),Y_metadata={'output_index':np.zeros_like(X).astype(int)})
    Ys, Vs =m.predict(np.hstack([XX_star,np.zeros_like(X_star)]),Y_metadata={'output_index':np.zeros_like(X_star).astype(int)})
    #dY1s, dV1s =m.predict(np.hstack([XX_star,np.ones_like(X_star)]),Y_metadata={'output_index':np.ones_like(X_star).astype(int)})
    dY1s, dV1s =m._raw_predict(np.hstack([XX_star,np.ones_like(X_star)]))

    #Kernel for predictions of Signal 1
    Ks1 = GPy.kern.RBF(1,active_dims=[0])
    Ks1.lengthscale = m.kern.lengthscale_1
    Ks1.variance = m.kern.variance_1

    Y1_pred, V1_pred =m.predict(np.hstack([XX,np.zeros_like(X)]),Y_metadata={'output_index':np.zeros_like(X).astype(int)},kern=Ks1)
    Y1s, V1s =m.predict(np.hstack([XX_star,np.zeros_like(X_star)]),Y_metadata={'output_index':np.zeros_like(X_star).astype(int)},kern=Ks1)


    _shift = 7

    fig2 = pb.figure(figsize=(11,5))
    ax1 = fig2.add_subplot(111,aspect='equal')

    ax1.plot(X,Y_pred,'b-',lw=3)
    ax1.plot(X_star,Ys,'rx',mew=1.5)

    ax1.plot(X,Y1_pred + _shift,'b--',lw=3)
    ax1.plot(X_star,Y1s+_shift,'rx',mew=1.5)

    ax1.set_ylim(4,20)
    ax1.set_xlim(-0,30)
    ax1.tick_params(labelsize=ticksize)
    ax1.grid()

    for x0,y0,dy,dv,y1s in zip(X_star,Ys,dY1s,dV1s,Y1s):
        theta = np.arctan(dy)
        thetasd = np.arctan(np.sqrt(dv))
        _S = np.tan(np.random.normal(theta,thetasd,150))
        for slopei in _S:
            _xl,_yl = arrow(centre=(x0,y0),slope=slopei,radius=1.)
            ax1.plot(_xl,_yl,color='gray',alpha=.2)
        _x,_y = arrow(centre=(x0,y0),slope=dy,radius=1.)
        _x1s,_y1s = arrow(centre=(x0,y1s),slope=dy,radius=1.)
        ax1.plot(_x,_y,'r-',lw=3)
        ax1.plot(_x1s,_y1s+_shift,'r-',lw=3) #Arrow plotted on top of signal 1
        ax1.vlines(x0,y0,y1s+_shift,color='red',linestyle='--',lw=2)

    fig2.savefig('../Figs/der_sinsignal.pdf',dpi=100.)

if signal2_model:

    K = GPy.kern.RBF(1,lengthscale=.9,variance=.3,active_dims=[0])
    K.lengthscale.constrain_bounded(10.,30.)
    K2 = GPy.kern.RBF(1,lengthscale=.5,variance=.8,active_dims=[1])
    K2.lengthscale.constrain_bounded(0,1.)

    G1 = GPy.kern.DiffRBF2S(K,K2,beta=1.)
    G1.beta.constrain_fixed(0)
    G1.alpha.constrain_fixed(1.)

    XX = np.hstack([X,X])
    XX_star = np.hstack([X_star,X_star])
    m = GPy.models.GPSPDE([XX,np.array([]).reshape(0,XX.shape[1])],[Y+E,np.array([]).reshape(0,1)],G1)
    if not noise:
        m.mixed_noise.Gaussian_noise_0.variance.constrain_fixed(1e-4)
    m.checkgrad(verbose=True)
    m.optimize()

    Y_pred, V_pred =m.predict(np.hstack([XX,np.zeros_like(X)]),Y_metadata={'output_index':np.zeros_like(X).astype(int)})
    Ys, Vs =m.predict(np.hstack([XX_star,np.zeros_like(X_star)]),Y_metadata={'output_index':np.zeros_like(X_star).astype(int)})
    #dY2s, dV2s =m.predict(np.hstack([XX_star,np.ones_like(X_star)]),Y_metadata={'output_index':np.ones_like(X_star).astype(int)})
    dY2s, dV2s =m._raw_predict(np.hstack([XX_star,np.ones_like(X_star)]))

    #Kernel for predictions of Signal 2
    Ks2 = GPy.kern.RBF(1,active_dims=[0])
    Ks2.lengthscale = m.kern.lengthscale_1
    Ks2.variance = m.kern.variance_1

    Y2_pred, V2_pred =m.predict(np.hstack([XX,np.zeros_like(X)]),Y_metadata={'output_index':np.zeros_like(X).astype(int)},kern=Ks2)
    Y2s, V2s =m.predict(np.hstack([XX_star,np.zeros_like(X_star)]),Y_metadata={'output_index':np.zeros_like(X_star).astype(int)},kern=Ks2)


    Y2s -= 3. #NOTE The magnitude of each signal cannot be estimated
    Y2_pred -= 3. #NOTE The magnitude of each signal cannot be estimated


    _shift = 7

    fig2 = pb.figure(figsize=(11,5))
    ax1 = fig2.add_subplot(111,aspect='equal')

    ax1.plot(X,Y_pred,'b-',lw=3)
    ax1.plot(X_star,Ys,'rx',mew=1.5)

    ax1.plot(X,Y2_pred,'b--',lw=3)
    ax1.plot(X_star,Y2s,'rx')


    ax1.set_ylim(4,20)
    ax1.set_xlim(-0,30)
    ax1.tick_params(labelsize=ticksize)
    ax1.grid()

    for x0,y0,dy,dv,y1s in zip(X_star,Ys,dY2s,dV2s,Y2s):
        theta = np.arctan(dy)
        thetasd = np.arctan(np.sqrt(dv))
        _S = np.tan(np.random.normal(theta,thetasd,150))
        for slopei in _S:
            _xl,_yl = arrow(centre=(x0,y0),slope=slopei,radius=1.)
            ax1.plot(_xl,_yl,color='gray',alpha=.2)
        _x,_y = arrow(centre=(x0,y0),slope=dy,radius=1.)
        _x1s,_y1s = arrow(centre=(x0,y1s),slope=dy,radius=1.)
        ax1.plot(_x,_y,'r-',lw=3)
        ax1.plot(_x1s,_y1s,'r-',lw=3) #Arrow plotted on top of signal 1
        ax1.vlines(x0,y0,y1s,color='red',linestyle='--',lw=2)
    ax1.plot(X_star,Y2s,'rx',mew=1.5)

    fig2.savefig('../Figs/der_parsignal.pdf',dpi=100.)
