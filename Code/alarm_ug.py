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


data = pd.read_csv('../../Data/filtered_hmis_data.csv',parse_dates=True)
data['date'] = pd.to_datetime(data['date'])
data.set_index(['district','splitID','date'],inplace=True)
data.sort_index(inplace=True)

parents = pd.read_csv('../../Data/districts_parenthood.csv',parse_dates=True)
epochs = pd.read_csv('../../Data/epochs.csv',parse_dates=True)

fontsize = 20
ticksize = 20
labelsize = 20
mymap = pb.matplotlib.colors.ListedColormap(['DarkCyan','CadetBlue','LightSteelBlue','SteelBlue','Navy'])

def data_transf(y):
    return np.log(y)
    #return y

def data_transfinv(y):
    #return np.exp(y)
    return y

_districts = parents.district[parents.splits==0][16:20]
#_districts = ['Kalangala']
#_districts = ['Kaberamaido']

tp = np.linspace(0,13,200)[:,None]
t0 = np.linspace(0,13,6)[:,None]

for k,colori in zip(_districts,['r','b']):
    print k

    s  = int(parents.splits[parents.district==k])
    _data = data.ix[k].ix[s]
    _mask = np.logical_or(_data.outliers==0,_data.outliers==9)
    _data = _data[_mask]
    _data.index.name = ''

    t1 = _data.X/360.
    rep1 = _data.reported
    X = np.vstack([t1,rep1]).T
    Y = data_transf(_data.malaria_cases)[:,None]

    S1 = GPy.kern.RBF(1,lengthscale=45./360.,variance=.05,active_dims=[0],name='S1')
    S1.lengthscale.constrain_bounded(1./12.,.5)
    S2 = GPy.kern.Linear(1,active_dims=[1])
    S3 = GPy.kern.RBF(1,lengthscale=200./360.,variance=50,active_dims=[0],name='S3')
    S3.lengthscale.constrain_bounded(5.,10.)

    G = GPy.kern.DiffRBF3S(kernel=S1,kernel2=S2,kernel3=S3,beta=1.)
    G.beta.constrain_fixed(0)
    G.alpha.constrain_fixed(1.)

    m = GPy.models.GPSPDE([X,np.array([]).reshape(0,X.shape[1])],[Y,np.array([]).reshape(0,1)],G)
    m.optimize()

    _Xp = tp.copy()
    Xp = np.hstack([_Xp,np.ones_like(_Xp)*rep1.mean()])
    _X0 = t0.copy()
    X0 = np.hstack([_X0,np.ones_like(_X0)*rep1.mean()])

    #Simulated data
    fig1 = pb.figure(figsize=(11,5))
    ax1 = fig1.add_subplot(111)#,aspect='equal')
    Y3p, V3p = m._raw_predict(np.hstack([Xp,np.ones_like(_Xp)]),kern=m.kern.adj3_kernel)

    _Y3p = Y3p - Y3p.mean() + data_transf(_data.malaria_cases).mean()
    Y3aux = pd.DataFrame({'trend':_Y3p.flatten()},index=GCode.XtoT(_Xp))


    data_transf(_data.malaria_cases).plot(ax=ax1,marker='o',color='gray',lw=0,alpha=.3,zorder=0)
    Y3aux.trend.plot(ax=ax1,color='r',lw=3)

    Y1p, V1p = m._raw_predict(np.hstack([Xp,np.zeros_like(_Xp)]),kern=m.kern.base_kernel)
    dY1p, dV1p = m._raw_predict(np.hstack([Xp,np.ones_like(_Xp)]))#,kern=m.kern.base_kernel)

    _Y1p = Y1p - Y1p.mean() + _Y3p
    Y1aux = pd.DataFrame({'short_term':_Y1p.flatten()},index=GCode.XtoT(_Xp))
    Y1aux.short_term.plot(ax=ax1,color='darkblue',lw=3,linestyle='-')

    a = Y1aux.shape[0] - 48
    b = Y1aux.shape[0] - 25

    _x1 = Y1aux.shape[0] - 45
    _x2 = Y1aux.shape[0] - 42
    _x3 = Y1aux.shape[0] - 27
    _x4 = Y1aux.shape[0] - 25

    #Y1aux[a:b].short_term.plot(ax=ax1,color='darkblue',lw=3)
    Y1aux[_x1-1:_x1].short_term.plot(ax=ax1,marker='o',color='Teal',lw=0,mew=2.5,mec='darkblue')
    Y1aux[_x2-1:_x2].short_term.plot(ax=ax1,marker='o',color='Teal',lw=0,mew=2.5,mec='darkblue')
    Y1aux[_x3-1:_x3].short_term.plot(ax=ax1,marker='o',color='Teal',lw=0,mew=2.5,mec='darkblue')
    Y1aux[_x4-1:_x4].short_term.plot(ax=ax1,marker='o',color='Teal',lw=0,mew=2.5,mec='darkblue')

    ax1.text(Y1aux[_x1-1:_x1].index,Y1aux[_x1-1:_x1].values-.3,'A',fontsize=20,color='darkblue')
    ax1.text(Y1aux[_x2-1:_x2].index,Y1aux[_x2-1:_x2].values-.3,'B',fontsize=20,color='darkblue')
    ax1.text(Y1aux[_x3-1:_x3].index,Y1aux[_x3-1:_x3].values+.05,'C',fontsize=20,color='darkblue')
    ax1.text(Y1aux[_x4-1:_x4].index,Y1aux[_x4-1:_x4].values-.3,'D',fontsize=20,color='darkblue')



    _x0,_xf = GCode.XtoT(np.array([4./360.,4204./360.])[:,None])
    ax1.set_xlim(_x0,_xf)
    ax1.set_ylim(6.,10)
    ax1.set_ylabel('malaria cases (log scale)',fontsize=fontsize)
    ax1.tick_params(labelsize=ticksize)
    pb.tight_layout()
    fig1.savefig('../Figs/ts_decomposed.pdf',dpi=100.)

    fig4 = pb.figure(figsize=(6,6))
    ax4 = fig4.add_subplot(111,aspect='equal')
    zx = (Y1p - Y1p.mean())/np.sqrt(V1p)
    zy = (dY1p)/np.sqrt(dV1p)

    ax4.plot( zx[a:b], zy[a:b],color='darkblue',lw=2)
    ax4.set_xlim(-5,5)
    ax4.set_ylim(-5,5)

    ax4.plot( zx[_x1-1], zy[_x1-1],'o',mew=3.5,color='darkblue',mec='darkblue')
    ax4.plot( zx[_x2-1], zy[_x2-1],'o',mew=3.5,color='darkblue',mec='darkblue')
    ax4.plot( zx[_x3-1], zy[_x3-1],'o',mew=3.5,color='darkblue',mec='darkblue')
    ax4.plot( zx[_x4-1], zy[_x4-1],'o',mew=3.5,color='darkblue',mec='darkblue')

    ax4.text( zx[_x1-1], zy[_x1-1]-.5,'A',color='darkblue',fontsize=20)
    ax4.text( zx[_x2-1], zy[_x2-1]+.2,'B',color='darkblue',fontsize=20)
    ax4.text( zx[_x3-1], zy[_x3-1]-.6,'C',color='darkblue',fontsize=20)
    ax4.text( zx[_x4-1], zy[_x4-1]+.2,'D',color='darkblue',fontsize=20)


    ax4.fill(np.array([-5,0,0,-5]), np.array([5, 5, 0, 0]),color='Yellow',ec='none',alpha=.5,zorder=0)
    ax4.fill(np.array([5,0,0,5]), np.array([-5, -5, 0, 0]),color='DarkOrange',ec='none',alpha=.5,zorder=0)
    ax4.fill(np.array([-5,0,0,-5]), np.array([-5, -5, 0, 0]),color='Forestgreen',ec='none',alpha=.5,zorder=0)
    ax4.fill(np.array([5,0,0,5]), np.array([5, 5, 0, 0]),color='red',ec='none',alpha=.5,zorder=0)
    ax4.set_xlabel('variation around the trend',fontsize=fontsize)
    ax4.set_ylabel('rate of change',fontsize=fontsize)
    ax4.tick_params(labelsize=ticksize)
    fig4.savefig('../Figs/ts_track.pdf',dpi=100.)

    ax4.grid()
    pb.tight_layout()

    stop
