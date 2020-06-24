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
#mymap = pb.matplotlib.colors.ListedColormap(['CadetBlue','LightSteelBlue','SteelBlue'])
mymap = pb.matplotlib.colors.ListedColormap(['DarkCyan','CadetBlue','LightSteelBlue','SteelBlue','Navy'])

def data_transf(y):
    return np.log(y)
    #return y

def data_transfinv(y):
    #return np.exp(y)
    return y

_districts = parents.district
tp = np.linspace(8,11,150)[55:60]

ZZZ = 0
for day in tp:

    Yb = []
    dYb = []
    for k in _districts:
        s  = int(parents.splits[parents.district==k])
        print k,s
        _data = data.ix[k].ix[s]
        _mask = np.logical_or(_data.outliers==0,_data.outliers==9)
        _data = _data[_mask]
        _data = _data[_data.X <= day*360]
        _data.index.name = ''

        if _data.shape[0] == 0:
            Yb.append(0)
            dYb.append(0)

        else:
            t1 = _data.X/360.
            rep1 = _data.reported
            X = np.vstack([t1,rep1]).T
            Y = data_transf(_data.malaria_cases)[:,None]

            S1 = GPy.kern.RBF(1,lengthscale=45./360.,variance=.05,active_dims=[0],name='S1')
            S1.lengthscale.constrain_bounded(9./36.,.5)
            S2 = GPy.kern.Linear(1,active_dims=[1])
            S3 = GPy.kern.RBF(1,lengthscale=200./360.,variance=50,active_dims=[0],name='S3')
            S3.lengthscale.constrain_bounded(5.,10.)

            G = GPy.kern.DiffRBF3S(kernel=S1,kernel2=S2,kernel3=S3,beta=1.)
            G.beta.constrain_fixed(0)
            G.alpha.constrain_fixed(1.)

            m = GPy.models.GPSPDE([X,np.array([]).reshape(0,X.shape[1])],[Y,np.array([]).reshape(0,1)],G)
            m.optimize()

            #_Xp = tp.copy()
            _Xp = np.array([day])[:,None]
            Xp = np.hstack([_Xp,np.ones_like(_Xp)*rep1.mean()])
            #_X0 = t0.copy()
            #X0 = np.hstack([_X0,np.ones_like(_X0)*rep1.mean()])

            Y1p, V1p = m._raw_predict(np.hstack([Xp,np.zeros_like(_Xp)]),kern=m.kern.base_kernel)
            dY1p, dV1p = m._raw_predict(np.hstack([Xp,np.ones_like(_Xp)]))#,kern=m.kern.base_kernel)


            _XX = m.X.copy()
            _XX[:,1] = rep1.mean()
            _dXX = _XX.copy()
            _dXX[:,-1] = 1
            Y1, V1 = m._raw_predict(_XX,kern=m.kern.base_kernel)
            dY1, dV1 = m._raw_predict(_dXX)#,kern=m.kern.base_kernel)

            Y1p = Y1p - Y1.mean()
            Y1p /= np.sqrt(V1p)
            dY1p/= np.sqrt(dV1p)

            Yb.append(Y1p[0,0])
            dYb.append(dY1p[0,0])


    fig = pb.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111,aspect='equal')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    GCode.show_ug_map2010(ax=ax1,edgecolor='Gainsboro',facecolor='Gainsboro')
    for k,y,dy in zip(_districts,Yb,dYb):

        _pin = True
        if abs(y) < 1 and abs(dy) < 1:
            _pin = False
        elif y > 0 and dy > 0:
            c = 'r'
        elif y < 0 and dy < 0:
            c = 'ForestGreen'
        elif y < 0 and dy > 0:
            c = 'Yellow'
        elif y > 0 and dy < 0:
            c = 'DarkOrange'
        else:
            _pin = False

        if _pin:
            GCode.pinpoint_districts2(districts=[k],current_map=True,ax=ax1,edgecolor='DarkRed',facecolor=c,alpha=.5,legend=False,legend_loc=4)

    aux = aux = GCode.XtoT(np.array([day])[:,None])[0]
    _year = aux.year
    _week = aux.weekofyear
    ax1.set_title('%s: week %s' %(_year,_week))
    framenum = str(ZZZ)
    if len(framenum) == 1:
        framenum = '0' + framenum
    fig.savefig('../Figs/all_country%s.pdf'%framenum,dpi=100.)
    pb.tight_layout()
    ZZZ += 1
    pb.close()
