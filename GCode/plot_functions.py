import GPy
import numpy as np
import pylab as pb
import datetime
import pandas as pd

#import sys
#path_ref = '../../'
#sys.path.append(path_ref)
import GCode

def XtoT(X,zero_day='20030101',scale=360.,offset=0.):
    """
    This function converts an input (days passed from zero_day) into dates
    """
    assert X.shape[1] == 1, 'More dimensions than expected'
    X = X*scale + offset
    zero_day = pd.Timestamp(zero_day)

    timeline = [zero_day + datetime.timedelta(days=int(x)) for x in X.flatten()]
    return timeline

def x_frame1D(X,plot_limits=None,resolution=None):
    """
    Internal helper function for making plots, returns a set of input values to plot as well as lower and upper limits
    """
    assert X.shape[1] ==1, "x_frame1D is defined for one-dimensional inputs"
    if plot_limits is None:
        xmin,xmax = X.min(0),X.max(0)
        xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
    elif len(plot_limits)==2:
        xmin, xmax = plot_limits
    else:
        raise ValueError, "Bad limits for plotting"

    Xnew = np.linspace(xmin,xmax,resolution or 200)[:,None]
    return Xnew, xmin, xmax


def plot1Dobserved(m,X=None,fixed_inputs=[],colormarker='Teal',colorbars='DarkKhaki',marker='o',ax=None,lw=2,ts=False,zero_day=None,scale=360.,offset=0,Y_metadata=None,colorbars2='red',scale_axis=(0,1)):

    if X is None:
        X = m.X

    if Y_metadata is None:
        pred,var = m.predict(X)
    else:
        pred,var = m.predict(X,Y_metadata=Y_metadata)

    #Defines the interpolation inputs
    fixed_dims = np.array([i for i,v in fixed_inputs])
    free_dims = np.setdiff1d(np.arange(m.input_dim),fixed_dims)

    D = pd.DataFrame({})
    D['pred'] = pred.flatten()
    D['error'] = 2.*np.sqrt(var)

    if not isinstance(m.likelihood,GPy.likelihoods.Gaussian):
        if Y_metadata is None:
            dl,du = m.predict_quantiles(X)
        else:
            dl,du = m.predict_quantiles(X,Y_metadata=Y_metadata)
        D['upper'] = du.flatten()
        D['lower'] = dl.flatten()

    if ax is None:
        fig = pb.figure()
        ax = fig.add_subplot(111)

    if ts:
        D['date'] = pd.to_datetime(XtoT(X[:,free_dims],scale=scale,offset=offset))
        D.set_index(['date'],inplace=True)
        D.index.name = None
        D.pred.plot(yerr=D.error,ecolor=colorbars,fmt=None,ax=ax,lw=2.)
        #D.sort_index(inplace=True)
        #if Y_metadata is None:
            #D.pred.plot(yerr=D.error,ecolor=colorbars,fmt=None,ax=ax)
        #else:
            #D.pred.plot(yerr=D.error,ecolor=colorbars,fmt=None,ax=ax)
            #mask = Y_metadata['output_index'].flatten() == 0
            #D.pred[mask].plot(yerr=D.error[mask],ecolor=colorbars,fmt=None,ax=ax)
            #D.pred[~mask].plot(yerr=D.error[~mask],ecolor=colorbars2,fmt=None,ax=ax,lw=2.)
        #D.pred.plot(color=colormarker,marker=marker,lw=0,ax=ax)
        #D.pred.plot(ax=ax,color='Teal',lw=2)

    else:
        D['X'] = X[:,free_dims]*(scale_axis[1]-scale_axis[0])
        if isinstance(m.likelihood,GPy.likelihoods.Gaussian):
            ax.errorbar(D.X,D.pred,yerr=D.error,ecolor=colorbars,fmt=None,lw=2.)
        else:
            interval = np.vstack([D.upper-D.pred,-D.lower+D.pred])
            ax.errorbar(D.X,D.pred,yerr=interval,ecolor=colorbars,fmt=None,lw=2.)
        ax.plot(D.X,D.pred,color=colormarker,marker=marker,lw=0)
        #ax.plot(ax=ax,color='Teal',lw=2)
        #D.pred.plot(color=colormarker,marker=marker,lw=0,ax=ax)
        #D.pred.plot(ax=ax,color='Teal',lw=2)
    ax.grid(True)


def plot1Dinterpolation(m,fixed_inputs=[],colorline='Teal',colorCI='DarkKhaki',marker='o',ax=None,plot_limits=None,resolution=None,lw=2,ts=False,zero_day=None,scale=360.,offset=0,latent=False,output_index=None,which_data_rows='all',kernel=None,Xgrid=None):

    #TODO kernel not always used

    if which_data_rows == 'all':
        which_data_rows = slice(None)

    #Defines the interpolation inputs
    fixed_dims = np.array([i for i,v in fixed_inputs])
    free_dims = np.setdiff1d(np.arange(m.input_dim),fixed_dims)
    assert free_dims.size == 1, 'Number of free dimensions is not 1'
    freeX,freeXmin,freeXmax = x_frame1D(m.X[which_data_rows,free_dims],plot_limits,resolution)

    if Xgrid is None:
        #Creates a new array with the interpolation points and the fixed inputs
        Xnew = np.empty((freeX.shape[0],m.input_dim))
        Xnew[:,free_dims] = freeX
        for i,v in fixed_inputs:
            Xnew[:,i] = v
    else:
        Xnew = Xgrid

    if latent:
        pred,var = m._raw_predict(Xnew,kern=kernel)
    else:
        if output_index is None:
            pred,var = m.predict(X,kern=kernel)
        else:
            Ymeta = {'output_index':(np.ones((Xnew.shape[0],1))*output_index).astype(int)}
            pred,var = m.predict(Xnew,kern=kernel,Y_metadata=Ymeta)

    #If the free input dimension is related to time
    D = pd.DataFrame({})
    D['pred'] = pred.flatten()
    if latent or isinstance(m.likelihood,GPy.likelihoods.Gaussian):
        D['upper'] = (pred + 2*np.sqrt(var)).flatten()
        D['lower'] = (pred - 2*np.sqrt(var)).flatten()
    else:
        if output_index is None:
            dl,du = m.predict_quantiles(Xnew)
        else:
            dl,du = m.predict_quantiles(Xnew,Y_metadata=Ymeta)
        D['upper'] = du.flatten()
        D['lower'] = dl.flatten()


    if ax is None:
        fig = pb.figure()
        ax = fig.add_subplot(111)

    if ts:
        D['date'] = pd.to_datetime(XtoT(Xnew[:,free_dims],scale=scale,offset=offset))
        D.set_index(['date'],inplace=True)
        D.index.name = None
        D.sort_index(inplace=True)
        if colorCI is not None:
            ax.fill_between(D.index,D.lower,D.upper,color=colorCI,alpha=.7)
        D.pred.plot(color=colorline,lw=lw,ax=ax)

    else:
        D['X'] = Xnew[:,free_dims]
        if colorline is not None:
            ax.plot(D.X,D.pred,color=colorline,lw=lw)
        if colorCI is not None:
            ax.fill_between(D.X,D.lower,D.upper,color=colorCI,alpha=.7)
        ax.grid()


def plotcycle(m,fixed_inputs=[],colorline='Teal',colorCI='DarkKhaki',marker='o',ax=None,plot_limits=None,resolution=None,lw=2,ts=False,zero_day=None,scale=360.,offset=0):

    #Defines the interpolation inputs
    fixed_dims = np.array([i for i,v in fixed_inputs])
    free_dims = np.setdiff1d(np.arange(m.input_dim),fixed_dims)
    assert free_dims.size == 1, 'Number of free dimensions is not 1'
    freeX,freeXmin,freeXmax = x_frame1D(m.X[:,free_dims],plot_limits,resolution)

    #Creates a new array with the interpolation points and the fixed inputs
    Xnew = np.empty((freeX.shape[0],m.input_dim))
    Xnew[:,free_dims] = freeX
    for i,v in fixed_inputs:
        Xnew[:,i] = v

    pred,var = m.predict(Xnew)
    dpred,dvar = GCode.dgp(m,Xnew)

    if ax is None:
        fig = pb.figure()
        ax = fig.add_subplot(111)

    if ts:
        stop

    ax.plot(dpred,pred,'r-')
    ax.plot(dpred[-1],pred[-1],'ro')
    ax.errorbar(dpred[-1],pred[-1],xerr=2*np.sqrt(dvar[-1]),yerr=2*np.sqrt(var[-1]))



def plot1Dinterpolation2(m,fixed_inputs=[],colorline='Teal',colorCI='DarkKhaki',marker='o',ax=None,plot_limits=None,resolution=None,lw=2,ts=False,zero_day=None,scale=360.,offset=0,latent=False,output_index=None,which_data_rows='all'):

    if which_data_rows == 'all':
        which_data_rows = slice(None)

    #Defines the interpolation inputs
    fixed_dims = np.array([i for i,v in fixed_inputs])
    free_dims = np.setdiff1d(np.arange(m.input_dim),fixed_dims)
    assert free_dims.size == 1, 'Number of free dimensions is not 1'
    freeX,freeXmin,freeXmax = x_frame1D(m.X[which_data_rows,free_dims],plot_limits,resolution)
    #freeX,freeXmin,freeXmax = x_frame1D(m.X[:,free_dims],plot_limits,resolution)

    #Creates a new array with the interpolation points and the fixed inputs
    Xnew = np.empty((freeX.shape[0],m.input_dim))
    Xnew[:,free_dims] = freeX
    for i,v in fixed_inputs:
        Xnew[:,i] = v

    if latent:
        pred,var = m._raw_predict(Xnew)
    else:
        if output_index is None:
            pred,var = m.predict(X)
        else:
            Ymeta = {'output_index':(np.ones((Xnew.shape[0],1))*output_index).astype(int)}
            pred,var = m.predict(Xnew,Y_metadata=Ymeta)

    #If the free input dimension is related to time
    D = pd.DataFrame({})
    D['pred'] = pred.flatten()
    #D['upper'] = (pred + 2*np.sqrt(var)).flatten()
    #D['lower'] = (pred - 2*np.sqrt(var)).flatten()
    if (not latent) and isinstance(m.likelihood,GPy.likelihoods.Gaussian):
        if output_index is None:
            dl,du = m1.predict_quantiles(Xnew)
        else:
            dl,du = m.predict_quantiles(Xnew,Y_metadata=Ymeta)
        D['upper'] = du.flatten()
        D['lower'] = dl.flatten()
    else:
        D['upper'] = (pred + 2*np.sqrt(var)).flatten()
        D['lower'] = (pred - 2*np.sqrt(var)).flatten()

    if ax is None:
        fig = pb.figure()
        ax = fig.add_subplot(111)

    if ts:



        D['date'] = XtoT(Xnew[:,free_dims],scale=scale,offset=offset)
        #pd.to_datetime(XtoT(Xnew[:,free_dims],scale=scale,offset=offset))
        D.set_index(['date'],inplace=True)
        D.index.name = None
        D.sort_index(inplace=True)
        if colorCI is not None:
            ax.fill_between(D.index,D.lower,D.upper,color=colorCI,alpha=.7)
        D.pred.plot(color=colorline,lw=lw,ax=ax)

    else:
        D['X'] = Xnew[:,free_dims]
        if colorline is not None:
            ax.plot(D.X,D.pred,color=colorline,lw=lw)
        if colorCI is not None:
            ax.fill_between(D.X,D.lower,D.upper,color=colorCI,alpha=.7)
        ax.grid()

def XtoT2(X,zero_day='20030101',scale=360.,offset=0.):
    """
    This function converts an input (days passed from zero_day) into dates
    """
    assert X.shape[1] == 1, 'More dimensions than expected'
    X = X*scale + offset
    #zero_day = pd.Timestamp(zero_day)
    zero_day = datetime.datetime(2003,1,1)

    timeline = pd.Series([zero_day + datetime.timedelta(days=int(x)) for x in X.flatten()])
    return timeline

