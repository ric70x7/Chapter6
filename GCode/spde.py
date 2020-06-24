import numpy as np
import pandas as pd
import pylab as pb
import datetime
import GPy
import shapefile

#import sys
#relative_path = '../'
#sys.path.append(relative_path)
#import Xtracode

def dtk(m,X,X2=None,dim=0):
    assert isinstance(m.kern,GPy.kern.Kern)

    assert X.shape[1] == m.kern.input_dim
    if X2 is None:
        X2 = X
    else:
        assert X2.shape[1] == m.kern.input_dim
    n1 = X.shape[0]
    n2 = X2.shape[0]
    L = m.kern.lengthscale if not m.kern.ARD else m.kern.lengthscale[dim]

    r = X[:,dim:dim+1] - X2[:,dim:dim+1].T
    if n1!=n2:
        pass
    if isinstance(m.kern,GPy.kern.RBF):
        K = - r/L**2. * m.kern.K(X,X2)
    else:
        raise NotImplementedError
    return K


def ddtk(m,X,X2=None,dim=0):
    assert isinstance(m.kern,GPy.kern.RBF)

    assert X.shape[1] == m.kern.input_dim
    if X2 is None:
        X2 = X
    else:
        assert X2.shape[1] == m.kern.input_dim
    n1 = X.shape[0]
    n2 = X2.shape[0]
    L = m.kern.lengthscale if not m.kern.ARD else m.kern.lengthscale[dim]

    r = X[:,dim:dim+1] - X2[:,dim:dim+1].T
    if isinstance(m.kern,GPy.kern.RBF):
        K = 1./L**2. - r**2./L**4.
    return K

def dgp(m,X,dim=0,full_cov=False):
    Kdff = dtk(m,X,m.X,dim=dim)
    Kdfdf = ddtk(m,X,X)
    mu = np.dot(Kdff,m.posterior.woodbury_vector)
    var = Kdfdf - np.dot(np.dot(Kdff,m.posterior.woodbury_inv),-Kdff.T)# + np.eye(X.size)
    if not full_cov:
        var = np.diag(var)[:,None]
    return mu,var

def intgp(m,X,dim=0,full_cov=False):
    Kfdf = - dtk(m,m.X,X,dim=dim)
    Kdfdf = ddtk(m,m.X,m.X)

    Kfdf2 = ddtk(m,X,m.X)
    Wi = np.linalg.inv(Kdfdf + np.eye(Kdfdf.shape[0])*m.likelihood.variance)
    WiY = np.dot(Wi,np.cos(m.Y))
    mu2 = np.dot(Kfdf2,WiY)
    var2 = Kdfdf - np.dot(np.dot(Kfdf2,m.posterior.woodbury_inv),-Kfdf2.T) + np.eye(X.size) * m.likelihood.variance
    if not full_cov:
        var2 = np.diag(var2)[:,None]
    return mu2,var2






