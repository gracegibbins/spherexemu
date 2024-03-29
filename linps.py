import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sp # for calculating standard error
from math import cos, exp, pi
from scipy.integrate import quad, quad_vec
from scipy import interpolate
from scipy import integrate
from scipy.special import eval_legendre
from fastpt import FASTPT
import camb
from camb import model
import pyDOE
from pyDOE import lhs
import time

from gpsclass import CalcGalaxyPowerSpec


#Galaxy Bias Parameter
bias = np.array([1.9,-0.6,(-4./7)*(1.9-1),(32./315.)*(1.9-1)])

#Cosmo Parameters
prior = np.array([[60,85], #H0
                  [0.01,0.04], #omega_b
                  [0.01,0.3], #omega_c
                  [1.2e-9,2.7e-9], #As
                  [0.87,1.07]]) #ns

#Number of cosmo params in prior array
n_dim = 5

#g: inputs number of wanted samples and prior array to output number of samples with randomly set cosmology from prior
def create_lhs_samples(n_samples , prior):
    lhs_samples = lhs(n_dim, n_samples) #creates lhc with values 0 to 1
    cosmo_samples = prior[:,0] + (prior[:,1] - prior[:,0]) * lhs_samples #scales each value to the given priors
    return cosmo_samples

#Creates linear power spectra from priors - input into galaxy ps class
def get_linps(params):
    npoints = 20 #smallest possible is four & need to be even numbers
    ps = np.zeros((len(params[:,0]),npoints)) #number of samples x number of k bins
    k = np.zeros((len(params[:,0]),npoints)) #number of samples x number of k bins
    for row in range(len(params[:,0])):
        print("params:", params[row])
        H0, ombh2, omch2, As, ns = params[row,0], params[row,1], params[row,2], params[row,3], params[row,4]
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
        pars.InitPower.set_params(As=As, ns=ns)
        pars.set_matter_power(redshifts=[0.6], kmax=2.0) #sets redshift and mode for ps
        pars.NonLinear = model.NonLinear_none #set to be linear
        results = camb.get_results(pars)
        kh, z, pk = results.get_matter_power_spectrum(minkh=1e-3, maxkh=.2, npoints=npoints) #pk is 2 values 
        f = .7 #####PLACEHOLDER
        nonlin = CalcGalaxyPowerSpec(f,pk[0],kh,bias,params[row])
        ps_nonlin = nonlin.get_nonlinear_ps(0)
        k[row] = (kh)
        ps[row] = ps_nonlin #(pk[0])
    return k, ps #karray, Psnonlin = get_linps(params)

#Number of PS to Generate
x = 1

k_array, p_array = (get_linps(create_lhs_samples(x,prior)))

print("K Values:", k_array, "\nPower Spectrum Values:", p_array)

plt.plot(k_array, p_array)

plt.show()