import numpy as np
import scipy as sp
from scipy.special import beta as betafunc
import matplotlib.pyplot as plt

import warnings

from project import *
from project.mcmc import *
from project.lorenz import *

warnings.filterwarnings("ignore")

def normal_lorenz_pdf(obs, pred):
	# This factor doesn't matter because we subtract the previous iteration anyways
	# We removed it to save computation
	#loglik = -1*len(obs)*np.log(10*np.sqrt(2*np.pi))
	loglik=0
	for i in xrange(len(obs)):
		p = pred[i][1]
		o = obs[i][1]
		loglik -= np.sum([x for x in 50*((o/p)-1)**2 if not np.isnan(x)])

	return loglik/len(obs)

def expo_lorenz_pdf(obs, pred):
	loglik = 0
	for i in xrange(len(obs)):
		p = pred[i][1]
		o = obs[i][1]
		loglik -= np.sum([x for x in 4*((o/p)-1)**2 if not np.isnan(x)])

	return loglik/len(obs)

############################################

def loop(mh, X):
	xs0 = np.array([mh.uniform("x",0,.2), mh.uniform("y",-.1,.1), mh.uniform("z",-.1,.1)])
	#xs0 = np.array([.1,0,0])

	sigma = mh.uniform("sigma",9,11)
	#sigma = 10.

	rho = mh.uniform("rho",27,29)
	#rho = 28.

	beta = mh.uniform("beta",7/3.,10/3.)
	#beta = 8/3.

	L = LorenzAttractor(xs0, sigma, rho, beta, tf=1.).rk4_out
	mh.condition(expo_lorenz_pdf(X, L))

mh = MCMC()
X = RLA(tf=1., rf=random_expo(1,5)).get_all()

t = mh.run(loop, 1000000, X)
multiplot(mh, 1, -1, title='Lorenz Attractor with 10 percent Exponential noise')

############################################

