import numpy as np
import scipy as sp
from scipy.special import beta as betafunc
import matplotlib.pyplot as plt

import warnings

from util.progress import Progress

from project import *
from project.mcmc import *
from project.lorenz import *

warnings.filterwarnings("ignore")

# KL Divergence
from scipy.stats import entropy

# Likelihood graphs
gen_lik_graph(27.,29., name='Rho', lf = lambda s,t: LorenzAttractor(np.array([0.1,0.,0.]), 10., s, 8/3., tf=t).rk4_out)
gen_lik_graph(-0.05, 0.05, name='Z', lf = lambda s,t: LorenzAttractor(np.array([0.1,0.,s]), 10., 28., 8/3., tf=t).rk4_out)

############################################
##### Generates with different priors ######
def loop(mh, X, i):
	xs0 = np.array([mh.uniform("x",0,.2), mh.uniform("y",-.1,.1), mh.uniform("z",-.1,.1)])
	#xs0 = np.array([.1,0,0])

	sigma = mh.uniform("sigma",10*(1.-i),10*(1.+i))
	#sigma = 10.

	rho = mh.uniform("rho",28.*(1.-i),28.*(1.+i))
	#rho = 28.

	beta = mh.uniform("beta",8*(1.-i)/3.,8*(1.+i)/3.)
	#beta = 8/3.

	L = LorenzAttractor(xs0, sigma, rho, beta, tf=10.).rk4_out
	mh.condition(normal_lorenz_pdf(X, L))

ivals = [0.01, 0.1, 0.2, 0.5, 0.75, 1.]
mh_res = []
for s in ivals:
	mh = MCMC()
	X = RLA(tf=10.).get_all()

	t = mh.run(loop, 100000, X, s)
	#multiplot(mh, 1, -1, title='Lorenz Attractor with %.2f percent noise' % s)
	mh_res.append(mh)
	mh = None

#############################################
##### Generates with different variances ####

def loop(mh, X):
	xs0 = np.array([mh.uniform("x",0,.2), mh.uniform("y",-.1,.1), mh.uniform("z",-.1,.1)])
	#xs0 = np.array([.1,0,0])

	sigma = mh.uniform("sigma",9,11)
	#sigma = 10.

	rho = mh.uniform("rho",27,29)
	#rho = 28.

	beta = mh.uniform("beta",7/3.,10/3.)
	#beta = 8/3.

	L = LorenzAttractor(xs0, sigma, rho, beta, tf=10.).rk4_out
	mh.condition(normal_lorenz_pdf(X, L))

sigmas = [0.01,0.1,1.,10.,20.,50.,75.,100.]
mh_res = []
for s in sigmas:
	mh = MCMC()
	X = RLA(tf=10., rf=random_norm_percent(0,s)).get_all()

	t = mh.run(loop, 100000, X)
	#multiplot(mh, 1, -1, title='Lorenz Attractor with %.2f percent noise' % s)
	mh_res.append(mh)
	mh = None

#############################################
#### Generates with tf=20 ##################

def loop(mh, X):
	xs0 = np.array([mh.uniform("x",0,.2), mh.uniform("y",-.1,.1), mh.uniform("z",-.1,.1)])
	#xs0 = np.array([.1,0,0])

	sigma = mh.uniform("sigma",9,11)
	#sigma = 10.

	rho = mh.uniform("rho",27,29)
	#rho = 28.

	beta = mh.uniform("beta",7/3.,10/3.)
	#beta = 8/3.

	L = LorenzAttractor(xs0, sigma, rho, beta, tf=20.).rk4_out
	mh.condition(normal_lorenz_pdf(X, L))

mh = MCMC()
X = RLA(tf=20.).get_all()

t = mh.run(loop, 30000, X)
multiplot(mh, 1, -1, title='Lorenz Attractor with %.2f percent noise' % s)
mh_res.append(mh)
mh = None

############################################
#### Generates standard MCMC ###############

mh = MCMC()
X = RLA(tf=10.).get_all()
#X = LorenzAttractor(tf=1.).rk4_out

def loop(mh, X):
	#xs0 = np.array([mh.uniform("x",0,.2), mh.uniform("y",-.1,.1), mh.uniform("z",-.1,.1)])
	xs0 = np.array([.1,0,0])

	sigma = mh.uniform("sigma",9,11)
	#sigma = 10.

	rho = mh.uniform("rho",27,29)
	#rho = 28.

	beta = mh.uniform("beta",7/3.,10/3.)
	#beta = 8/3.

	L = LorenzAttractor(xs0, sigma, rho, beta, tf=10.).rk4_out
	mh.condition(normal_lorenz_pdf(X, L))

temp = mh.run(loop, 100000, X)
multiplot(mh,1,-1)