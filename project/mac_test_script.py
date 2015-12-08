import numpy as np
import scipy as sp
from scipy.special import beta as betafunc
import matplotlib.pyplot as plt

import warnings

from project.mcmc import *
from project.lorenz import *

warnings.filterwarnings("ignore")

def beta_pdf(xs, alpha=1, beta=1):
	return np.array([x**(alpha-1) * (1-x)**(beta-1) for x in xs])/betafunc(alpha, beta)

def plot(mh, varname):
	output = []
	for i in mh.path[1:]:
		output.append(i[varname])

	h = plt.hist(output, bins=50, alpha=0.3)
	plt.show(block=False)

def multiplot(mh, start=1, end=-1, varnames=['sigma', 'rho', 'beta'], actual=[10,28,8/3.], title=""):
	output = {v:[] for v in varnames}
	for i in mh.path[start:end]:
		for v in varnames:
			output[v].append(i[v])

	i = 1
	plt.figure(figsize=(15,10))
	plt.title(title)
	for j,v in enumerate(varnames):
		ax = plt.subplot(3,3,i)
		h = plt.hist(output[v], bins=50, alpha=0.3)
		ax.set_ylabel(v)

		bin_vals = np.digitize(output[v], h[1])-1
		llvals = [[] for k in xrange(max(bin_vals)+1)]
		for k,va in enumerate(mh.lls[start:end]):
			llvals[bin_vals[k]].append(va)

		mu = np.array([np.mean(l) for l in llvals])
		std = np.array([np.std(l) for l in llvals])
		ax = plt.subplot(3,3,i+1)
		plt.plot(h[1], mu)
		ax.fill_between(h[1], mu-std, mu+std, alpha=0.3, color='red')

		plt.subplot(3,3,i+2)
		plt.plot(output[v], 'r')
		plt.plot([actual[j] for val in output[v]], 'black')
		i += 3

	plt.show(block=False)


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
		loglik -= np.sum([x for x in ((o/p)-1)**2 if not np.isnan(x)])

	return loglik/len(obs)

############################################

def loop(mh, X):
	#xs0 = np.array([mh.uniform("x",0,.2), mh.uniform("y",-.1,.1), mh.uniform("z",-.1,.1)])
	xs0 = np.array([.1,0,0])

	sigma = mh.uniform("sigma",9,11)
	#sigma = 10.

	rho = mh.uniform("rho",27,29)
	#rho = 28.

	beta = mh.uniform("beta",7/3.,10/3.)
	#beta = 8/3.

	L = LorenzAttractor(xs0, sigma, rho, beta, tf=1.).rk4_out
	mh.condition(expo_lorenz_pdf(X, L))

mh = MCMC()
X = RLA(tf=1., rf=random_expo(1,10)).get_all()

t = mh.run(loop, 100000, X)
multiplot(mh, 1, -1, title='Lorenz Attractor with 10 percent Exponential noise')
