import os
import numpy as np
import matplotlib.pyplot as plt

DIRPATH = os.path.dirname(os.path.realpath(__file__))+"\\"

def beta_pdf(xs, alpha=1, beta=1):
	"""
	The beta-binomial PDF for a given set of xs.

	Used in testing the MCMC code for correctness
	"""
	return np.array([x**(alpha-1) * (1-x)**(beta-1) for x in xs])/betafunc(alpha, beta)

def plot(mh, varname):
	"""
	Plot the posterior of a single variable from a MCMC object
	"""
	output = []
	for i in mh.path[1:]:
		output.append(i[varname])

	h = plt.hist(output, bins=50, alpha=0.3)
	plt.show(block=False)

def normal_lorenz_pdf(obs, pred):
	"""
	The likelihood function for our lorenz attractor - given the observations
	and the predictions, what is the likelihood?
	"""

	# This factor doesn't matter because we subtract the previous iteration anyways
	# We removed it to save computation
	#loglik = -1*len(obs)*np.log(10*np.sqrt(2*np.pi))

	loglik=0
	for i in xrange(len(obs)):
		p = pred[i][1]
		o = obs[i][1]
		# Normal loglik
		loglik -= np.sum([x for x in 50*((o/p)-1)**2 if not np.isnan(x)])

	return loglik/len(obs)

def gen_lik_graph(minval=0.05, maxval=0.15, n=1000, name='X', lf = lambda s,t: LorenzAttractor(np.array([s,0.,0.]), 10., 28., 8/3., tf=t).rk4_out):
	"""
	Generate the likelihood graph that can be seen in the writeup.
	"""

	# For t_f = 1
	xs0 = np.array([.1,0.,0.])
	X = RLA(xs0=xs0, sigma=10., rho=28., beta=8/3., tf=1.).get_all()

	xs = np.linspace(minval, maxval, n)
	ys = []
	prg = Progress(len(xs))
	for s in xs:
		Y = lf(s,1.)
		ys.append(normal_lorenz_pdf(X, Y))
		prg.increment(1)

	# For t_f = 10
	X2 = RLA(xs0=xs0, sigma=10., rho=28., beta=8/3., tf=10.).get_all()

	xs2 = np.linspace(minval, maxval, n)
	ys2 = []
	prg = Progress(len(xs))
	for s in xs2:
		Y2 = lf(s,10.)
		ys2.append(normal_lorenz_pdf(X2, Y2))
		prg.increment(1)

	# For t_f = 20
	X3 = RLA(xs0=xs0, sigma=10., rho=28., beta=8/3., tf=20.).get_all()

	xs3 = np.linspace(minval, maxval, n)
	ys3 = []
	prg = Progress(len(xs))
	for s in xs3:
		Y3 = lf(s,20.)
		ys3.append(normal_lorenz_pdf(X3, Y3))
		prg.increment(1)


	# Actual plotting. Hopefully things look pretty
	plt.plot(xs, [-1*np.log(-1*y) for y in ys])
	plt.plot(xs2, [-1*np.log(-1*y) for y in ys2])
	plt.plot(xs3, [-1*np.log(-1*y) for y in ys3])
	plt.legend(['tf=1', 'tf=10', 'tf=20'])
	plt.title("Likelihood Surface for Initial Parameter "+name)
	plt.ylabel("log(log(L))")
	plt.xlabel(name)
	plt.show(block=False)
	return [(xs, ys), (xs2, ys2), (xs3, ys3)]

def multiplot(mh, start=1, end=-1, varnames=['sigma', 'rho', 'beta'], actual=[10,28,8/3.], title=""):
	"""
	Plots several variables worth of graphs on a single page. Nice for saving xs and space.
	"""
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

		mu = np.array([np.median(l) for l in llvals])
		std = np.array([np.std(l) for l in llvals])
		ax = plt.subplot(3,3,i+1)
		plt.plot(h[1], mu)
		ax.fill_between(h[1], mu-std, mu+std, alpha=0.3, color='red')

		plt.subplot(3,3,i+2)
		plt.plot(output[v], 'r')
		plt.plot([actual[j] for val in output[v]], 'black')
		i += 3

	plt.show(block=False)