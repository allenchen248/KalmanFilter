import numpy as np
import scipy as sp
from scipy.special import beta as betafunc
import matplotlib.pyplot as plt

from KalmanFilter.mcmc import *

def beta_pdf(xs, alpha=1, beta=1):
	return np.array([x**(alpha-1) * (1-x)**(beta-1) for x in xs])/betafunc(alpha, beta)

def plot(pd):
	plt.plot(pd)
	plt.show(block=False)

mh = MCMC()

n = 100000
pdist = np.zeros((n))
for j in xrange(n):
	p = mh.uniform("Prob", temperature=0.2)
	#x = [mh.bernoulli(i, p) for i in xrange(10)]
	#obs_p = np.sum(x)/10.
	obs_p = p
	mh.condition(np.log(obs_p)*9 + np.log(1-obs_p))
	#mh.condition(p*10-9)
	pdist[j] = p

output = []
for i in mh.path[1:]:
	output.append(i['Prob'].value)

h = plt.hist(output, bins=50, alpha=0.3)

xs = np.linspace(0,1,n)
plt.plot(xs, 0.2*np.max(h[0])*beta_pdf(xs, 10, 2))

plt.show()

class Tester:
	def __init__(self, a, b):
		self.a = a
		self.b = b

	def get_a(self):
		return self.a

	def put_a(self, a):
		self.a = a