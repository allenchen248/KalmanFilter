import numpy as np

###########################################
mh = MCMC()
X = RLA().get_all()

n = 10000
for i in xrange(n):
	xs0 = np.array([mh.uniform("x",-1,1), mh.uniform("y",-1,1), mh.uniform("z",-1,1)])
	sigma = mh.uniform("sigma",5,15)
	rho = mh.uniform("rho",14,42)
	beta = mh.uniform("beta",4/3.,12/3.)
	L = LorenzAttractor(xs0, sigma, rho, beta).rk4_out
	mh.condition(normal_lorenz_pdf(X, L))

output = []
for i in mh.path[1:]:
	output.append(i['sigma'].value)

h = plt.hist(output, bins=50, alpha=0.3)
plt.show()

###########################################
mh = MCMC()

n = 100000
pdist = np.zeros((n))
for j in xrange(n):
	p = mh.uniform("Prob", temperature=0.1)
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