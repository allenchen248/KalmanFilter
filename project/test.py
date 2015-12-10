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

class AutoData:
	def __init__(self, p, l, c):
		self.p = {p[0][i]:p[1][i][0] for i in xrange(len(p[1]))}
		#self.l = l
		#self.c = c

		vstime = zip(*c[1])
		vstime.extend(zip(*l[1]))

		self.data = sorted(vstime, key=lambda x:x[0])
		self.curind = 0

	def __getitem__(self, val):
		return self.data[val]

	def iteritems(self):
		return ([d[0],d[1:]] for d in self.data)

def loop(mh, X, s):
	xs0 = np.array([mh.uniform("x",0,.2), mh.uniform("y",-.1,.1), mh.uniform("z",-.1,.1)])
	#xs0 = np.array([.1,0,0])

	sigma = mh.uniform("sigma",9,11)
	#sigma = 10.

	rho = mh.uniform("rho",27,29)
	#rho = 28.

	beta = mh.uniform("beta",7/3.,10/3.)
	#beta = 8/3.

	L = LorenzAttractor(xs0, sigma, rho, beta, tf=s).rk4_out
	mh.condition(normal_lorenz_pdf(X, L))

sigmas = np.linspace(1,20,40)
mh_res = []
for s in sigmas:
	mh = MCMC()
	X = RLA(tf=s).get_all()

	t = mh.run(loop, 10, X, s)
	#multiplot(mh, 1, -1, title='Lorenz Attractor with %.2f percent noise' % s)
	mh_res.append(mh)
	mh = None

rs4 = np.array([len(r.path) for r in mh_res])
plt.plot(sigmas, (10*(rs-1)+rs2+10*(rs3-1)+10*(rs4-1))/3.)
plt.show(block=False)

rs2 = np.array([len(r.path) for r in mh_res])

