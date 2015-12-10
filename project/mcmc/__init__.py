import numpy as np
import numpy.random as random

from project.progress import Progress

class Uniform:
	"""
	Uniform RV Object
	"""
	def __init__(self, a, b, temperature=0.1):
		if b <= a:
			raise ValueError("Can't have a uniform with lower bound higher than upper bound.")
		self.a = a
		self.b = b
		self.temp = temperature*(b-a)
		self.value = random.uniform(self.a, self.b)

	def __str__(self):
		return "Unif(%.2f, %.2f) with value %.4f" % (self.a, self.b, self.value)

	def __repr__(self):
		return self.__str__() 

	def generate(self):
		return self.value

	def resample(self):
		# Do not store in self.value because that involves a pointer dereference
		newval = random.triangular(self.value-self.temp, self.value, self.value+self.temp)
		#while (newval > self.b) or (newval < self.a):
		#	newval = random.triangular(self.value-self.temp, self.value, self.value+self.temp)

		self.value = newval
		return newval

	def prior_loglik(self):
		return -1*np.log(self.b-self.a)

class Bernoulli:
	"""
	Bernoulli RV Object
	"""
	def __init__(self, p):
		self.p = p
		self.value = int(random.uniform(0,1) < self.p)

	def __str__(self):
		return "Bern(%.4f) with value %d" % (self.p, self.value)

	def __repr__(self):
		return self.__str__()

	def resample(self):
		#self.value = int(random.uniform(0,1) < self.p)
		self.value = 1-self.value
		return self.value

	def prior_loglik(self):
		if self.value == 1:
			return np.log(self.p)
		else:
			return np.log(1-self.p)

class Normal:
	"""
	Normal RV Object
	"""
	def __init__(self, mu, sigma, temperature=0.1):
		if sigma <= 0:
			raise ValueError("Can't have a gaussian with standard deviation < 0")

		self.mu = mu
		self.sigma = sigma
		self.temp = temperature*sigma
		self.value = random.normal(mu, sigma)

	def __str__(self):
		return "N(%.2f, %.2f) with value %.4f" % (self.mu, self.sigma, self.value)

	def __repr__(self):
		return self.__str__()

	def generate(self):
		return self.value

	def resample(self):
		self.value = random.normal(self.value, self.temp)
		return self.value

	def prior_loglik(self):
		return -1*((self.value - self.mu)**2/(2*self.sigma**2) + np.log(self.sigma * np.sqrt(2*np.pi)))

class MCMC:
	"""
	Implements MCMC. Call MCMC.run() and make sure to condition within run().

	See test_script.py for more information.
	"""
	def __init__(self):
		self.rvs = {}
		self.prev = {}
		self.prev_loglik = None
		self.changed = None
		self.path = []
		self.lls = []

	def generate_rv(self, name, rv_func):
		"""
		Internal method for generating a random variable generally
		"""
		# Can't handle changing numbers of vars yet
		if name not in self.prev:
			self.rvs[name] = rv_func()
		else:
			self.rvs[name] = self.prev[name]

		#self.rvs[name] = rv_func()
		return self.rvs[name].value

	def uniform(self, name, a=0, b=1, **kwargs):
		"""
		External-facing function for a uniform distribution
		"""
		return self.generate_rv(name, lambda: Uniform(a, b, **kwargs))

	def normal(self, name, mu=0, sigma=1, **kwargs):
		"""
		External-facing function for a normal distribution
		"""
		return self.generate_rv(name, lambda: Normal(mu, sigma, **kwargs))

	def bernoulli(self, name, p=0.5, **kwargs):
		"""
		External-facing function for a bernoulli distribution
		"""
		return self.generate_rv(name, lambda: Bernoulli(p, **kwargs))

	def accept(self, val):
		"""
		Accept this random sample
		"""
		self.path.append({k:v.value for k,v in self.prev.iteritems()})
		self.lls.append(val)
		self.prev = self.rvs
		self.resample()

		self.rvs = {}

		self.prev_loglik = val

	def reject(self):
		"""
		Reject this sample
		"""
		self.prev[self.changed[0]].value = self.changed[1]
		self.resample()
		
		self.rvs = {}

	def resample(self):
		"""
		Performs the resampling for a given set of RVs
		"""
		# Improve the speed of this
		name = self.prev.keys()[random.random_integers(0, len(self.prev)-1)]
		val = self.prev[name].value
		self.prev[name].resample()
		self.changed = (name, val)

	def get_prior(self):
		"""
		Get the prior log likelihoods for each of the RVs in the problem
		"""
		return np.sum([rv.prior_loglik() for name,rv in self.rvs.iteritems() if self.active[name]])

	def condition(self, loglik_val):
		"""
		Perform a conditioning step.
		"""
		if self.prev_loglik is None:
			return self.accept(loglik_val)

		prior_ratio = np.sum([rv.prior_loglik() for rv in self.rvs.itervalues()]) - np.sum([rv.prior_loglik() for rv in self.prev.itervalues()])
		cur_ratio = loglik_val - self.prev_loglik

		if prior_ratio+cur_ratio > 1:
			accept_ratio = 1
		else:
			accept_ratio = min(1, np.exp(prior_ratio + cur_ratio))

		if random.uniform(0,1) < accept_ratio:
			# For bug fixing
			#print accept_ratio, prior_ratio, cur_ratio
			#print "ACCEPT"
			return self.accept(loglik_val)

		#print "REJECT"
		return self.reject()

	def run(self, f, n, *args, **kwargs):
		"""
		Run this with a function to actually run MCMC.
		"""
		prg = Progress(n)
		output = []
		for i in xrange(n):
			output.append(f(self, *args, **kwargs))
			prg.increment(1, "Accepted: "+str(round(100.*len(self.path)/float(i+1), 2))+"%    ")
		prg.finish()

		return output