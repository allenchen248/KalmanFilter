import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def lorentz_func(t, xs, sigma, rho, beta, **kwargs):
	"""
	Given [x, y, z] and parameters, returns [dx/dt, dy/dt, dz/dt]
	with the given parameters.
	"""
	return np.array([sigma*(xs[1]-xs[0]), xs[0]*(rho-xs[2])-xs[1], xs[0]*xs[1] - beta*xs[2]])

def keep_run(t, xs, tf=1, **kwargs):
	return t < tf

def rk4(h, xs0=[1,1,1], keep_running=keep_run, f=lorentz_func, **kwargs):
	"""
	Implements the classical fourth-order Runge-Kutta method,
	and stops at end condition keep_running.

	Evaluates the function y' = f() [f is the argument]
	"""
	output = [(0,np.array([float(x) for x in xs0]))]
	h = float(h)
	t = 0
	while keep_running(t, output[-1][1], **kwargs):
		xs = output[-1][1]
		k1 = f(t, xs, **kwargs)
		k2 = f(t+h/2, xs+h*k1/2, **kwargs)
		k3 = f(t+h/2, xs+h*k2/2, **kwargs)
		k4 = f(t+h, xs+h*k3, **kwargs)
		t += h
		output.append((t, xs+h*(k1+2*k2+2*k3+k4)/6))

	return output

def plot_lorentz(output):
	"""
	Plots the lorentz function output!
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	coords = zip(*[o[1] for o in output])
	ax.plot(coords[0], coords[1], coords[2])
	plt.show(block=False)

class LorenzAttractor:
	def __init__(self, xs0=[.1,0.,0.], sigma=10., rho=28., beta=8/3., tf=50., 
			h=0.005, f=lorentz_func):
		self.rk4_out = rk4(h, xs0, sigma=float(sigma), rho=float(rho), beta=float(beta), tf=float(tf), f=f)
		self.f = f
		self.tf = tf
		self.la_params = {'sigma':sigma, 'rho':rho, 'beta':beta}
		self.ts = [o[0] for o in self.rk4_out]
		self.xs = {o[0]:o[1] for o in self.rk4_out}

	def __getitem__(self, tnew):
		"""
		If xs at that time doesn't exist, run a step of RK4 in order to
		generate what the x value should be there.
		"""
		if (tnew > self.tf) or (tnew <= 0):
			raise ValueError("Time Isn't Within Bounds")

		if tnew in self.xs:
			return self.xs[tnew]

		told = self.ts[np.searchsorted(self.ts, tnew)]
		h = float(tnew-told)
		x = self.xs[told]
		k1 = self.f(told, x, **self.la_params)
		k2 = self.f(told+h/2, x+h*k1/2, **self.la_params)
		k3 = self.f(told+h/2, x+h*k2/2, **self.la_params)
		k4 = self.f(told+h, x+h*k3, **self.la_params)
		return x+h*(k1+2*k2+2*k3+k4)/6

	def plot(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		coords = zip(*[self.xs[t] for t in self.ts])
		ax.plot(coords[0], coords[1], coords[2])
		plt.show(block=False)

def random_norm_percent(mu, sigma):
	def output(x):
		return (x*(1.+np.random.normal(mu, sigma, size=(1,3))/100.))[0]

	return output

class RLA:
	def __init__(self, LA=None, rf=random_norm_percent(0,10), **kwargs):
		if LA is None:
			LA = LorenzAttractor(**kwargs)

		self.la = LA
		self.rf = rf
		self.vals = {}

	def __getitem__(self, val):
		return self.rf(self.la[val])

	def plot(self):
		return self.la.plot()

	def get_all(self):
		all_vals = []
		for t,v in self.la.xs.iteritems():
			all_vals.append((t,self.rf(v)))
		return sorted(all_vals, key=lambda x:x[0])