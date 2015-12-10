p = read_csv("./input/input_properties.csv")
l = read_csv("./input/input_laser.csv")
c = read_csv("./input/input_control.csv")
s = read_csv("./input/input_sensor.csv")
g_eval = read_csv("./eval_data/eval_gps.csv")

# States: X, Y, v, alpha, psi, a
# State Evolution
# X -> X+

def predict_from_control(S, d, t, ascale=0., vscale=0.5, L=0.257717, a=0.299541, b=0.0500507):
	dt = t-S.time
	newV = vscale*(S.v + S.a)+(1.-vscale)*d[0] # Should change this to take earlier into account?
	newY = S.y + dt*newV*np.cos(S.psi) - newV/L * (a*np.sin(S.psi) + b*np.cos(S.psi))*np.tan(S.alpha)
	newX = S.x + dt*newV*np.sin(S.psi) + newV/L * (a*np.cos(S.psi) - b*np.sin(S.psi))*np.tan(S.alpha)
	newPsi = newV/L * np.tan(S.alpha)
	
	newAlpha = ascale*S.alpha+(1.-ascale)*d[1] # Should this take old alpha into account?
	newA = d[0]-S.v
	return State([newX, newY, newV, newAlpha, newPsi, newA], t)

def predict_from_deriv(S, d, t, ascale=0., vscale=0.5, L=0.257717, a=0.299541, b=0.0500507):
	dt = t-S.time
	nV = vscale*(S.v + S.a) + (1.-vscale)*d[0]
	dY = nV*np.cos(S.psi) - nV/L * (a*np.sin(S.psi) + b*np.cos(S.psi))*np.tan(S.alpha)
	dX = nV*np.sin(S.psi) + nV * (a*np.cos(S.psi) - b*np.sin(S.psi))*np.tan(S.alpha)/L
	dPsi = nV*np.tan(S.alpha)/L

	nA = ascale*S.alpha+(1.-ascale)*d[1]
	na = d[0]-S.v
	return State([S.x+dX*dt, S.y+dY*dt, nV, nA, S.psi+dt*dPsi, na], t)

def get_y(S, d, dt, L=0.257717, a=0.299541, b=0.0500507):
	na = (d[0]-S.v)/dt
	nV = d[0]
	nA = d[1]
	nPsi = nV*np.tan(nA)/L


def predict_from_laser(S, d, t):
	dt = t - S.time

def visualize_laser(l):
	xs = []
	ys = []
	for i in xrange(361):
		xs.append(l[i]*np.cos(np.pi*(i)/360.))
		ys.append(l[i]*np.sin(np.pi*(i)/360.))
	plt.plot(xs, ys)
	plt.show()

def control_cov(P, Q, dt):
	AP = [predict_control(State(P[:,i]), dt) for i in xrange(len(P[0,:]))]
	APA = np.transpose([predict_control(p, dt).stored for p in AP])
	return APA+Q

###############################################

class State:
	def __init__(self, X):
		self.stored = X
		self.x = X[0]
		self.y = X[1]
		self.psi = X[2]
		self.v = X[3]
		self.alpha = X[4]

	def __str__(self):
		return "State of Automobile:\n   X = %.4f \n   Y = %.4f \n   v = %.4f \n   alpha = %.4f \n   psi = %.4f " % (self.x, self.y, self.v, self.alpha, self.psi)

	def __repr__(self):
		return self.__str__()

def predict_control(S, dt, L=0.257717, a=0.299541, b=0.0500507):
	nV = S.v
	dY = nV*np.cos(S.psi) - nV/L * (a*np.sin(S.psi) + b*np.cos(S.psi))*np.tan(S.alpha)
	dX = nV*np.sin(S.psi) + nV * (a*np.cos(S.psi) - b*np.sin(S.psi))*np.tan(S.alpha)/L
	dPsi = nV*np.tan(S.alpha)/L

	nA = S.alpha
	return State([S.x+dX*dt, S.y+dY*dt, S.psi+dt*dPsi, nV, nA])

def gen_jacobian(S, dt, L=0.257717, a=0.299541, b=0.0500507):
	dX = [1, 0, dt*(S.v*np.cos(S.psi) - S.v*(a*np.sin(S.psi) + b*np.cos(S.psi))/L)*np.tan(S.alpha), \
	dt*(np.sin(S.psi) + np.tan(S.alpha)*(a*np.cos(S.psi) - b*np.sin(S.psi))/L), \
	S.v*(a*np.cos(S.psi)-b*np.sin(S.psi))/(L*(np.cos(S.psi)**2))]

	dY = [0, 1, -1*dt*(S.v*np.sin(S.psi) - S.v*(a*np.cos(S.psi) - b*np.sin(S.psi))/L)*np.tan(S.alpha), \
	dt*(np.cos(S.psi) - np.tan(S.alpha)*(a*np.sin(S.psi) + b*np.cos(S.psi))/L), \
	S.v*(a*np.sin(S.psi)+b*np.cos(S.psi))/(L*(np.cos(S.psi)**2))]

	dPsi = [0, 0, 1, np.tan(S.alpha)/L, S.v/(L*(np.cos(S.psi)**2))]

	dV = [0, 0, 0, 1, 0]

	dAlpha = [0, 0, 0, 0, 1]

	return np.array([dX, dY, dPsi, dV, dAlpha])

H =  np.array([[0.,0.,0.,1.,0.],
				[0.,0.,0.,0.,1.]])# measurement function
R = np.array([[0.1,0.],
				 [0.,0.1]])# measurement uncertainty
I = np.eye(5)
u = np.array([0., 0., 0., 0., 0.])
L=0.257717
a=0.299541
b=0.0500507

adata = AutoData(p,l,c)
# Improve with first step?
xs = [[adata.p['GPSLat'], adata.p['GPSLon'], 0., 0., adata.p['InitialAngle']]]
ts = [0]
Ps = [np.diag([100,100,100,100, 100])]
for d in adata.iteritems():
	# If it is a control point
	if len(d[1]) == 2:
		# Preprocessing
		dt = d[0]-ts[-1]
		ts.append(d[0])
		F = gen_jacobian(State(xs[-1]), dt)

		# prediction
		x = np.dot(F, xs[-1]) + u
		xtemp = predict_control(State(xs[-1]), dt)
		print x
		print xtemp.stored
		P = np.dot(F, np.dot(Ps[-1], np.transpose(F)))
		
		# measurement update
		Z = np.array(d[1])
		y = np.transpose(Z) - np.dot(H, x)
		S = np.dot(H, np.dot(P, np.transpose(H))) + R
		K = np.dot(P, np.dot(np.transpose(H), np.linalg.inv(S)))
		xs.append(x + np.dot(K, y))
		Ps.append(np.dot(I - np.dot(K, H), P))

	# If it is a laser point
	if len(d[1]) == 722:
		pass

xvals = [s[0] for s in xs]
yvals = [s[1] for s in xs]

plt.plot(g_eval[1][1], g_eval[1][2])
plt.plot(xvals,yvals)
plt.show(block=False)

#############################################################
 
H =  np.array([[0.,0.,1.,0.],
						 [0.,0.,0.,1.]])# measurement function
R = np.array([[0.1,0.],
				 [0.,0.1]])# measurement uncertainty
I = np.array([[1.,0.,0.,0.],
						 [0.,1.,0.,0.],
						 [0.,0.,1.,0.],
						 [0.,0.,0.,1.]])# identity matrix
u = np.array([0., 0., 0., 0.])
L=0.257717
a=0.299541
b=0.0500507

adata = AutoData(p,l,c)
# Improve with first step?
xs = [[adata.p['GPSLat'], adata.p['GPSLon'], 100.,1.]]
ts = [0]
Ps = [np.diag([100,100,100,100])]
for d in adata.iteritems():
	# If it is a control point
	if len(d[1]) == 2:
		# Preprocessing
		dt = d[0]-ts[-1]
		op = np.arctan(xs[-1][3]/xs[-1][2]) + dt*d[1][0]*np.tan(d[1][1])/L
		ts.append(d[0])
		vy = d[1][0]*np.cos(op) - (a*np.sin(op) + b*np.cos(op))*d[1][0]*np.tan(d[1][1])/L
		vx = d[1][0]*np.sin(op) + (a*np.cos(op) - b*np.sin(op))*d[1][0]*np.tan(d[1][1])/L

		F =  np.array([[1.,0.,dt,0.],
					[0.,1.,0.,dt],
					[0.,0.,1.,0.],
					[0.,0.,0.,1.]])# next state function

		# prediction
		x = np.dot(F, xs[-1]) + u
		P = np.dot(F, np.dot(Ps[-1], np.transpose(F)))
		
		# measurement update
		Z = np.array([vx, vy])
		y = np.transpose(Z) - np.dot(H, x)
		S = np.dot(H, np.dot(P, np.transpose(H))) + R
		K = np.dot(P, np.dot(np.transpose(H), np.linalg.inv(S)))
		xs.append(x + np.dot(K, y))
		Ps.append(np.dot(I - np.dot(K, H), P))

	# If it is a laser point
	if len(d[1]) == 722:
		pass

xvals = [s[0] for s in xs]
yvals = [s[1] for s in xs]

plt.plot(g_eval[1][1], g_eval[1][2])
plt.plot(xvals,yvals)
plt.show(block=False)


###############################################

class State:
	def __init__(self, X):
		self.stored = X
		self.x = X[0]
		self.y = X[1]
		self.psi = X[4]
		self.v = X[2]
		self.alpha = X[3]

	def __str__(self):
		return "State of Automobile:\n   X = %.4f \n   Y = %.4f \n   v = %.4f \n   alpha = %.4f \n   psi = %.4f " % (self.x, self.y, self.v, self.alpha, self.psi)

	def __repr__(self):
		return self.__str__()

def predict_control(S, dt, L=0.257717, a=0.299541, b=0.0500507):
	nV = S.v
	dY = nV*np.cos(S.psi) - nV/L * (a*np.sin(S.psi) + b*np.cos(S.psi))*np.tan(S.alpha)
	dX = nV*np.sin(S.psi) + nV * (a*np.cos(S.psi) - b*np.sin(S.psi))*np.tan(S.alpha)/L
	dPsi = nV*np.tan(S.alpha)/L

	nA = S.alpha
	return State([S.x+dX*dt, S.y+dY*dt, S.psi+dt*dPsi, nV, nA])

def gen_jacobian(S, dt, L=0.257717, a=0.299541, b=0.0500507):
	dX = [1, 0, dt*(S.v*np.cos(S.psi) - S.v*(a*np.sin(S.psi) + b*np.cos(S.psi))/L)*np.tan(S.alpha), \
	dt*(np.sin(S.psi) + np.tan(S.alpha)*(a*np.cos(S.psi) - b*np.sin(S.psi))/L), \
	S.v*(a*np.cos(S.psi)-b*np.sin(S.psi))/(L*(np.cos(S.psi)**2))]

	dY = [0, 1, -1*dt*(S.v*np.sin(S.psi) - S.v*(a*np.cos(S.psi) - b*np.sin(S.psi))/L)*np.tan(S.alpha), \
	dt*(np.cos(S.psi) - np.tan(S.alpha)*(a*np.sin(S.psi) + b*np.cos(S.psi))/L), \
	S.v*(a*np.sin(S.psi)+b*np.cos(S.psi))/(L*(np.cos(S.psi)**2))]

	dPsi = [0, 0, 1, np.tan(S.alpha)/L, S.v/(L*(np.cos(S.psi)**2))]

	dV = [0, 0, 0, 1, 0]

	dAlpha = [0, 0, 0, 0, 1]

	return np.array([dX, dY, dPsi, dV, dAlpha])


#Q = np.diag([0.2,0.2,0.2,0.2,0.2])
Q = np.diag([1]*5)
#R = np.diag([0.2,0.2,0.2,0.2,0.2])/2.
R = np.diag([1]*5)

x = AutoData(p,l,c)
# Improve with first step?
S = [(State([x.p['GPSLat'], x.p['GPSLon'], 0., x.p['InitialAngle'], 0.,0.]),0)]
Ps = [np.diag([100,100,100,100,100])]
for d in x.iteritems():
	# If it is a control point
	if len(d[1]) == 2:
		prev_state = S[-1][0]
		dt = d[0]-S[-1][1]
		F = gen_jacobian(prev_state, dt)
		Xka = predict_control(prev_state, dt)
		P_pred = np.dot(F, np.dot(Ps[-1], np.transpose(F))) + Q
		W = np.dot(P_pred, np.linalg.inv(P_pred+R))
		obs_state = predict_control(State([prev_state.x, prev_state.y, prev_state.psi, d[1][0], d[1][1]]), dt)
		dX = np.dot(W,np.array(obs_state.stored)-np.array(Xka.stored))
		S.append((State(Xka.stored + dX), d[0]))
		Ps.append(P_pred - np.dot(W, np.dot(P_pred+R, np.transpose(W))))

	# If it is a laser point
	if len(d[1]) == 722:
		pass

xs = [s[0].x for s in S]
ys = [s[0].y for s in S]

plt.plot(g_eval[1][1], g_eval[1][2])
plt.plot(xs,ys)
plt.show(block=False)



#################################################
##### ORIGINAL FROM THE BOOK MODEL ##############

class State:
	def __init__(self, X):
		self.stored = X
		self.x = X[0]
		self.y = X[1]
		self.psi = X[4]
		self.v = X[2]
		self.alpha = X[3]

	def __str__(self):
		return "State of Automobile:\n   X = %.4f \n   Y = %.4f \n   v = %.4f \n   alpha = %.4f \n   psi = %.4f " % (self.x, self.y, self.v, self.alpha, self.psi)

	def __repr__(self):
		return self.__str__()

def predict_control(S, dt, L=0.257717, a=0.299541, b=0.0500507):
	nV = S.v
	dY = nV*np.cos(S.psi) - nV/L * (a*np.sin(S.psi) + b*np.cos(S.psi))*np.tan(S.alpha)
	dX = nV*np.sin(S.psi) + nV * (a*np.cos(S.psi) - b*np.sin(S.psi))*np.tan(S.alpha)/L
	dPsi = nV*np.tan(S.alpha)/L

	nA = S.alpha
	return State([S.x+dX*dt, S.y+dY*dt, S.psi+dt*dPsi, nV, nA])

def control_cov(P, Q, dt):
	AP = [predict_control(State(P[:,i]), dt) for i in xrange(len(P[0,:]))]
	APA = np.transpose([predict_control(p, dt).stored for p in AP])
	return APA+Q

def gen_jacobian(S, dt, L=0.257717, a=0.299541, b=0.0500507):
	dX = [1, 0, dt*(S.v*np.cos(S.psi) - S.v*(a*np.sin(S.psi) + b*np.cos(S.psi))/L)*np.tan(S.alpha), \
	dt*(np.sin(S.psi) + np.tan(S.alpha)*(a*np.cos(S.psi) - b*np.sin(S.psi))/L), \
	S.v*(a*np.cos(S.psi)-b*np.sin(S.psi))/(L*(np.cos(S.psi)**2))]

	dY = [0, 1, -1*dt*(S.v*np.sin(S.psi) - S.v*(a*np.cos(S.psi) - b*np.sin(S.psi))/L)*np.tan(S.alpha), \
	dt*(np.cos(S.psi) - np.tan(S.alpha)*(a*np.sin(S.psi) + b*np.cos(S.psi))/L), \
	S.v*(a*np.sin(S.psi)+b*np.cos(S.psi))/(L*(np.cos(S.psi)**2))]

	dPsi = [0, 0, 1, np.tan(S.alpha)/L, S.v/(L*(np.cos(S.psi)**2))]

	dV = [0, 0, 0, 1, 0]

	dAlpha = [0, 0, 0, 0, 1]

	return np.array([dX, dY, dPsi, dV, dAlpha])


Q = np.diag([0.2,0.2,0.2,0.2,0.2])
R = np.diag([0.2,0.2,0.2,0.2,0.2])

x = AutoData(p,l,c)
# Improve with first step?
S = [(State([x.p['GPSLat'], x.p['GPSLon'], 0., x.p['InitialAngle'], 0.,0.]),0)]
Ps = [np.diag([1,1,1,1,1])]
for d in x.iteritems():
	# If it is a control point
	if len(d[1]) == 2:
		prev_state = S[-1][0]
		dt = d[0]-S[-1][1]
		F = gen_jacobian(prev_state, dt)
		Xka = predict_control(prev_state, dt)
		P_pred = np.dot(F, np.dot(Ps[-1], np.transpose(F))) + Q
		W = np.dot(P_pred, np.linalg.inv(P_pred+R))
		obs_state = State([prev_state.x, prev_state.y, prev_state.psi, d[1][0], d[1][1]])
		dX = np.dot(W,np.array(obs_state.stored)-np.array(Xka.stored))
		S.append((State(Xka.stored + dX), d[0]))
		Ps.append(P_pred - np.dot(W, np.dot(P_pred+R, np.transpose(W))))

	# If it is a laser point
	if len(d[1]) == 722:
		pass

xs = [s[0].x for s in S]
ys = [s[0].y for s in S]

plt.plot(g_eval[1][1], g_eval[1][2])
plt.plot(xs,ys)
plt.show(block=False)

###########################
#### FAILED XY REVERSE MODEL #

class State:
	def __init__(self, X):
		self.stored = X
		self.x = X[0]
		self.y = X[1]
		self.psi = X[4]
		self.v = X[2]
		self.alpha = X[3]

	def __str__(self):
		return "State of Automobile:\n   X = %.4f \n   Y = %.4f \n   v = %.4f \n   alpha = %.4f \n   psi = %.4f " % (self.x, self.y, self.v, self.alpha, self.psi)

	def __repr__(self):
		return self.__str__()

def predict_control(S, dt, L=0.257717, a=0.299541, b=0.0500507):
	nV = S.v
	dX = nV*np.cos(S.psi) - nV/L * (a*np.sin(S.psi) + b*np.cos(S.psi))*np.tan(S.alpha)
	dY = nV*np.sin(S.psi) + nV * (a*np.cos(S.psi) - b*np.sin(S.psi))*np.tan(S.alpha)/L
	dPsi = nV*np.tan(S.alpha)/L

	nA = S.alpha
	return State([S.x+dX*dt, S.y+dY*dt, S.psi+dt*dPsi, nV, nA])

def gen_jacobian(S, dt, L=0.257717, a=0.299541, b=0.0500507):
	dY = [0, 1, dt*(S.v*np.cos(S.psi) - S.v*(a*np.sin(S.psi) + b*np.cos(S.psi))/L)*np.tan(S.alpha), \
	dt*(np.sin(S.psi) + np.tan(S.alpha)*(a*np.cos(S.psi) - b*np.sin(S.psi))/L), \
	S.v*(a*np.cos(S.psi)-b*np.sin(S.psi))/(L*(np.cos(S.psi)**2))]

	dX = [1, 0, -1*dt*(S.v*np.sin(S.psi) - S.v*(a*np.cos(S.psi) - b*np.sin(S.psi))/L)*np.tan(S.alpha), \
	dt*(np.cos(S.psi) - np.tan(S.alpha)*(a*np.sin(S.psi) + b*np.cos(S.psi))/L), \
	S.v*(a*np.sin(S.psi)+b*np.cos(S.psi))/(L*(np.cos(S.psi)**2))]

	dPsi = [0, 0, 1, np.tan(S.alpha)/L, S.v/(L*(np.cos(S.psi)**2))]

	dV = [0, 0, 0, 1, 0]

	dAlpha = [0, 0, 0, 0, 1]

	return np.array([dX, dY, dPsi, dV, dAlpha])


Q = np.diag([0.2,0.2,0.2,0.2,0.2])
R = np.diag([0.2,0.2,0.2,0.2,0.2])

x = AutoData(p,l,c)
# Improve with first step?
S = [(State([x.p['GPSLat'], x.p['GPSLon'], 0., x.p['InitialAngle'], 0.,0.]),0)]
Ps = [np.diag([1,1,1,1,1])]
for d in x.iteritems():
	# If it is a control point
	if len(d[1]) == 2:
		prev_state = S[-1][0]
		dt = d[0]-S[-1][1]
		F = gen_jacobian(prev_state, dt)
		Xka = predict_control(prev_state, dt)
		P_pred = np.dot(F, np.dot(Ps[-1], np.transpose(F))) + Q
		W = np.dot(P_pred, np.linalg.inv(P_pred+R))
		obs_state = State([prev_state.x, prev_state.y, prev_state.psi, d[1][0], d[1][1]])
		dX = np.dot(W,np.array(obs_state.stored)-np.array(Xka.stored))
		S.append((State(Xka.stored + dX), d[0]))
		Ps.append(P_pred - np.dot(W, np.dot(P_pred+R, np.transpose(W))))

	# If it is a laser point
	if len(d[1]) == 722:
		pass

xs = [s[0].x for s in S]
ys = [s[0].y for s in S]

plt.plot(g_eval[1][1], g_eval[1][2])
plt.plot(xs,ys)
plt.show(block=False)


############################
#### OLD CRAPPY MODEL ######


Q = np.diag([0.2,0.2,0.2,0.2,0.2])
R = np.diag([0.2,0.2,0.2,0.2,0.2])

x = AutoData(p,l,c)
# Improve with first step?
S = [(State([x.p['GPSLat'], x.p['GPSLon'], 0., x.p['InitialAngle'], 0.,0.]),0)]
Ps = [np.diag([1,1,1,1,1])]
for d in x.iteritems():
	# If it is a control point
	if len(d[1]) == 2:
		#prev_state = S[-1][0]
		#dt = d[0]-S[-1][1]
		#F = gen_jacobian(prev_state, dt)
		#Xka = predict_control(prev_state, dt)
		#P_pred = np.dot(F, np.dot(Ps[-1], np.transpose(F))) + Q
		#W = P_pred*np.linalg.inv(P_pred+R)

		
		prev_state = S[-1][0]
		dt = d[0]-S[-1][1]
		A = gen_jacobian(prev_state, dt)
		Xka = predict_control(prev_state, dt)
		P_pred = np.dot(A, np.dot(Ps[-1], np.transpose(A))) + Q
		obs_state = State([prev_state.x, prev_state.y, prev_state.psi, d[1][0], d[1][1]])
		J = gen_jacobian(obs_state, dt)
		P_obs = np.dot(J, np.dot(P_pred, np.transpose(J))) + R
		K = np.dot(np.dot(P_pred, J), np.linalg.inv(P_obs))
		Ps.append(np.dot(np.eye(5)-np.dot(K, J), P_pred))
		S.append((State(Xka.stored+np.dot(K, -1*np.array(Xka.stored) + np.array(obs_state.stored))), d[0]))


	# If it is a laser point
	if len(d[1]) == 722:
		pass

xs = [s[0].x for s in S]
ys = [s[0].y for s in S]

plt.plot(g_eval[1][1], g_eval[1][2])
plt.plot(xs,ys)
plt.show(block=False)