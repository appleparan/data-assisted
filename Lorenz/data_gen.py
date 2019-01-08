import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Lorenz(object):
	"""
		dx/dt = \sigma (y - x)
		dy/dt = x (\rho  - z) - y
		dz/dt = xy - \beta z

		dL[0]/dt = -\sigma*L[0]	+ \sigma*L[1]
		dL[1]/dt = 	\rho*L[0] 	- L[1] 						- L[0]*L[2]
		dL[2]/dt = 								-\beta*L[2]	+ L[0]*L{1]
	"""

	# for chaotic solution
	rho = 28.0
	sigma = 10.0
	beta = 8.0 / 3.0

	# linear part of the operator
	L = np.zeros((3, 3))
	# L
	# [ -sigma	sigma 	0 	]
	# [ rho		-1		0 	]
	# [ 0		0		beta]
	L[0, 0], L[1, 0] = -sigma, sigma
	L[0, 1], L[1, 1] = rho, -1.0
	L[2, 2] = -beta

	b = np.zeros((1, 3))

	@staticmethod
	def NL(x):

		assert len(x.shape) == 2, 'Input needs to be a two-dimensional array.'
		Nx = np.zeros(x.shape) # [*, 3]

		Nx[:, 1] = -x[:, 0] * x[:, 2]
		Nx[:, 2] = x[:, 0] * x[:, 1]

		return Nx

	def dynamics(x):

		assert len(x.shape) == 2, 'Input needs to be a two-dimensional array.'
		# dxdt = np.zeros(x.shape)

		# dxdt[:,0] = CdV.gamma_m_star[0]*x[:,2] - CdV.C*(x[:,0] - CdV.x1s)
		# dxdt[:,1] = -(CdV.alpha[0]*x[:,0] - CdV.beta[0])*x[:,2] - CdV.C*x[:,1] - CdV.delta[0]*x[:,3]*x[:,5]
		# dxdt[:,2] = (CdV.alpha[0]*x[:,0] - CdV.beta[0])*x[:,1] - CdV.gamma_m[0]*x[:,0] - CdV.C*x[:,2] + CdV.delta[0]*x[:,3]*x[:,4]
		# dxdt[:,3] = CdV.gamma_m_star[1]*x[:,5] - CdV.C*(x[:,3] - CdV.x4s) + CdV.epsilon*(x[:,1]*x[:,5] - x[:,2]*x[:,4])
		# dxdt[:,4] = -(CdV.alpha[1]*x[:,0] - CdV.beta[1])*x[:,5] - CdV.C*x[:,4] - CdV.delta[1]*x[:,2]*x[:,3]
		# dxdt[:,5] = (CdV.alpha[1]*x[:,0] - CdV.beta[1])*x[:,4] - CdV.gamma_m[1]*x[:,3] - CdV.C*x[:,5] + CdV.delta[1]*x[:,3]*x[:,1]
		dxdt = np.matmul(x, Lorenz.L) + Lorenz.NL(x)

		return dxdt


def main():

	#x0 = np.random.rand(1,6)
	x0 = np.array([[-8.0, 7.0, 27.0]])
	print(x0.shape)
	X = []
	# time
	dt,T = .01, 200.0
	#T = 300*dt

	# iteration from 0 to T, with step 1
	tt = np.arange(0,T,1)

	L = Lorenz.L
	# eigen value
	v,w = np.linalg.eig(L)

	print('eigenvalues:',v)
	Linv = np.linalg.inv(L)
	x_linsol = np.matmul(Lorenz.b, Linv)
	print(np.shape(x_linsol))
	x_nlinsol = Lorenz.NL(x_linsol)
	print('Linear solution:',x_linsol)
	print('NL of that:', x_nlinsol)

	# initial spin-up
	for t in np.arange(0,T,dt):
		dxdt = Lorenz.dynamics(x0)
		x0 = x0 + dt*dxdt
		X.append(np.squeeze(x0))

	## formal stepping
	#for t in range(T):
	#	for t2 in np.arange(0,1,dt):
	#		dxdt = Lorenz.dynamics(x0)
	#		x0 = x0 + dt*dxdt
	#	X.append(np.squeeze(x0))

	X = np.array(X)
	print(X.shape)

	## save data
	np.savez('data/lorenz_traj_pt100k_dt0.001.npz',X=X)

	## plot trajectory in (x1, x4) plane
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=0.5)
	ax.set_xlabel("X Axis")
	ax.set_ylabel("Y Axis")
	ax.set_zlabel("Z Axis")
	ax.set_title("Lorenz Attractor")

	#plt.scatter(X[:,0], X[:,3], s=5, marker='o', facecolor='k', edgecolor='none', )
	#plt.xlim([0.7, 1])
	#plt.ylim([-0.8, -0.1])

	# plt.show()
	plt.savefig("Lorenz_init_dynamics.png", format="png")


if __name__ == '__main__':
	main()
