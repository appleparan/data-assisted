import numpy as np
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

"""
Calculate error between a trajectory prediction and truth

"""

class traj_err(object):

	@staticmethod
	def corr_2D(a1, a2):	
		# calculates correlation between columns of 2D matrices a1, a2
  		assert a1.shape == a2.shape, 'input shape mismatch!'
  		c = np.zeros(a1.shape[1])	
  		for col in range(a1.shape[1]):
  			c[col] = pearsonr(a1[:,col],a2[:,col])[0]
  		return c

	def corr_with_time(X1, X2):	
		""" 
		calculates prediction-truth correlation at each time
  		X1,X2 are 3D matrices with dimension [sample,time,feature] 		
	
  		"""
		assert X1.shape == X2.shape, 'input shape mismatch!'
		X1_split, X2_split = np.split(X1,X1.shape[1],axis=1), np.split(X2,X2.shape[1],axis=1)
  		
		X1_list = [np.squeeze(v) for v in X1_split]
		X2_list = [np.squeeze(v) for v in X2_split]

		r = [traj_err.corr_2D(u,v) for (u,v) in zip(X1_list,X2_list)]
		r = np.array(r)
  		
		return r
		
	def mse_with_time(X1, X2):
		""" 
		calculates mean squared error (mse) between X1, X2 at each time
  		X1,X2 are 3D matrices with dimension [sample,time,feature]
  		output is 2D [time, feature] 		

  		"""
		assert X1.shape == X2.shape, 'input shape mismatch!'
		sqerr = (X1 - X2)**2
		mse = np.mean(sqerr, axis=0)

		return mse

	def rmse_with_time(X1, X2):
		""" 
		calculates root mean squared error (rmse) between X1, X2 at each time
  		X1,X2 are 3D matrices with dimension [sample,time,feature] 	

  		"""
		return np.sqrt(traj_err.mse_with_time(X1, X2))


