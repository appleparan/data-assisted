import os
import numpy as np

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_series import Lorenz_base_dynamics
from data_gen import Lorenz
from error_func import traj_err

"""
Implementation of model architecture 1 for CdV

"""

class lstm_hybrid(object):
	
	fdim = 3

	def __init__(self, trunc_dim, l, hid_units_x, hid_units_B, params, savepath):

		self.dim = self.fdim - trunc_dim		# should be 2 = 3 - 1
		self.l = l 								# length of training series  (time step)
		self.hist = None
		
		self.savepath = savepath
		if not os.path.exists(savepath):
			os.makedirs(savepath)
		
		## declare layer objects in the model
		self.inputs = layers.Input(shape=(l,self.dim), name='ySeq')
		self.x_lstm = layers.recurrent.LSTM(hid_units_x, input_shape=(None, self.fdim), 
							return_sequences=True, name='x_lstm', implementation=2)
		self.Bxy_hidden = layers.Dense(units=hid_units_B, activation='relu', name='Bxy_hidden')
		self.Bxy_output = layers.Dense(units=self.dim, activation='linear', name='Bxy_output')
		self.Ly_lambda = layers.core.Lambda(function=self.trunc_dynamics, output_shape=(self.dim,),
								arguments={'params':params}, name='Ly_lambda')
		

	## implement POD dynamics in Keras environment
	def trunc_dynamics(self, y_trunc, params):
		# x_mean & eigenvector
		xm, w = K.constant(params['xm']), K.constant(params['W'])
		# Linear operator & offset operator on chi
		L, b = K.constant(params['L_y']), K.constant(params['b_y'])
		# standardardized
		ys = K.constant(params['y_std'])

		# recover original state representations and calculate nonlinear terms
		x = xm + K.dot(y_trunc*ys[:self.dim], K.transpose(w[:,:self.dim]))
		NX_list = []
		NX_list.append(-x[:, 0] * x[:, 2])
		NX_list.append(x[:, 0] * x[:, 1])
		#NX_list.append(-CdV.alpha[0]*x[:,0]*x[:,2] - CdV.delta[0]*x[:,3]*x[:,5])
		#NX_list.append(CdV.alpha[0]*x[:,0]*x[:,1] + CdV.delta[0]*x[:,3]*x[:,4])
		#NX_list.append(CdV.epsilon*(x[:,1]*x[:,5] - x[:,2]*x[:,4]))
		#NX_list.append(-CdV.alpha[1]*x[:,0]*x[:,5] - CdV.delta[1]*x[:,2]*x[:,3])
		#NX_list.append(CdV.alpha[1]*x[:,0]*x[:,4] + CdV.delta[1]*x[:,3]*x[:,1])
		NX = K.stack(NX_list, axis=-1)

		dydt_trunc = K.dot(y_trunc, L[:self.dim]) + b + K.dot(NX, w[1:])/ys

		return dydt_trunc[:,:self.dim]


	## build computation graph and compile 
	def compile(self):
		
		x_series = self.x_lstm(self.inputs)
		
		xy = layers.concatenate([x_series, self.inputs], axis=-1)
		Bxy = layers.TimeDistributed(self.Bxy_hidden)(xy)
		Bxy = layers.TimeDistributed(self.Bxy_output)(Bxy)

		Ly = layers.TimeDistributed(self.Ly_lambda)(self.inputs)
		dydt = layers.add([Ly,Bxy])
		# self.outputs = dydt
		self.outputs = Bxy
		self.total_dynamics = dydt

		self.model = Model(inputs=[self.inputs], outputs=[self.outputs])
		self.model.compile(optimizer='adam', loss='mse')
		self.dynamics_predictor = K.function(inputs=[self.inputs], outputs=[self.total_dynamics])
		self.hidden_calc = K.function(inputs=[self.inputs], outputs=[x_series])


	def train(self, inputs, outputs, pretrain=None, val_ratio=.1, 
				batch_sz=256, epochs=50, saveWeights=True):
		""" 
		inputs, outputs = 3D np array with dimensions (d,nb_samples,features)

		"""
		nb_samples = inputs.shape[0]
		idx_cut = int(nb_samples * (1 - val_ratio))

		inputs_train, inputs_val = inputs[:idx_cut,:,:], inputs[idx_cut:,:,:]
		outputs_train, outputs_val = outputs[:idx_cut,:,:], outputs[idx_cut:,:,:]

		print('Training...')
		if pretrain is not None:				# load pretrained weights if provided
			self.model.load_weights(pretrain)
		self.hist = self.model.fit([inputs_train], [outputs_train], epochs=epochs, batch_size=batch_sz,
								validation_data=(inputs_val, outputs_val), verbose=2)

		if saveWeights:
			self.model.save_weights(self.savepath+'trained_weights.h5')

		return self.hist
		

	## predict complementary dynamics as a series
	def predict(self, y_trunc, loadWeights=None):

		if loadWeights is not None:
			self.model.load_weights(loadWeights)
		
		return self.model.predict([y_trunc])


	## predict total dynamics as a series
	def predict_total(self, y_trunc, loadWeights=None):

		if loadWeights is not None:
			self.model.load_weights(loadWeights)
		
		return self.dynamics_predictor([y_trunc])


	## output hidden state of the lstm
	def predict_hid(self, y_trunc, loadWeights=None):

		if loadWeights is not None:
			self.model.load_weights(loadWeights)
		
		return self.hidden_calc([y_trunc])


	## plot training history
	def plot_history(self, saveFig=True):
		
		assert self.hist is not None, 'No training history found.'
		print('Plotting history...')

		plt.figure()
		plt.semilogy(self.hist.history['loss'],'b-',label='training')
		plt.semilogy(self.hist.history['val_loss'],'r-',label='validation')
		plt.legend(frameon=False)
		plt.title('Loss Curve - Overall')
		if saveFig:
			plt.savefig(self.savepath+'loss_overall.png',dpi=300)


## function that generates data series for training
def data_proc(fname, trunc_dim, l, idx_start=0, idx_end=10000, dt=.01):
	cwd = os.getcwd()
	npzfile = np.load(os.path.join(cwd, fname))

	# load L_y, b_y, xm, w, y_std
	phys_model = Lorenz_base_dynamics(npzfile, dt)
	# initial xi
	y0 = npzfile['y'][idx_start:idx_end].copy()
	inputs, outputs = [],[]
	total_dyn, trunc_dyn = [], []
	hid_coords = []

	for i in range(l):
		# without trunc_dim	(except last 1 dim), known
		inputs.append(y0[:,:-trunc_dim].copy())
		# with trunc_dim (last 1 dim), truncated
		hid_coords.append(y0[:,-trunc_dim:].copy())
		# evolve sampled data with xi coord
		dy0dt = phys_model.true_dynamics(y0)
		
		y0_trunc = y0.copy()
		# truncated data only
		y0_trunc[:,-trunc_dim:] = 0
		# start without truncdata
		dy0dt_trunc = phys_model.true_dynamics(y0_trunc)

		# real evolution - truncated data only evolution
		dy0dt_diff = dy0dt - dy0dt_trunc

		# time integration
		y0 += dt*dy0dt
		outputs.append(dy0dt_diff[:,:-trunc_dim].copy())
		total_dyn.append(dy0dt[:,:-trunc_dim].copy())
		trunc_dyn.append(dy0dt_trunc[:,:-trunc_dim].copy())
	
	inputs, outputs = np.array(inputs), np.array(outputs)
	inputs = np.swapaxes(inputs, 0, 1)
	outputs = np.swapaxes(outputs, 0, 1)
	total_dyn = np.swapaxes(np.array(total_dyn), 0, 1)
	trunc_dyn = np.swapaxes(np.array(trunc_dyn), 0, 1)

	hid_coords = np.swapaxes(np.array(hid_coords), 0, 1)

	return inputs, outputs, total_dyn, trunc_dyn, hid_coords


def main():
	# read initial data, from generate_series_true
	cwd = os.getcwd()
	fname = 'data\\modal_coords_100k_std_lorenz.npz'
	npzfile = np.load(os.path.join(cwd, fname))

	nx, nB = 1, 16    # number of hidden units in layers, nx = n_LSTM, nB = n_FC
	trunc_dim = 1    # number of truncated dimensions (1 means 5-d POD) n_LSTM
	l = 50       # number of time steps in model
	sp = './logs/lstm_arch1/test2/'
	total_sample = 4000
	# 80% for training
	train_inputs, train_outputs, *_ = data_proc(fname, trunc_dim, l, idx_end=total_sample)
	# 10% for testing
	test_inputs, test_outputs, test_total, test_trunc, trunc_coords = data_proc(fname, trunc_dim, l, idx_start=total_sample)
	# 10% for validation

	lstm_model = lstm_hybrid(trunc_dim, l, hid_units_x=nx, hid_units_B=nB, params=npzfile, savepath=sp)
	lstm_model.compile()
	lstm_model.model.summary()
	lstm_model.train(train_inputs, train_outputs, epochs=500, pretrain=None)
	lstm_model.plot_history()

	## load saved weights and make test predictions
	weights = sp + 'trained_weights.h5'
	test_pred = lstm_model.predict_total(test_inputs, loadWeights=weights)
	
	## plot predictions for a test case
	case_idx = l - 1
	tt = np.linspace(.01,.01*l,l)
	R,C = 1, lstm_model.fdim - trunc_dim
	f, axarr = plt.subplots(R,C)
	row = 0
	for col in range(C):
		dim = row*C + col
		axarr[col].plot(tt, test_total[case_idx,:,dim], 'r-.', label='true')
		axarr[col].plot(tt, test_pred[0][case_idx,:,dim], 'b-', label='pred')
		axarr[col].plot(tt, test_trunc[case_idx,:,dim], 'g-', label='trunc')
		axarr[col].set_xlim([0,.01*l])
		axarr[col].set_title('mode '+str(dim+1))
		axarr[col].legend(frameon=False)
	plt.tight_layout()
	plt.savefig(lstm_model.savepath+'tc'+str(case_idx)+'_total_l'+str(l)+'.png', dpi=300)


if __name__ == '__main__':
	main()
