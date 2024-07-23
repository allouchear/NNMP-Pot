import tensorflow as tf
from tensorflow.keras.layers import Layer
from .RBFLayer import *
from .InteractionBlock import *
from .OutputBlock      import *
from Utils.ActivationFunctions import *

def softplus_inverse(x):
	'''numerically stable inverse of softplus transform'''
	return x + np.log(-np.expm1(-x))

class MessagePassingNeuralNetwork(Layer):
	def __str__(self):
		return "Message Passing Neural Network"+"\n"+str(self.rbf_layer.basis_type)+" radial basis"

	def __init__(self,
		F,                              #dimensionality of feature vector
		K,                              #number of radial basis functions
		sr_cut,                         #cutoff distance for short range interactions
		num_blocks=5,                   #number of building blocks to be stacked
		num_residual_atomic=2,          #number of residual layers for atomic refinements of feature vector
		num_residual_interaction=3,     #number of residual layers for refinement of message vector
		num_residual_output=1,          #number of residual layers for the output blocks
		num_outputs=2,        		#number of outputs by atom
                shifts=None,                    #initial value for output shift (makes convergence faster)
                scales=None,                    #initial value for output scale (makes convergence faster)
                drop_rate=None,                 #initial value for drop rate (None=No drop)
		activation_fn=shifted_softplus, #activation function
		basis_type="Default",           #radial basis type : GaussianNet (Default), Gaussian, Bessel, Slater, 
		beta=0.2,			#for Gaussian basis type
		dtype=tf.float32,               #single or double precision
		type_output=0,                  # 0=> average, 1=> Wx+b (modules outputs)
		seed=None):
		super().__init__(dtype=dtype, name="MessagePassingNeuralNetwork")

		assert(num_blocks > 0)
		assert(num_outputs > 0)
		self._num_blocks = num_blocks
		self._dtype = dtype
		self._num_outputs = num_outputs
		self._F = F
		self._K = K
		self._sr_cut = tf.constant(sr_cut,dtype=dtype) #cutoff for neural network interactions
		
		self._activation_fn = activation_fn
		#initialize output scale/shift variables
		if shifts is not None:
			sh0 = tf.constant(shifts,dtype=dtype)
		else:
			sh0 = tf.zeros([self.num_outputs],dtype=dtype)
		sh = []
		for i in range(self.num_outputs):
			sh.append(tf.constant(sh0[i], shape=[95], dtype=dtype))
		self._shifts = tf.stack(sh)
		if shifts is not None:
			self._shifts = tf.Variable(self._shifts,name="shifts")

		if scales is not None:
			sc0 = tf.constant(scales,dtype=dtype)
		else:
			sc0 = tf.ones([self.num_outputs],dtype=dtype)
		sc = []
		for i in range(self.num_outputs):
			sc.append(tf.constant(sc0[i], shape=[95], dtype=dtype))
		self._scales = tf.stack(sc)
		if scales is not None:
			self._scales = tf.Variable(self._scales,name="scales")


		#drop rate regularization
		if drop_rate is None:
			self._drop_rate = tf.Variable(0.0, shape=[], name="drop_rate",dtype=dtype,trainable=False)
		else:
			self._drop_rate = tf.Variable(0.0, shape=[], name="drop_rate",dtype=dtype,trainable=True)

		#atom embeddings (we go up to Pu(94), 95 because indices start with 0)
		self._embeddings = tf.Variable(tf.random.uniform([95, self.F], minval=-tf.math.sqrt(tf.cast(3.0,dtype=dtype)), maxval=tf.math.sqrt(tf.cast(3.0,dtype=dtype))
				   , seed=seed, dtype=dtype), name="embeddings", dtype=dtype,trainable=True)
		tf.summary.histogram("embeddings", self.embeddings)  

		#radial basis function expansion layer
		self._rbf_layer = RBFLayer(K,  self._sr_cut, beta=beta, basis_type=basis_type, name="rbf_layer",dtype=dtype)

		#embedding blocks and output layers
		self._interaction_block = []
		self._output_block = []
		for i in range(num_blocks):
			self.interaction_block.append(
			InteractionBlock(F, num_residual_atomic, num_residual_interaction, activation_fn=activation_fn, name="InteractionBlock"+str(i),
					seed=seed, drop_rate=self.drop_rate, dtype=dtype))
			self.output_block.append(
				OutputBlock(F, num_outputs, num_residual_output, activation_fn=activation_fn, name="OutputBlock"+str(i),
					seed=seed, drop_rate=self.drop_rate, dtype=dtype))

		one=tf.ones([self._num_blocks,self._num_outputs],dtype=dtype)
		self._W = tf.Variable(one,name="W", trainable=type_output==1)
		#zr=tf.zeros([self._num_outputs],dtype=dtype)
		#self._b = tf.Variable(zr, name="b", trainable=type_output==1)
		zr=tf.zeros([self._num_blocks,self._num_outputs],dtype=dtype)
		self._b = tf.Variable(zr,name="W", trainable=type_output==1)

	def calculate_interatomic_distances(self, R, idx_i, idx_j, offsets=None):
		#calculate interatomic distances
		Ri = tf.gather(R, idx_i)
		Rj = tf.gather(R, idx_j)
		if offsets is not None:
			Rj += offsets
		Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri-Rj)**2, -1))) #relu prevents negative numbers in sqrt
		return Dij

	#calculates output nn
	def nn_outputs(self, x, rbf, sr_idx_i=None, sr_idx_j=None):
		#outputs = tf.zeros([x.shape[0],self._num_outputs], dtype=x.dtype)
		outputs = 0
		#print("outputs=",outputs)
		nhloss = 0 #non-hierarchicality loss
		for i in range(self.num_blocks):
			x = self.interaction_block[i](x, rbf, sr_idx_i, sr_idx_j)
			out = self.output_block[i](x)
			#outputs += out*self._W[i]+self._b
			outputs += out*self._W[i]+self._b[i]
			#compute non-hierarchicality loss
			out2 = out**2
			if i > 0:
				nhloss += tf.reduce_mean(out2/(out2 + lastout2 + 1e-7))
			lastout2 = out2
		return outputs, nhloss


	def apply_scale_shift(self, outputs, Z):
		sc = []
		sh = []
		#sr = []
		for i in range(self.num_outputs):
			sc.append(tf.gather(self.scales[i], Z))
			sh.append(tf.gather(self.shifts[i], Z))
			#sr.append(tf.reduce_sum(R, -1))  # necessary to guarantee no "None" in force evaluation
		sc = tf.stack(sc)
		sc = tf.transpose(sc)
		sh = tf.stack(sh)
		sh = tf.transpose(sh)
		#sr = tf.stack(sr)
		#sr = tf.transpose(sr)
		#print("sc=",sc)
		#print("sh=",sh)
		outputs *= sc
		outputs += sh
		return outputs

	def get_distances(self, R, idx_i, idx_j, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None):
		#calculate distances (for long range interaction)
		Dij_lr = self.calculate_interatomic_distances(R, idx_i, idx_j, offsets=offsets)
		#optionally, it is possible to calculate separate distances for short range interactions (computational efficiency)
		si=sr_idx_i is not None and len(sr_idx_i)>0
		sj=sr_idx_j is not None and len(sr_idx_j)>0
		if si and sj :
			Dij_sr = self.calculate_interatomic_distances(R, sr_idx_i, sr_idx_j, offsets=sr_offsets)
		else:
			sr_idx_i = idx_i
			sr_idx_j = idx_j
			Dij_sr = Dij_lr
		return Dij_lr, Dij_sr, sr_idx_i, sr_idx_j


	#calculates the atomic properties and distances (needed if unscaled charges are wanted e.g. for loss function)
	def atomic_properties(self, Z, R, idx_i, idx_j, M=None, QaAlpha=None, QaBeta=None, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None, batch_seg=None):
		#calculate distances (for long range interaction)
		Dij_lr, Dij_sr, sr_idx_i, sr_idx_j = self.get_distances(R, idx_i, idx_j, offsets=offsets, sr_idx_i=sr_idx_i, sr_idx_j=sr_idx_j, sr_offsets=sr_offsets)

		#calculate radial basis function expansion
		rbf = self.rbf_layer(Dij_sr)
		x = tf.gather(self.embeddings, Z)
		#outputs, nhloss = self.nn_outputs(x, rbf, sr_idx_i=sr_idx_i, sr_idx_j=sr_idx_j)
		outputs, nhloss = self.nn_outputs(x, rbf, sr_idx_i=sr_idx_i, sr_idx_j=sr_idx_j)
		outputs = self.apply_scale_shift(outputs, Z)

		return outputs, Dij_lr, nhloss

	@property
	def drop_rate(self):
		return self._drop_rate
    
	@property
	def num_blocks(self):
		return self._num_blocks

	@property
	def num_outputs(self):
		return self._num_outputs

	@property
	def dtype(self):
		return self._dtype

	@property
	def embeddings(self):
		return self._embeddings

	@property
	def F(self):
		return self._F

	@property
	def K(self):
		return self._K

	@property
	def sr_cut(self):
		return self._sr_cut

	@property
	def activation_fn(self):
		return self._activation_fn
    
	@property
	def rbf_layer(self):
		return self._rbf_layer

	@property
	def interaction_block(self):
		return self._interaction_block

	@property
	def output_block(self):
		return self._output_block

	@property
	def scales(self):
		return self._scales

	@property
	def shifts(self):
		return self._shifts

