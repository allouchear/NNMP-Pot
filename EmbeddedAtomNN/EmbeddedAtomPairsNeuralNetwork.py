import tensorflow as tf
from tensorflow.keras.layers import Layer
from .BFLayer import *
from .InteractionBlock import *
from .OutputBlock      import *
from Utils.ActivationFunctions import *

def softplus_inverse(x):
	'''numerically stable inverse of softplus transform'''
	return x + np.log(-np.expm1(-x))

class EmbeddedAtomPairsNeuralNetwork(Layer):
	def __str__(self):
		return "Embedded Atom Neural Network/"+str(self.bf_layer.basis_type)+" radial basis"

	def __init__(self,
		K,                              	# number of basis functions for a given L
		Lmax,                           	# Lmax (same value for all atoms)
		sr_cut,                         	# cutoff distance for short range interactions
		beta=0.2,                       	# beta value for basis functions
		num_blocks=5,                   	# number of building blocks to be stacked
		num_interaction_layers=1,     		# number of hidden layers for interaction block (1 at least)
		num_output_layers=1,          		# number of hidden layers for the output blocks (can be =0 !)
		num_interaction_nodes=10,     		# number of nodes in each hidden layer of interaction block 
		num_output_nodes=10,          		# number of nodes in each hidden layer of output blocks 
		num_outputs=2,        			# number of outputs by atom
                shifts=None,                     	# initial value for output shift (makes convergence faster)
                scales=None,                     	# initial value for output scale (makes convergence faster)
                drop_rate=None,                  	# initial value for drop rate (None=No drop)
		activation_fn=shifted_softplus, 	# activation function
		basis_type="Default",           # radial basis type : Gaussian (Default), GaussianNet, Bessel, Slater, 
		dtype=tf.float32,               	# single or double precision
		seed=None):
		super().__init__(dtype=dtype, name="EmbeddedAtomNeuralNetwork")

		assert(num_blocks > 0)
		assert(num_outputs > 0)
		self._num_blocks = num_blocks
		self._dtype = dtype
		self._num_outputs = num_outputs
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

		# basis function expansion layer
		self._bf_layer = BFLayer(K,  self._sr_cut, Lmax=Lmax, beta=beta, basis_type=basis_type, name="bf_layer",dtype=dtype)

		self._interaction_block = []
		self._output_block = []
		for i in range(num_blocks):
			self.interaction_block.append(
			InteractionBlock(num_interaction_nodes, num_interaction_layers, activation_fn=activation_fn, name="InteractionBlock_"+str(i),
					seed=seed, drop_rate=self.drop_rate, dtype=dtype))
			self.output_block.append(
				OutputBlock(num_output_nodes, num_outputs, num_output_layers, activation_fn=activation_fn, name="OutputBlock_"+str(i),
					seed=seed, drop_rate=self.drop_rate, dtype=dtype))

	#calculates the atomic properties and distances (needed if unscaled charges are wanted e.g. for loss function)
	def atomic_properties(self, Z, R, idx_i, idx_j, M=None, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None, batch_seg=None):
		#optionally, it is possible to calculate separate distances for short range interactions (computational efficiency)
		if sr_idx_i is not None and sr_idx_j is not None:
			bf, rij = self.bf_layer.getPairs(Z, R, sr_idx_i, sr_idx_j, offsets=sr_offsets)
			idx_i = sr_idx_i
			idx_j = sr_idx_j
		else:
			bf, rij = self.bf_layer.getPairs(Z, R, idx_i, idx_j, offsets=offsets)

		#apply blocks
		#outputs = tf.zeros([x.shape[0],self._num_outputs], dtype=x.dtype)
		outputs = 0
		#print("outputs=",outputs)
		nhloss = 0 #non-hierarchicality loss
		x = bf
		#print("idxi=",idx_i)
		#print("idxj=",idx_j)
		#print("rij=",rij)
		#print("x(bf)=",x)
		for i in range(self.num_blocks):
			x = self.interaction_block[i](x)
			#if i==0:
			#	print("x0=",x)
			#print("x=",x)
			out = self.output_block[i](x)
			# Pairs out, Si = sum(Sij)
			out = tf.squeeze(tf.math.segment_sum(out, idx_i))
			#print("out=",out)
			outputs += out
			#compute non-hierarchicality loss
			out2 = out**2
			if i > 0:
				nhloss += tf.reduce_mean(out2/(out2 + lastout2 + 1e-7))
			lastout2 = out2
		#print("outputsAll=",outputs)
		#apply scaling/shifting
		#print("outputs=",outputs)
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
		#print("outputs=",outputs)
		#print("sh=",sh)
		outputs *= sc
		outputs += sh
		#outputs += 0*sr # necessary to guarantee no "None" in force evaluation

		#outputs = tf.constant(newout)
		#print("outputsShape=",outputs.shape)
		#print("outputs=",outputs)
		return outputs, rij, nhloss

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
	def F(self):
		return self._F

	@property
	def sr_cut(self):
		return self._sr_cut

	@property
	def activation_fn(self):
		return self._activation_fn
    
	@property
	def bf_layer(self):
		return self._bf_layer

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

