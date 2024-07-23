import tensorflow as tf
from tensorflow.keras.layers import Layer
from .RBFLayer import *
from .InteractionBlock import *
from .OutputBlock      import *
from .ElementalModesBlock import *
from Utils.ActivationFunctions import *

def softplus_inverse(x):
	'''numerically stable inverse of softplus transform'''
	return x + np.log(-np.expm1(-x))

#returns scaled charges such that the sum of the partial atomic charges equals Q_tot (defaults to 0)
def scaled_charges(Z, Qa, Q_tot=None, batch_seg=None): # Qa not Qal including for GFN1
	if batch_seg is None:
		batch_seg = tf.zeros_like(Z)
	#number of atoms per batch (needed for charge scaling)
	Na_per_batch = tf.math.segment_sum(tf.ones_like(batch_seg, dtype=Qa.dtype), batch_seg)
	if Q_tot is None: #assume desired total charge zero if not given
		Q_tot = tf.zeros_like(Na_per_batch, dtype=Qa.dtype)
	#return scaled charges (such that they have the desired total charge)
	return Qa + tf.gather(((Q_tot-tf.math.segment_sum(Qa, batch_seg))/Na_per_batch), batch_seg)

#returns scaled atomic spin charges to reproduce the molecule spin charges
def NSE(Z, QaA, QaB, faA, faB, QAlpha_mol=None, QBeta_mol=None, batch_seg=None): # Q_mol=None => mol charge = 0
	if batch_seg is None:
		batch_seg = tf.zeros_like(Z) # all atoms in one molecule
	#number of atoms per batch (needed for charge scaling)
	Na_per_batch = tf.math.segment_sum(tf.ones_like(batch_seg, dtype=QaA.dtype), batch_seg)
	if QAlpha_mol is None: #assume desired total charge zero if not given
		QAlpha_mol = tf.zeros_like(Na_per_batch, dtype=QaA.dtype)
	if QBeta_mol is None: #assume desired total charge zero if not given
		QBeta_mol = tf.zeros_like(Na_per_batch, dtype=QaB.dtype)
	#return scaled charges (such that they have the desired total charge)
	QaAlpha = QaA + faA*tf.gather(((QAlpha_mol-tf.math.segment_sum(QaA, batch_seg))/tf.math.segment_sum(faA, batch_seg)), batch_seg)
	QaBeta  = QaB + faB*tf.gather(((QBeta_mol -tf.math.segment_sum(QaB, batch_seg))/tf.math.segment_sum(faB, batch_seg)), batch_seg)
	return QaAlpha, QaBeta



class ElementalModesMessagePassingNeuralNetwork(Layer):
	def __str__(self):
		return "Elemental Modes Message Passing Neural Network"+"\n"+str(self.rbf_layer.basis_type)+" radial basis"

	def __init__(self,
		F,                              #dimensionality of feature vector
		K,                              #number of radial basis functions
		sr_cut,                         #cutoff distance for short range interactions
		num_scc=0,                      #number of cycles of updating of spin atomic charges (0 => no cycle)
		em_type = 0,       		#elemental modes type : 0(default) => only Z, 1=> Z + Masses, 2=> Z+Masses+QaAlpha+QaBeta (so we use charge &multiplicity)
		num_hidden_nodes_em = None, 	#number of hidden nodes by layer in element modes block , None => F
		num_hidden_layers_em = 2, 	#number of hidden layer in element modes block
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
		type_output=0,                  # 0=> average, 1=> Wx+b (modules outputs)
		dtype=tf.float32,               #single or double precision
		seed=None):
		super().__init__(dtype=dtype, name="MessagePassingNeuralNetwork")

		assert(num_blocks > 0)
		assert(num_outputs > 0)
		self._num_blocks = num_blocks
		self._dtype = dtype
		self._num_scc=num_scc

		self._num_outputs = num_outputs 

		self._F = F
		self._K = K
		self._em_type = em_type
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

		#elemental_modes_block blocks and output layers
		self._elemental_modes_block = ElementalModesBlock(F, num_hidden_nodes=num_hidden_nodes_em, num_hidden_layers=num_hidden_layers_em, activation_fn=activation_fn, seed=seed, drop_rate=drop_rate, dtype=dtype, name="elemental_modes_block")

		#radial basis function expansion layer
		self._rbf_layer = RBFLayer(K,  self._sr_cut, beta=beta, basis_type=basis_type, name="rbf_layer",dtype=dtype)

		#embedding blocks and output layers
		self._interaction_block = []
		self._output_block = []
		nouts = num_outputs 
		if num_scc > 0:
			nouts += 4 # for QaAlpha, faAlpha, QaBeta & faBeta : the 4 first outputs

		for i in range(num_blocks):
			self.interaction_block.append(
			InteractionBlock(F, num_residual_atomic, num_residual_interaction, activation_fn=activation_fn, name="InteractionBlock"+str(i),
					seed=seed, drop_rate=self.drop_rate, dtype=dtype))
			self.output_block.append(
				OutputBlock(F, nouts, num_residual_output, activation_fn=activation_fn, name="OutputBlock"+str(i),
					seed=seed, drop_rate=self.drop_rate, dtype=dtype))
		one=tf.ones([self._num_blocks,self._num_outputs],dtype=dtype)
		zr=tf.zeros([self._num_outputs],dtype=dtype)
		self._W = tf.Variable(one,name="W", trainable=type_output==1)
		self._b = tf.Variable(zr, name="b", trainable=type_output==1)


	def calculate_interatomic_distances(self, R, idx_i, idx_j, offsets=None):
		#calculate interatomic distances
		Ri = tf.gather(R, idx_i)
		Rj = tf.gather(R, idx_j)
		if offsets is not None:
			Rj += offsets
		Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri-Rj)**2, -1))) #relu prevents negative numbers in sqrt
		return Dij

	def get_input_elements(self, Z, M, QaAlpha, QaBeta):

		#QaAlpha = None
		#QaBeta = None

		Z =tf.Variable(Z,dtype=self.dtype)
		Z = tf.reshape(Z,[Z.shape[0],1])
		if M is not None and self.em_type >=1 :
			M =tf.Variable(M,dtype=self.dtype)
			M = tf.reshape(M,[M.shape[0],1])
		if QaAlpha is not None and self.em_type >=2 :
			QaAlpha =tf.Variable(QaAlpha,dtype=self.dtype)
			QaAlpha = tf.reshape(QaAlpha,[QaAlpha.shape[0],1])
		if QaBeta is not None and self.em_type >=2 :
			QaBeta =tf.Variable(QaBeta,dtype=self.dtype)
			QaBeta = tf.reshape(QaBeta,[QaBeta.shape[0],1])
		#print("QaAlpha=",QaAlpha)
		#print("QaBeta=",QaBeta)
		#print("M=",M)

		f = None
		if M is not None and QaAlpha is not None and  QaBeta is not None and self.em_type >=2:
			f=tf.concat([Z,M,QaAlpha, QaBeta],1)
		elif M is not None and self.em_type >=1:
			f=tf.concat([Z,M],1)
		else:
			f=tf.concat([Z],1)

		return f

	#calculates output nn
	def nn_outputs(self, rbf, Z, M=None, QaAlpha=None, QaBeta=None, sr_idx_i=None, sr_idx_j=None, batch_seg=None):

		QaA = tf.Variable(QaAlpha, dtype=self.dtype)
		QaB = tf.Variable(QaBeta, dtype=self.dtype)
		faA = tf.ones([len(QaAlpha)],dtype=self.dtype)
		faB = tf.ones([len(QaBeta)],dtype=self.dtype)
		QAlpha_mol = tf.math.segment_sum(QaA, batch_seg)
		QBeta_mol  = tf.math.segment_sum(QaB, batch_seg)
		#print("QAlpha_mol=",QAlpha_mol)
		#print("QBeta_mol=",QBeta_mol)
		ibegin=0
		if self.num_scc>0:
			ibegin=4
		for j in range(self.num_scc+1):
			f = self.get_input_elements(Z, M, QaA, QaB)
			x = self.elemental_modes_block(f)
			#print("x=",x)
			outputs = 0
			#print("outputs=",outputs)
			nhloss = 0 #non-hierarchicality loss
			for i in range(self.num_blocks):
				x = self.interaction_block[i](x, rbf, sr_idx_i, sr_idx_j)
				out = self.output_block[i](x)
				outputs += out*self._W[i]+self._b
				#compute non-hierarchicality loss
				if j==self.num_scc:
					out2 = out[ibegin:]**2
					if i > 0:
						nhloss += tf.reduce_mean(out2/(out2 + lastout2 + 1e-7))
					lastout2 = out2

			if self.num_scc>0:
				QaA = outputs[:,0]
				faA = outputs[:,1]
				QaB = outputs[:,2]
				faB = outputs[:,3]
				QaA, QaB = NSE(Z, QaA, QaB, faA, faB, QAlpha_mol=QAlpha_mol, QBeta_mol=QBeta_mol, batch_seg=batch_seg)
				#print("j=",j,"QAlpha_mol=", tf.math.segment_sum(QaA, batch_seg))

		if self.num_scc>0:
			outputs = outputs[:,ibegin:]
	
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
		if sr_idx_i is not None and sr_idx_j is not None:
			Dij_sr = self.calculate_interatomic_distances(R, sr_idx_i, sr_idx_j, offsets=sr_offsets)
		else:
			sr_idx_i = idx_i
			sr_idx_j = idx_j
			Dij_sr = Dij_lr
		return Dij_lr, Dij_sr, sr_idx_i, sr_idx_j



	#calculates the atomic properties and distances (needed if unscaled charges are wanted e.g. for loss function)
	def atomic_properties(self, Z, R, idx_i, idx_j, M=None, QaAlpha=None, QaBeta=None, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None, batch_seg=None):
		Dij_lr, Dij_sr, sr_idx_i, sr_idx_j = self.get_distances(R, idx_i, idx_j, offsets=offsets, sr_idx_i=sr_idx_i, sr_idx_j=sr_idx_j, sr_offsets=sr_offsets)

		#calculate radial basis function expansion
		rbf = self.rbf_layer(Dij_sr)
		outputs, nhloss = self.nn_outputs(rbf, Z, M=M, QaAlpha=QaAlpha, QaBeta=QaBeta, sr_idx_i=sr_idx_i, sr_idx_j=sr_idx_j, batch_seg=batch_seg)
		outputs = self.apply_scale_shift(outputs, Z)

		return outputs, Dij_lr, nhloss

	@property
	def drop_rate(self):
		return self._drop_rate

	@property
	def num_scc(self):
		return self._num_scc
    
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
	def elemental_modes_block(self):
		return self._elemental_modes_block

	@property
	def F(self):
		return self._F

	@property
	def K(self):
		return self._K

	@property
	def em_type(self):
		return self._em_type

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

