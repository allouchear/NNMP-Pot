import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import itertools

class Ewald(Layer):

	def __str__(self):
		return "Electrostatic/Ewald"

	def __init__(self,
		cutoff=None,
		use_ewald=True,		    # use ewald summation for periodic systems, coulomb if not
		use_scaled_charges=False,
		Nmax=[2,2,2],               # Nmax for k
		dtype=tf.float32            #single or double precision
		):
		super().__init__(dtype=dtype, name="Electrostatic/Ewald")
		self._use_scaled_charges = use_scaled_charges
		self._dtype=dtype
		self._cutoff = cutoff
		self._on_cut = 0.25 * cutoff
		self._off_cut = 0.75 * cutoff
		self._use_ewald_summation = use_ewald
		self._alpha = 0.0
		self._alpha2 = 0.0
		self._two_pi = 2.0 * math.pi
		self._kvecs = None
		self.set_alpha()
		self._set_kvecs(Nmax)

	#returns scaled charges such that the sum of the partial atomic charges equals Q_tot (defaults to 0)
	def scaled_charges(self, Z, Qa, Q_tot=None, batch_seg=None):
		if batch_seg is None:
			batch_seg = tf.zeros_like(Z) # one molecule
		#number of atoms per batch (needed for charge scaling)
		Na_per_batch = tf.math.segment_sum(tf.ones_like(batch_seg, dtype=self.dtype), batch_seg)
		if Q_tot is None: #assume desired total charge zero if not given
			Q_tot = tf.zeros_like(Na_per_batch, dtype=self.dtype)
		#return scaled charges (such that they have the desired total charge)
		return Qa + tf.gather(((Q_tot-tf.math.segment_sum(Qa, batch_seg))/Na_per_batch), batch_seg)


	def _set_kvecs(self, Nmax):
		""" Set integer reciprocal space cutoff for Ewald summation """
		k = []
		for i in range(3):
			kk = tf.range(Nmax[i] + 1,dtype=self.dtype)
			kk = tf.concat([kk, -kk[1:]],0)
			kk = kk.numpy().tolist()
			k.append(kk)
		lk=list(itertools.product(k[0],k[1],k[2]))[1:]  # 0th entry is 0 0 0
		#print('lk=',lk)
		kvecs=tf.constant(lk,dtype=self.dtype)
		kmax = max(Nmax)
		self._kvecs=kvecs[tf.math.reduce_sum(kvecs**2, axis=-1) <= kmax**2]
		#print('kvecs=',self._kvecs)

	def set_alpha(self, alpha=None):
		""" Set real space damping parameter for Ewald summation """
		# alpha here corresponds to kappa = sqrt(alpha) in some paper
		if alpha is None:  # automatically determine alpha
			alpha = 4.0 / self._cutoff + 1e-3
		self._alpha = alpha
		self._alpha2 = alpha ** 2
		self._two_pi = 2.0 * math.pi
		self._one_over_sqrtpi = 1 / math.sqrt(math.pi)
		# print a warning if alpha is so small that the reciprocal space sum
		# might "leak" into the damped part of the real space coulomb interaction
		if alpha * self._off_cut < 4.0:  # erfc(4.0) ~ 1e-8
			print("Warning: Damping parameter alpha is", alpha, 
			"but probably should be at least", 4.0 / self._off_cut)

	def _switch_component(self, x, ones, zeros):
		""" Component of the switch function, only for internal use. """
		x_ = tf.where(x <= 0, ones, x)  # prevent nan in backprop
		return tf.where(x <= 0, zeros, tf.math.exp(-ones / x_))


	def _switch_function(self,x):
		"""
		Switch function that smoothly (and symmetrically) goes from f(x) = 1 to
		f(x) = 0 in the interval from x = cuton to x = cutoff. For x <= cuton,
		f(x) = 1 and for x >= cutoff, f(x) = 0. This switch function has infinitely
		many smooth derivatives.
		NOTE: The implementation with the "_switch_component" function is
		numerically more stable than a simplified version, it is not recommended to change this!
		"""
		x = (x - self._on_cut) / (self._off_cut - self._on_cut)
		ones = tf.ones_like(x)
		zeros = tf.zeros_like(x)
		fp = self._switch_component(x, ones, zeros)
		fm = self._switch_component(1 - x, ones, zeros)
		return tf.where(x <= 0, ones, tf.where(x >= 1, zeros, fm / (fp + fm)))

	def _real_space(self, Qa, Dij, idx_i, idx_j):
		fac = tf.gather(Qa, idx_i)*tf.gather(Qa, idx_j)
		#print("fac=",fac)
		#print("Qa=",Qa)
		#print("idxi=",idx_i)
		#print("Dij=",Dij)
		f = self._switch_function(Dij)
		coulomb = 1.0 / Dij
		damped = 1.0 / (Dij**2 + 1.0)**(0.5)
		pw = fac*(f*damped + (1 - f)*coulomb)*tf.math.erfc(self._alpha*Dij)
		e_real = 0.5*tf.math.segment_sum(pw, idx_i)
		#print("e_real=",e_real.shape)
		return e_real

	def _reciprocal_space( self, Qa, R, Cell, idx_i, batch_seg, eps: float = 1e-8):
		# extract box dimensions from cells
		#print("Cell=",Cell)
		recip_box = 2.0 * np.pi * tf.linalg.inv(Cell)
		#print("recip_box before transpose=",recip_box)
		recip_box = tf.transpose(recip_box, perm = [0,2,1])
		#print("recip_box after transpose=",recip_box)
	
		#print(recip_box)
		v_box = tf.abs(tf.linalg.det(Cell))
		prefactor = 2.0 * np.pi / v_box
		#print("vbox=",v_box)
		#print("recip_box=",recip_box.shape)
		#print("kvecs=", self._kvecs.shape)
		
		# M =number of molecules, K=number of k vectors
		# calculate k-space vectors
		# setup kvecs M x K x 3 
		k = tf.linalg.matmul(self._kvecs, recip_box)
		#print("self._kvecs=",self._kvecs)
		#print("k=", k)

		# Squared length of vectors M x K
		k2 = tf.math.reduce_sum(k * k, axis=-1)  # squared length of k-vectors
		#print("k2=", k2)

		# gaussian charge density M x K
		#print("k2=", k2.shape)
		qg = tf.math.exp(-0.25 * k2 / self._alpha2)
		#print("qg=", qg.shape)
		#print("R shape=", R.shape)
		#print("idxi shape=", len(idx_i))

		# Compute charge density fourier terms
		# Dot product in exponent -> MN x K, expand kvecs in MN batch structure
		#print("k=", k)
		kk = tf.gather(k, batch_seg)
		#print("kk=", kk)
		kk = tf.transpose(kk, perm = [1,0,2])
		#print("kk shape=", kk.shape)
		#print("R shape=", R.shape)
		#print("kk*R=", (kk*R).shape)
		kvec_dot_pos = tf.math.reduce_sum(kk*R, axis=-1)
		#print("kvec_dot_pos=", kvec_dot_pos.shape)
		#print("Qa=", Qa.shape)

		# q_real MN x K
		q_real = Qa*tf.math.cos(kvec_dot_pos)
		q_real = tf.transpose(q_real)
		#print("qreal=", q_real.shape)
		# q_real M x K
		q_real = tf.math.segment_sum(q_real,batch_seg)
		q_imag = Qa*tf.math.sin(kvec_dot_pos)
		q_imag = tf.transpose(q_imag)
		# q_imag M x K
		q_imag = tf.math.segment_sum(q_imag,batch_seg)
		#print("qimag=", q_imag.shape)

		# Compute square of density
		q_dens = q_real**2 + q_imag**2
		#print("q_dens=", q_dens.shape)

		# Sum over k vectors -> M x K -> M
		#print("qg=", qg.shape)
		#print("k2=", k2.shape)
		e_reciprocal = prefactor * tf.math.reduce_sum(q_dens * qg / k2)
		#print("e_reciprocal=", e_reciprocal)

		# self interaction correction -> M
		Qa2 = Qa * Qa
		Qa2 = tf.math.segment_sum(Qa2, batch_seg)
		#print("Qa2=", Qa2.shape)
		e_self = -self._alpha * self._one_over_sqrtpi * Qa2
		#print("e_self=", e_self.shape)

		# back-ground charge -> M
		QaR= tf.math.segment_sum(Qa*R, batch_seg)
		e_bc = prefactor/3.0*QaR*QaR

		# e_reciprocal by molecule
		e_reciprocal = (e_reciprocal + e_self + e_bc)
		#print("e_reciprocal=", e_reciprocal.shape)

		#number of atoms per batch
		Na_per_batch = tf.math.segment_sum(tf.ones_like(batch_seg, dtype=self._dtype), batch_seg)
		er = []
		for imol in range(e_reciprocal.shape[0]):
			nAtoms=int(Na_per_batch[imol])
			er +=  [e_reciprocal[imol]/nAtoms]*nAtoms

		er= tf.convert_to_tensor(er, dtype=self._dtype)
		#print("er=", er.shape)

		return er

	def _ewald( self, Qa, R, Dij, idx_i, idx_j, Cell, batch_seg):
		e_real = self._real_space(Qa, Dij, idx_i, idx_j)
		e_reciprocal = self._reciprocal_space(Qa, R, Cell, idx_i, batch_seg)
		#print("ereal ewald =", e_real)
		#print("ereal e_reciprocal =", e_reciprocal)
		return e_real + e_reciprocal

	def energy_per_atom( self, Z, Dij, Qa, idx_i, idx_j, R= None, Q_tot=None, batch_seg=None,Cell=None):
		assert R is not None
		assert Cell is not None
		assert batch_seg is not None
		if self.use_scaled_charges:
			assert Q_tot is not None
			Qa = self.scaled_charges(Z, Qa, Q_tot=Q_tot, batch_seg=batch_seg)
		return self._ewald(Qa, R, Dij, idx_i, idx_j, Cell, batch_seg), Qa
		
	def print_parameters(self):
		pass

	@property
	def dtype(self):
		return self._dtype

	@property
	def sr_cut(self):
		return self._sr_cut

	@property
	def use_scaled_charges(self):
		return self._use_scaled_charges

	@property
	def num_outputs(self):
		return 2

