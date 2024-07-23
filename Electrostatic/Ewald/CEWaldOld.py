import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class CEWald(Layer):

	def __str__(self):
		return "Electrostatic/CEWald"

	def __init__(self,
		cutoff=None,
		use_scaled_charges=False,
		dtype=tf.float32            #single or double precision
		):
		super().__init__(dtype=dtype, name="Electrostatic/CEWald")
		self._use_scaled_charges = use_scaled_charges
		self._dtype=dtype
		self.on_cut = 0.25 * self.cutoff
		self._off_cut = 0.75 * self.cutoff
		self._use_ewald_summation = False
		self._alpha = 0.0
		self._alpha2 = 0.0
		self._two_pi = 2.0 * math.pi
		self._kmul = None
		self.set_alpha()

	#returns scaled charges such that the sum of the partial atomic charges equals Q_tot (defaults to 0)
	def scaled_charges(self, Z, Qa, Q_tot=None, batch_seg=None):
		if batch_seg is None:
			batch_seg = tf.zeros_like(Z)
		#number of atoms per batch (needed for charge scaling)
		Na_per_batch = tf.math.segment_sum(tf.ones_like(batch_seg, dtype=self.dtype), batch_seg)
		if Q_tot is None: #assume desired total charge zero if not given
			Q_tot = tf.zeros_like(Na_per_batch, dtype=self.dtype)
		#return scaled charges (such that they have the desired total charge)
		return Qa + tf.gather(((Q_tot-tf.math.segment_sum(Qa, batch_seg))/Na_per_batch), batch_seg)


	def set_kmax(self, Nxmax, Nymax, Nzmax):
		""" Set integer reciprocal space cutoff for Ewald summation """
		kx = tf.range(Nxmax + 1,dtype=self.dtype)
		kx = tf.concat([kx, -kx[1:]])
		ky = tf.range(Nymax + 1,dtype=self.dtype)
		ky = tf.concat([ky, -ky[1:]])
		kz = tf.range(Nzmax + 1,dtype=self.dtype)
		kz = tf.concat([kz, -kz[1:]])
		kmul = torch.cartesian_prod(kx, ky, kz)[1:]  # 0th entry is 0 0 0
		kmax = max(max(Nxmax, Nymax), Nzmax)
		sel._kmul=kmul[tf.sum(kmul**2, axis=-1) <= kmax**2]

	def set_alpha(self, alpha=None):
		""" Set real space damping parameter for Ewald summation """
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

	def _real_space(self, N, Qa, rij, idx_i, idx_j):
		fac = tf.gather(Qa, idx_i)*tf.gather(Qa, idx_j)
		f = self._switch_function(rij)
		coulomb = 1.0 / rij
		damped = 1.0 / (rij**2 + 1.0)**(0.5)
		pw = fac*(f*damped + (1 - f)*coulomb)*tf.math.erfc(self._alpha*rij)
		return tf.math.segment_sum(pw, idx_i)


	#returns scaled charges such that the sum of the partial atomic charges equals Q_tot (defaults to 0)
	def scaled_charges(self, Z, Qa, Q_tot=None, batch_seg=None):
		if batch_seg is None:
			batch_seg = tf.zeros_like(Z)
		#number of atoms per batch (needed for charge scaling)
		Na_per_batch = tf.math.segment_sum(tf.ones_like(batch_seg, dtype=self.dtype), batch_seg)
		if Q_tot is None: #assume desired total charge zero if not given
			Q_tot = tf.zeros_like(Na_per_batch, dtype=self.dtype)
		#return scaled charges (such that they have the desired total charge)
		return Qa + tf.gather(((Q_tot-tf.math.segment_sum(Qa, batch_seg))/Na_per_batch), batch_seg)

	#switch function for electrostatic interaction (switches between shielded and unshielded electrostatic interaction)
	def _switch(self, Dij):
		cut = self.sr_cut/2.0
		x  = Dij/cut
		x3 = x*x*x
		x4 = x3*x
		x5 = x4*x
		return tf.where(Dij < cut, 6*x5-15*x4+10*x3, tf.ones_like(Dij))


	def _switch_component( x, ones, zeros):
		""" Component of the switch function, only for internal use. """
		x_ = tf.where(x <= 0, ones, x)  # prevent nan in backprop
		return tf.where(x <= 0, zeros, tf.math.exp(-ones / x_))


	def _switch_function(x):
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
		fp = _switch_component(x, ones, zeros)
		fm = _switch_component(1 - x, ones, zeros)
		return tf.where(x <= 0, ones, tf.where(x >= 1, zeros, fm / (fp + fm)))


	def _reciprocal_space( self, q, R, cell, num_batch, batch_seg, eps: float = 1e-8):
		# calculate k-space vectors
		box_length = tf.diagonal(cell, dim1=-2, dim2=-1)
		k = self._two_pi * tf.expand_dims(self.kmul, axis=0) / tf.expand_dims(box_length, axis=-2)
		# gaussian charge density
		k2 = tf.sum(k * k, axis=-1)  # squared length of k-vectors
		qg = tf.math.exp(-0.25 * k2 / self._alpha2) / k2
		# fourier charge density
		dot = tf.sum(k[batch_seg] * tf.expand_dims(R,axis=-2), axis=-1)
		q_real = q.new_zeros(num_batch, dot.shape[-1]).index_add_( 0, batch_seg, q.unsqueeze(-1) * tf.cos(dot))
		q_imag = q.new_zeros(num_batch, dot.shape[-1]).index_add_( 0, batch_seg, q.unsqueeze(-1) * tf.sin(dot))
		qf = q_real ** 2 + q_imag ** 2
		# reciprocal energy
		e_reciprocal = ( self.two_pi / tf.prod(box_length, dim=1) * tf.sum(qf * qg, dim=-1))
		# self interaction correction
		q2 = q * q
		e_self = self.alpha * self.one_over_sqrtpi * q2
		# spread reciprocal energy over atoms (to get an atomic contributions)
		w = q2 + eps  # epsilon is added to prevent division by zero
		wnorm = w.new_zeros(num_batch).index_add_(0, batch_seg, w)
		w = w / tf.gather(wnorm, batch_seg)
		e_reciprocal = w * tf.gather(e_reciprocal, batch_seg)
		return (e_reciprocal - e_self)

	def _ewald( self, N, q, R, rij, idx_i, idx_j, cell, num_batch, batch_seg):
		e_real = self._real_space(N, q, rij, idx_i, idx_j)
		e_reciprocal = self._reciprocal_space(q, R, cell, num_batch, batch_seg)
		return e_real + e_reciprocal

	def energy_per_atom( self, N, Qa, rij, idx_i, idx_j, R= None, cell=None, Q_tot=None, batch_seg=None):
		assert R is not None
		assert cell is not None
		assert batch_seg is not None
		if self.use_scaled_charges:
			assert Q_tot is not None
			Qa = self.scaled_charges(Z, Qa, Q_tot=Q_tot, batch_seg=batch_seg)
		num_batch = tf.math.segment_sum(tf.ones_like(batch_seg, dtype=self.dtype), batch_seg)
		return self._ewald(N, Qa, R, rij, idx_i, idx_j, cell, num_batch, batch_seg), Qa
		
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

