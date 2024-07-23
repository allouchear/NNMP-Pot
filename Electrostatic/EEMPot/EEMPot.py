import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from Electrostatic.EEM.EEM import *

class EEMPot(Layer):

	def __str__(self):
		return "EEMPot"

	def __init__(self,
		fit_parameters=0,
		use_scaled_charges=False,
		dtype=tf.float32            #single or double precision
		):
		super().__init__(dtype=dtype, name="EEMPot")
		self._use_scaled_charges = use_scaled_charges
		self._dtype=dtype
		self._eem = EEM(fit_parameters=fit_parameters,dtype=dtype)

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

	#calculates the electrostatic energy per atom 
	#for very small distances, the 1/r law is shielded to avoid singularities
	def energy_per_atom(self, Z, Dij, Xia, Qa, idx_i, idx_j, Q_tot=None, batch_seg=None):
		if self.use_scaled_charges:
			Qa = self.scaled_charges(Z, Qa, Q_tot=Q_tot, batch_seg=batch_seg)
		Eae =self.eem.energy_per_atom(Z, Dij, Xia, Qa, idx_i, idx_j)
		return Eae, Qa

		
	def print_parameters(self):
		self.eem.print_parameters()

	@property
	def dtype(self):
		return self._dtype

	@property
	def use_scaled_charges(self):
		return self._use_scaled_charges

	@property
	def eem(self):
		return self._eem

	@property
	def num_outputs(self):
		return 3

