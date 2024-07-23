import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class Coulomb(Layer):

	def __str__(self):
		return "Electrostatic/Coulomb"

	def __init__(self,
		sr_cut=None,
		lr_cut=None,
		use_scaled_charges=False,
		dtype=tf.float32            #single or double precision
		):
		super().__init__(dtype=dtype, name="Electrostatic/Coulomb")
		self._use_scaled_charges = use_scaled_charges
		self._dtype=dtype
		self._sr_cut=sr_cut
		self._lr_cut=lr_cut

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


	#calculates the electrostatic energy per atom 
	#for very small distances, the 1/r law is shielded to avoid singularities
	def energy_per_atom(self, Z, Dij, Qa, idx_i, idx_j, Q_tot=None, batch_seg=None):
		if self.use_scaled_charges:
			Qa = self.scaled_charges(Z, Qa, Q_tot=Q_tot, batch_seg=batch_seg)
		#gather charges
		Qi = tf.gather(Qa, idx_i)
		Qj = tf.gather(Qa, idx_j)
		#calculate variants of Dij which we need to calculate
		#the various shileded/non-shielded potentials
		DijS = tf.sqrt(Dij*Dij + 1.0) #shielded distance
		#calculate value of switching function
		switch = self._switch(Dij) #normal switch
		cswitch = 1.0-switch #complementary switch
		#calculate shielded/non-shielded potentials
		if self.lr_cut is None: #no non-bonded cutoff
			Eele_ordinary = 1.0/Dij   #ordinary electrostatic energy
			Eele_shielded = 1.0/DijS  #shielded electrostatic energy
			#combine shielded and ordinary interactions and apply prefactors 
			Eele = 0.5*Qi*Qj*(cswitch*Eele_shielded + switch*Eele_ordinary)
		else: #with non-bonded cutoff
			cut   = self.lr_cut
			cut2  = cut*cut
			Eele_ordinary = 1.0/Dij  +  Dij/cut2 - 2.0/cut
			Eele_shielded = 1.0/DijS + DijS/cut2 - 2.0/cut
			#combine shielded and ordinary interactions and apply prefactors 
			Eele = 0.5*Qi*Qj*(cswitch*Eele_shielded + switch*Eele_ordinary)
			Eele = tf.where(Dij <= cut, Eele, tf.zeros_like(Eele))
		Eele = tf.math.segment_sum(Eele, idx_i) 
		#print("Eelec=",Eele)
		return Eele , Qa

		
	def print_parameters(self):
		pass

	@property
	def dtype(self):
		return self._dtype

	@property
	def sr_cut(self):
		return self._sr_cut

	@property
	def lr_cut(self):
		return self._lr_cut

	@property
	def use_scaled_charges(self):
		return self._use_scaled_charges

	@property
	def num_outputs(self):
		return 2

