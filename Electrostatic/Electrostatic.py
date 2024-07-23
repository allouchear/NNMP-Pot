import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from .EEM.EEM import *
from .EEMPot.EEMPot import *
from .GFN.GFN1 import *
from .Coulomb.Coulomb import *
from .Ewald.Ewald import *

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


class Electrostatic(Layer):

	def __str__(self):
		if self.model_type==None:
			return "Without Electrostatic"
		else:
			return self.model_type+" Electrostatic"


	def __init__(self,
		model_type=None,			# electristatic model : 0=Ewald/Coulomb, 1=Coulomb, 2=EEM, 3=EEMPot, 4=GFN1
		sr_cut=None,
                lr_cut=None,
		eem_fit_parameters=0,
		gfn_fit_parameters=0,
		orbfile=None,               	# read orbital for each atom type : Format : Z l1 l2 ...lmax for each Z
		use_scaled_charges=False,
		kmax=[2,2,2],                     # kmax for ewald sum
		dtype=tf.float32            	#single or double precision
		):
		super().__init__(dtype=dtype, name="Electrostatic")
		self._model_type = model_type
		self._model = None
		if model_type == "Ewald/Coulomb":
			self._model_type = model_type
			self._model = Ewald(cutoff=sr_cut, use_scaled_charges=use_scaled_charges, dtype=dtype, kmax=kmax)
		elif model_type == "Coulomb":
			self._model_type = model_type
			self._model = Coulomb(sr_cut=sr_cut, lr_cut=lr_cut, use_scaled_charges=use_scaled_charges, dtype=dtype)
		elif model_type == "EEM":
			self._model_type = model_type
			self._model =  EEM(fit_parameters=eem_fit_parameters,dtype=dtype)
		elif model_type == "EEMPot":
			self._model_type = model_type
			self._model =  EEMPot(fit_parameters=eem_fit_parameters,use_scaled_charges=use_scaled_charges, dtype=dtype)
		elif model_type == "GFN1":
			self._model_type = model_type
			self._model =  GFN1(fit_parameters=gfn_fit_parameters,dtype=dtype, orbfile=orbfile)
		else:
			self._model_type = "None" # no electrostatic energy
			
	#calculates charge per atom 
	def compute_atomic_charges(self, Z, R, Xia=None, Qa=None, Q_tot=None, batch_seg=None):
		if self.model_type == "Coulomb":
			return Qa
		if self.model_type == "Ewald/Coulomb":
			return Qa
		if self.model_type == "EEM":
			return self.model.compute_atomic_charges(Z,R, Xia, Q_tot=Q_tot, batch_seg=batch_seg)
		if self.model_type == "EEMPot":
			return Qa
		if self.model_type == "GFN1":
			return self.model.compute_atomic_charges(Z, R, Qa) # Qa = Qa[:,l]

	#calculates the electrostatic energy per atom and scaled charges if required
	def energy_per_atom(self, Z=None, R=None, Dij=None, Xia=None, Qa=None, idx_i=None, idx_j=None, Q_tot=None, batch_seg=None,Cell=None):
		if self.model_type == "Coulomb":
			return self.model.energy_per_atom(Z, Dij, Qa, idx_i, idx_j, Q_tot=Q_tot, batch_seg=batch_seg)
		if self.model_type == "Ewald/Coulomb":
			return self.model.energy_per_atom(Z, Dij, Qa, idx_i, idx_j, Q_tot=Q_tot, batch_seg=batch_seg,R=R,Cell=Cell)
		if self.model_type == "EEM":
			return self.model.compute_atomic_charges_and_energies(Z,R, Xia, Q_tot=Q_tot, batch_seg=batch_seg)
		if self.model_type == "EEMPot":
			return self.model.energy_per_atom( Z, Dij, Xia, Qa, idx_i, idx_j, Q_tot=Q_tot, batch_seg=batch_seg)
		if self.model_type == "GFN1":
			return self.model.energy_per_atom(Z, R, Dij, Qa, idx_i, idx_j, Q_tot=Q_tot, batch_seg=batch_seg)

	def print_parameters(self):
		if self.model is not None:
			self.model.print_parameters()

	@property
	def model(self):
		return self._model

	@property
	def model_type(self):
		return self._model_type

	@property
	def num_outputs(self):
		return self._model._num_outputs


