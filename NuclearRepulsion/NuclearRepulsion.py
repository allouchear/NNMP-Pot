import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
from .ZBL.ZBL import *
from .GFN.GFN1 import *

class NuclearRepulsion(Layer):

	def __str__(self):
		if self.model_type==None:
			return "Without Repulsion pot."
		else:
			return self.model_type+" Repultion pot."

	def __init__(self,
		model_type=0,			# repulsion model : 0=ZBL, 1=GFN1, other=None
		fit_parameters=0,		# fit parameters
		dtype=tf.float32            	# single or double precision
		):
		super().__init__(dtype=dtype, name="NuclearRepulsion")
		self._model_type = model_type
		self._model = None
		if model_type == 0:
			self._model_type = "ZBL" # Ziegler-Biersack-Littmark
			self._model = ZBL(fit_parameters=fit_parameters, dtype=dtype)
		elif model_type == 1:
			self._model_type = "GFN1"
			self._model =  GFN1(fit_parameters=fit_parameters, dtype=dtype)
		else:
			self._model_type = None # no nuclear replusion energy
			
	def energy_per_atom(self, Z, Dij, idx_i, idx_j):
		if self.model_type is not None: 
			return self.model.energy_per_atom(Z, Dij, idx_i, idx_j)
		return None

	def print_parameters(self):
		if self.model is not None:
			self.model.print_parameters()

	@property
	def model(self):
		return self._model

	@property
	def model_type(self):
		return self._model_type


