from __future__ import absolute_import
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class GFN1(Layer):

	def __str__(self):
		return str(self._l)+" \n" +str(self._J)+"\n"

	def __init__(self,
		fit_parameters=0,             # fit parameters (0 no, 1 fit eta, 2 fit gamma, 3 fit eta & gamma)
		dtype=tf.float32            #single or double precision
		):
		super().__init__(dtype=dtype, name="GFN1")

		self._dtype=dtype
		zmax = 95

		self._alpha = tf.Variable(tf.ones(shape=[zmax],dtype=dtype), name="GFN1/alpha", dtype=dtype,trainable=(fit_parameters==1 or fit_parameters==3))
		self._alpha  = self._alpha * 1e-6
		self._Zeff = tf.Variable(tf.ones(shape=[zmax],dtype=dtype), name="GFN1/Zeff", dtype=dtype,trainable=(fit_parameters==2 or fit_parameters==3))

	#calculates repulsion energy per atom 
	#calculates the nuclear repulsion energy per atom 
	#for very small distances, the 1/r law is shielded to avoid singularities
	def energy_per_atom(self, Z, Dij, idx_i, idx_j):
		Zi = tf.gather(Z, idx_i)
		Zj = tf.gather(Z, idx_j)
		alphai =  tf.abs(tf.gather(self.alpha, Zi))
		alphaj =  tf.abs(tf.gather(self.alpha, Zj))
		Zeff_i =  tf.abs(tf.gather(self.Zeff, Zi))
		Zeff_j =  tf.abs(tf.gather(self.Zeff, Zj))

		vij = Zeff_i*Zeff_j/Dij*tf.math.exp(-tf.math.sqrt(alphai*alphaj*Dij**3))
		return tf.math.segment_sum(vij, idx_i)

	def print_parameters(self):
		print("alpha=",self.alpha.numpy())
		print("Zeff=",self.Zeff.numpy())

	@property
	def alpha(self):
		return self._alpha
	@property
	def Zeff(self):
		return self._Zeff

