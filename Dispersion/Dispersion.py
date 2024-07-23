import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
from .GrimmeD3.GrimmeD3 import *

def softplus_inverse(x):
	'''numerically stable inverse of softplus transform'''
	return x + np.log(-np.expm1(-x))


class Dispersion(Layer):
	def __str__(self):
		return "Dispersion"

	def __init__(self,
		dP=None,
		cutoff=None, # long range cutoff
		dtype=tf.float32            #single or double precision
		):
		super().__init__(dtype=dtype, name="Dispersion")
		self._gD3=None
		self._cutoff = cutoff
		self._set_dispersion_parameters(dP)

	def energy(self, Z, Dij, idx_i, idx_j):
		if self.cutoff is not None:   
			return self.gD3.edisp(Z, Dij, idx_i, idx_j, s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2, cutoff=self.cutoff)
		else:
			return self.gD3.edisp(Z, Dij, idx_i, idx_j, s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2)

	@property
	def gD3(self):
		return self._gD3
	@property
	def cutoff(self):
		return self._cutoff

	@property
	def s6(self):
		return self._s6
	@property
	def s8(self):
		return self._s8
	@property
	def a1(self):
		return self._a1
	@property
	def a2(self):
		return self._a2

	def _set_dispersion_parameters(self, dP):
		#initialize variables for d3 dispersion (the way this is done, positive values are guaranteed)
		self._gD3=GrimmeD3()
		if dP is None or dP[0] is None:
		    self._s6 = tf.Variable(tf.nn.softplus(softplus_inverse(self.gD3.d3_s6)), name="s6", dtype=self.dtype, trainable=True)
		else:
		    self._s6 = tf.Variable(dP[0], name="s6", dtype=self.dtype, trainable=False)
		tf.summary.scalar("d3-s6", self._s6)

		if dP is None or dP[1] is None:
		    self._s8 = tf.Variable(tf.nn.softplus(softplus_inverse(self.gD3.d3_s8)), name="s8", dtype=self.dtype, trainable=True)
		else:
		    self._s8 = tf.Variable(dP[1], name="s8", dtype=self.dtype, trainable=False)
		tf.summary.scalar("d3-s8", self._s8)

		if dP is None or dP[2] is None:
		    self._a1 = tf.Variable(tf.nn.softplus(softplus_inverse(self.gD3.d3_a1)), name="a1", dtype=self.dtype, trainable=True)
		else:
		    self._a1 = tf.Variable(dP[2], name="a1", dtype=self.dtype, trainable=False)
		tf.summary.scalar("d3-a1", self._a1)

		if dP is None or dP[3] is None:
		    self._a2 = tf.Variable(tf.nn.softplus(softplus_inverse(self.gD3.d3_a2)), name="a2", dtype=self.dtype, trainable=True)
		else:
		    self._a2 = tf.Variable(dP[3], name="a2", dtype=self.dtype, trainable=False)
		tf.summary.scalar("d3-a2", self._a2)

	def print_parameters(self):
		#initialize variables for d3 dispersion (the way this is done, positive values are guaranteed)
		if( self._s6 is not None):
		    print("Grimme/s6=",self._s6.numpy())
		if( self._s8 is not None):
		    print("Grimme/s8=",self._s8.numpy())
		if( self._a1 is not None):
		    print("Grimme/a1=",self._a1.numpy())
		if( self._a2 is not None):
		    print("Grimme/a2=",self._a2.numpy())

