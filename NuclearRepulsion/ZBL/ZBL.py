from __future__ import absolute_import
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
#from Utils.PeriodicTable import *
from Utils.Aufbau import get_l_all_atoms

class ZBL(Layer):

	def __str__(self):
		st = "p="+str(self._p)+" \n" +"d="+str(self._d)+"\n"
		for k,ak in enumerate(a):
			st = st + "a["+str(k)+"]="+str(ak)+" "
		st = st + "\n"
		for k,ck in enumerate(a):
			st = st + "c["+str(k)+"]="+str(ck)+" "
		st = st + "\n"
		return st

	def __init__(self,
		fit_parameters=0,             # fit parameters (0 no, 1 fit  a&c, 2 fit p, d, 3 fit all)
		dtype=tf.float32            #single or double precision
		):
		super().__init__(dtype=dtype, name="ZBL")

		self._p = 0.23
		self._d = 0.885 # in Bohr
		self._c = np.array([0.1818, 0.5099, 0.2802, 0.02817])
		self._a = np.array([3.2000, 0.9423, 0.4028, 0.20160])
		self._dtype=dtype

		self._p = tf.Variable(self._p, name="ZBL/p", dtype=dtype,trainable=(fit_parameters==2 or fit_parameters==3))
		self._d = tf.Variable(self._d, name="ZBL/d", dtype=dtype,trainable=(fit_parameters==2 or fit_parameters==3))
		self._a = tf.Variable(self._a, name="ZBL/a", dtype=dtype,trainable=(fit_parameters==1 or fit_parameters==3))
		self._c = tf.Variable(self._c, name="ZBL/c", dtype=dtype,trainable=(fit_parameters==1 or fit_parameters==3))

	#calculates the nuclear repulsion energy per atom 
	#for very small distances, the 1/r law is shielded to avoid singularities
	def energy_per_atom(self, Z, Dij, idx_i, idx_j):
		Zi = tf.gather(Z, idx_i)
		Zj = tf.gather(Z, idx_j)
		Zi = tf.cast(Zi, dtype=Dij.dtype)
		Zj = tf.cast(Zj, dtype=Dij.dtype)

		pre = Zi*Zj/Dij
		expv = Dij/self.d*(Zi**self.p+Zj**self.p)
		vij = 0
		for i in range(self.c.shape[0]):
			vij += self.c[i] *tf.math.exp(-self.a[i]*expv)
		return tf.math.segment_sum(vij, idx_i)

	def print_parameters(self):
		print("p=",self.p.numpy())
		print("d=",self.d.numpy())
		print("c=",self.c.numpy())
		print("a=",self.a.numpy())

	@property
	def p(self):
		return self._p
	@property
	def d(self):
		return self._d

	@property
	def c(self):
		return self._c

	@property
	def a(self):
		return self._a
