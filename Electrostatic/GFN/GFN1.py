from __future__ import absolute_import
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
#from Utils.PeriodicTable import *
from Utils.Aufbau import get_l_all_atoms

class GFN1(Layer):

	def __str__(self):
		return str(self._l)+" \n" +str(self._J)+"\n"

	def __init__(self,
		use_scaled_charges=False,
		fit_parameters=0,             # fit parameters (0 no, 1 fit eta, 2 fit gamma, 3 fit eta & gamma)
		orbfile=None,                     # read orbital for each atom type : Format : Z l1 l2 ...lmax for each Z
		dtype=tf.float32            #single or double precision
		):
		super().__init__(dtype=dtype, name="GFN1")

		l,nl = self._get_l_parameters(orbfile)
		self._num_outputs=1+nl
		self._l=l
		self._nl=nl

		self._use_scaled_charges = use_scaled_charges
		self._dtype=dtype

		zmax=len(l)

		norbs=[]
		norbmax=0
		for z in range(0,zmax):
			norbs += [len(l[z])] 
			if norbmax<len(l[z]):
				norbmax=len(l[z])

		ll=[]
		for z in range(0,zmax):
			lll = l[z]
			if len(l[z]) <norbmax:
				lll += [-1]*(norbmax-len(l[z]))
			ll += [ lll ]

		self._eta = tf.Variable(tf.ones(shape=[zmax,norbmax],dtype=dtype), name="GFN1/eta", dtype=dtype,trainable=(fit_parameters==1 or fit_parameters==3))
		self._gamma = tf.Variable(tf.ones(shape=[zmax,norbmax],dtype=dtype), name="GFN1/gamma", dtype=dtype,trainable=(fit_parameters==2 or fit_parameters==3))
		self._norbs = tf.Variable(norbs, name="GFN1/norbs", dtype=dtype,trainable=False)
		self._l = tf.Variable(ll, name="GFN1/l", dtype=dtype,trainable=False) # not trainable, can be save

	#returns scaled charges such that the sum of the partial atomic charges equals Q_tot (defaults to 0)
	def scaled_charges(self, Z, Qal, Q_tot=None, batch_seg=None):
		if batch_seg is None:
			batch_seg = tf.zeros_like(Z)
		#number of atoms per batch (needed for charge scaling)
		Na_per_batch = tf.math.segment_sum(tf.ones_like(batch_seg, dtype=self.dtype), batch_seg)
		if Q_tot is None: #assume desired total charge zero if not given
			Q_tot = tf.zeros_like(Na_per_batch, dtype=self.dtype)
		#return scaled charges (such that they have the desired total charge)
		Qa = self.compute_atomic_charges(Z, R, Qal)
		dQa = tf.gather(((Q_tot-tf.math.segment_sum(Qa, batch_seg))/Na_per_batch), batch_seg)
		dQa = dQa/norbsa
		dQi = tf.gather(dQa, idx_i)
		dQj = tf.gather(dQa, idx_j)
		QaScaled = Qa + dQa
		return QaScaled

		
	def compute_atomic_charges(self, Z, R, Qal):
		norbsa =  tf.gather(self.norbs, Z)
		nout=Qal.shape[1]
		Qa = 0
		for l in  range(nout):
			fact=tf.cast(tf.where(norbsa>=l, 1.0, 0.0),dtype=self.dtype)
			Qa += fact*Qal[:,l]

		return Qa

	#calculates the electrostatic energy per atom 
	#for very small distances, the 1/r law is shielded to avoid singularities
	# Qa[i] is a vector containg Qa[z][l0], Qa[z][l1], Qa[z][l2], ...
	def energy_per_atom(self, Z, R, Dij, Qal, idx_i, idx_j, Q_tot=None, batch_seg=None):
		#Xia =  tf.gather(self.Xi, Z)
		#gather charges
		norbsa =  tf.gather(self.norbs, Z)
		eta_a =  tf.gather(self.eta, Z)
		gamma_a =  tf.gather(self.gamma, Z)

		dQi = 0
		dQj = 0
		QaScaled = None
		if self.use_scaled_charges:
			QaScaled = self.scaled_charges(Z, Qal, Q_tot=Q_tot, batch_seg=batch_seg)

		nout=Qal.shape[1]
		for l in  range(nout):
			fact=tf.cast(tf.where(norbsa>=l, 1.0, 0.0),dtype=self.dtype)
			if l==0:
				Ql = [fact*Qal[:,l]]
				eta_l = [eta_a[:,l]]
				gamma_l = [gamma_a[:,l]]
			else:
				Ql = tf.concat([Ql,[fact*Qal[:,l]]],0)
				eta_l = tf.concat([eta_l,[eta_a[:,l]]],0)
				gamma_l = tf.concat([gamma_l,[gamma_a[:,l]]],0)

		Eelei=0.0
		for l in  range(nout):
				gamma_i = tf.gather(gamma_l[l], idx_i)
				Qi = tf.gather(Ql[l], idx_i)
				Qi += dQi
				Eelei += 1.0/3.0*gamma_i*Qi

		Eeleij=0.0
		for l in  range(nout):
			for lp in  range(nout):
				eta_i = tf.gather(eta_l[l], idx_i)
				eta_j = tf.gather(eta_l[l], idx_j)
				Qi = tf.gather(Ql[l], idx_i)
				Qi += dQi
				Qj = tf.gather(Ql[lp], idx_j)
				Qj += dQj
				etam2=0.5*(eta_i+eta_j)
				etam2=1/(etam2*etam2)
				r2etam2=1/(Dij+etam2)
				Eeleij += 0.5*Qi*Qj*r2etam2


		Eleci=tf.math.segment_mean(Eelei, idx_i)
		Eeleij=tf.math.segment_sum(Eeleij, idx_i) 

		#print("Elec",Eleci+Eeleij)
		return Eleci+Eeleij, QaScaled

	def print_parameters(self):
		#print("Alpha=",end='')
		#[ print("(",i,",", self.alpha.numpy()[i],"), ",end='') for i in range(1,len(self.alpha.numpy())) ]
		#print("")
		print("eta=",self.eta.numpy()[self.eta.numpy()>0])
		print("gamma=",self.gamma.numpy()[self.gamma.numpy()>0])
		print("l=",self.l.numpy()[self.l.numpy()>0])
		print("nobrs=",self.norbs.numpy()[1:])
		print("eta & gamma for Z=",[ i for i in range(1,len(self.eta.numpy())+1) ])

	def _get_l_parameters(self,orbfile):
		l,nla = get_l_all_atoms (95)
		if orbfile is not None:
			try:
				f=open(orbfile,"r")
				lines=f.readlines()
				nlines=len(lines)
				self.R = tf.zeros(shape=(natoms,3),dtype=Rtype)
				for i in range(nlines):
					line = lines[i].split()
					if len(line)<1 :
						continue
					z = line[0]	
					if z<1:
						continue
					nl=len(line)-1
					if nl<1:
						continue
					if nl>nla:
						nla=nl
					ll = []
					for lll in range(1,nl+1):
						ll.append(lll)
					l[z]=ll
				f.close()
			except Exception as Ex:
				print("Read Failed.", Ex)
				raise Ex
		return l,nla
	


	@property
	def eta(self):
		return self._eta
	@property
	def gamma(self):
		return self._gamma

	@property
	def l(self):
		return self._l

	@property
	def norbs(self):
		return self._norbs

	@property
	def dtype(self):
		return self._dtype

	@property
	def use_scaled_charges(self):
		return self._use_scaled_charges

	@property
	def num_outputs(self):
		return self._num_outputs

	@property
	def l(self):
		return self._l

	@property
	def nl(self):
		return self._nl
