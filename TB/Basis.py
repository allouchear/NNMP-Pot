from __future__ import absolute_import
import os
import math
import numpy as np
import tensorflow as tf
from .GTF import *
from tensorflow.keras.layers import Layer
from Utils.PeriodicTable import *

class Basis:
	def __str__(self):
		lines=""
		for i in range (len(self.cgtfs)):
			lines = lines+str(self.cgtfs[i])
		return lines

	def __init__(self,
			l = None,
			m= None,
			zeta=None,
			cont=None,
			R=None,
			dtype=tf.float32            #single or double precision
		):
		self._dtype=dtype
		self._cgtfs= None
		if not (l is None and m is None and zeta is None and cont is None):
			self.add(l, m, zeta, cont, R)

	def add(self, l, m, zeta, cont, R=None):
		if R is None:
			R=[0.0,0.0,0.0]
		cgtf=CGTF( l, m, zeta, cont, R, dtype=self.dtype)
		if self._cgtfs is None:
			self._cgtfs=[]
		self._cgtfs.append(cgtf)

	def scalCoordinates(self,R):
		for cgtf in self.cgtfs:
			cgtf.scalCoordinates(R)

	@property
	def cgtfs(self):
		return self._cgtfs

	@property
	def dtype(self):
		return self._dtype

def lorb (s):
	if (s[0]) == "s" or (s[0]) == "S":
                l=0
	elif (s[0]) == "p" or (s[0]) == "P":
                l=1
	elif (s[0]) == "d" or (s[0]) == "D":
                l=2
	elif (s[0]) == "f" or (s[0]) == "F":
                l=3
	elif (s[0]) == "g" or (s[0]) == "G":
                l=4
	elif (s[0]) == "h" or (s[0]) == "H":
                l=5
	else:
		print("Error : unknown ",s, " atomic orbital\n")
		exit(1)
	return l

class BasisSet(Layer):
	def __str__(self):
		lines=""
		for i in range (len(self.basis)):
			if self.basis[i] is not None:
				symbol=self._periodicTable.elementZ(i).symbol
				lines = lines+symbol+"(Z="+str(i)+")\n"
				lines = lines+"=========="+"\n"
				lines = lines+str(self.basis[i])
		lines = lines+"============================================\n"
		return lines

	def __init__(self,
		fit_parameters=0,           # fit parameters (0 no, 1 fit Coefs, 2 fit alp, 3 fit coefs & alp
		dtype=tf.float32            #single or double precision
		):
		super().__init__(dtype=dtype, name="BasisSet")
		self._basis = [None]*95 # 95 atoms : z=0,1,2,....
		self._periodicTable = PeriodicTable()
		self._dtype=dtype

	def add(self, symbol,  l, m, zeta, cont, R=None):
		z=self._periodicTable.element(symbol).atomicNumber
		if self._basis[z] is None:
			self._basis[z] = Basis(l,m,zeta,cont,R, dtype=self.dtype)
		else:
			self._basis[z].add(l,m,zeta,cont,R)

	def readFrom(self,path):
		try:
			f=open(path,"r")
			io=0
			no=0
			for line in f.readlines():
				ll = line.split()
			
				# new atom
				if len(ll)==1:
					symbol = ll[0]
				# new cgtf
				elif len(ll)==2 and float(ll[1]).is_integer():
					l=lorb(ll[0])
					no = int(ll[1])
					io=0
					zetas =[]
					coefs =[]
				elif len(ll)==2:
					ll[0] = tf.constant( float(ll[0]), dtype=self.dtype)
					ll[1] = tf.constant( float(ll[1]), dtype=self.dtype)
					io = io+1
					zetas.append(ll[0])
					coefs.append(ll[1])
					if io==no:
						for m in range(-l,l+1):
							self.add(symbol,l,m,zetas,coefs)
						no=0
						io=0
			f.close()
		except Exception as Ex:
			print("Read Failed.", Ex)
			raise Ex
		return
	def getBasis(self,z):
		return self.basis[z]

	def overlap(self,Z,R):
		nAtoms = Z.shape[0]

		nBasis=0
		for i in range(nAtoms):
			nBasis = nBasis + len(self.basis[Z[i]].cgtfs)

		S = np.ones( (nBasis, nBasis) )
		ki=-1
		for i in range(nAtoms):
			cgtfsi = self.getBasis(Z[i]).cgtfs
			for ib in range(len(cgtfsi)):
				bi = cgtfsi[ib]
				ki=ki+1
				S[ki][ki] =   bi.overlap(bi,R[i],R[i])
				kj=ki
				for iib in range(ib+1,len(cgtfsi)):
					bj = cgtfsi[iib]
					kj=kj+1
					S[ki][kj] =   S[kj][ki]  =  bi.overlap(bj, R[i],R[i])

				for j in range(i+1,nAtoms):
					cgtfsj = self.getBasis(Z[j]).cgtfs
					for bj in cgtfsj:
						kj=kj+1
						S[ki][kj] =   S[kj][ki]  =  bi.overlap(bj,R[i],R[j])
		return S

	@property
	def dtype(self):
		return self._dtype
	@property
	def basis(self):
		return self._basis

