from __future__ import absolute_import
import os
import tensorflow as tf
from math import pi
from .Zlm import *
from Utils.UtilsFunctions import *

"""
def doubleFactorial(n):
	#return tf.exp(tf.math.lgamma(float(n)/2+1.))*tf.math.sqrt(tf.math.pow(2.0,n+1.0)/pi)
	v=1
	if n%2==0:
		for i in range(2,n+1,2):
			v *= i 
	else:
		for i in range(3,n+1,2):
			v *= i 
	return v
"""

"""
def binomial(a,b,dtype=tf.float32):
	return tf.cast(factorial(a)/factorial(b)/factorial(a-b),dtype=dtype)
"""


def f(i,l,m,A,B,dtype=tf.float32):
	s=tf.Variable(0.0, dtype=dtype)
	jmin=0
	if jmin<i-m:
		jmin=i-m
	jmax=i
	if jmax>l:
		jmax=l
	for j in range(jmin,jmax+1):
		s = s+  binomial(l,j,dtype=dtype)*binomial(m,i-j,dtype=dtype)*tf.math.pow(-A,tf.cast(l-j,dtype=dtype))*tf.math.pow(-B,tf.cast(m-i+j,dtype=dtype))
	return s

class GTF:
	def __str__(self):
		lines=""
		lines = lines+"{:<14.8f}\t{:<14.8f}\tl={:2d},{:2d},{:2d} \n".format(self.coef.numpy(),self.ex.numpy(),self.l[0].numpy(),self.l[1].numpy(),self.l[2].numpy())
		return lines

	def __init__(self, ex, coef, l, R, dtype=tf.float32):
		self._dtype=dtype
		self._ex=ex*tf.constant(1.0,dtype=dtype)
		self._coef=coef*tf.constant(1.0,dtype=dtype)
		self._l=l
		self._R=R*tf.constant(1.0,dtype=dtype)

	def star(self, right, Rl=None, Rr=None):
		if Rl is None:
			Rl = self.R
		if Rr is None:
			Rr = right.R
		s = [0,0,0]
		PA = [0,0,0]
		PB = [0,0,0]
		gama=self.ex+right.ex
		R2=0
		c=0
		for j in range(3):
			t=(self.ex*Rl[j]+right.ex*Rr[j])/gama;
			PA[j] = Rl[j]-t 
			PB[j] = Rr[j]-t 
			R2 = R2 + (Rl[j]-Rr[j])*(Rl[j]-Rr[j])
		c = pi/gama*tf.math.sqrt(pi/gama)*tf.math.exp(-self.ex*right.ex/gama*R2)
		for j in range(3):
			for i in range((self.l[j]+right.l[j])//2+1):
				s[j] += f(2*i,self.l[j],right.l[j],PA[j],PB[j],dtype=self.dtype)*doubleFactorial(2*i-1)/tf.math.pow(2*gama,tf.cast(i,dtype=self.dtype))
		return c*s[0]*s[1]*s[2]

	def norme(self):
		num=2*self.ex/pi*tf.math.sqrt(2*self.ex/pi)*tf.math.pow(4*self.ex,float(self.l[0]+self.l[1]+self.l[2]))
		dnom=doubleFactorial(self.l[0])* doubleFactorial(self.l[1])*doubleFactorial(self.l[2])
		return tf.math.sqrt(num/dnom)

	def normalize(self):
		self._coef *= self.norme()

	def overlap(self, right, Rl=None, Rr=None):
		return self.coef*right.coef*self.star(right,Rl,Rr)

	def normalizeRadial(self):
		l=self.l[0]+self.l[1]+self.l[2]
		ll = [l,0,0]
		q = GTF(self.ex, self.coef,ll, self.R, dtype=self.dtype)
		n = q.norme()
		self.scalCoef(n)

	def scalCoef(self,s):
		self._coef *= s

	def setCoordinates(self,R):
		self._R=R*tf.constant(1.0,dtype=self.dtype)


	@property
	def dtype(self):
		return self._dtype

	@property
	def l(self):
		return self._l
	@property
	def coef(self):
		return self._coef
	@property
	def ex(self):
		return self._ex
	@property
	def R(self):
		return self._R

class CGTF:
	def __str__(self):
		lines = "{:<14s}\t{:<14s}\tl={:<2d},m={:<2d}\n".format("Coefs","Exp", self.l,self.m)
		for i in range (len(self.gtfs)):
			lines = lines+str(self.gtfs[i])
		lines += "--------------------------------------------\n"
		return lines

	def __init__(self, l, m, zeta, cont, R, dtype=tf.float32):
		self._dtype=dtype
		self._l=l
		self._m=m
		zlm = Zlm(l,m, dtype=dtype)
		self._gtfs = []
		for iz in range(len(zlm.coefs)):
			ll=[zlm.lx[iz], zlm.ly[iz],zlm.lz[iz]]
			for i in range(len(zeta)):
				gtf = GTF(zeta[i], cont[i], ll, R, dtype=self.dtype)
				gtf.normalizeRadial()
				gtf.scalCoef(zlm.coefs[iz])
				self._gtfs.append(gtf)
		self.normalize()

	def normalize(self):
		nf=len(self.gtfs)
		s = 0.0
		for n in range(nf):
			s += self.gtfs[n].coef*self.gtfs[n].coef*self.gtfs[n].star(self.gtfs[n])
		for n in range(nf-1):
			for np in range(n+1,nf):
				s += 2*self.gtfs[n].coef*self.gtfs[np].coef*self.gtfs[n].star(self.gtfs[np])


		try:
			s = 1.0/tf.math.sqrt(s)
			for n in range(nf):
				self.gtfs[n].scalCoef(s)
		except Exception as Ex:
                        print("A Contracted Gaussian Type function is nul.", Ex)
                        raise Ex

	def overlap(self, right,Rl=None, Rr=None):
		s = 0.0
		for n in range(len(self.gtfs)):
			for np in range(len(right.gtfs)):
				s += self.gtfs[n].overlap(right.gtfs[np],Rl,Rr)
		return s

	def setCoordinates(self,R):
		for gtf in self.gtfs:
			gtf.setCoordinates(R)

	@property
	def l(self):
		return self._l

	@property
	def m(self):
		return self._m

	@property
	def gtfs(self):
		return self._gtfs

	@property
	def dtype(self):
		return self._dtype
