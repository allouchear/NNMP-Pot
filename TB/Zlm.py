import tensorflow as tf
from math import pi

from Utils.UtilsFunctions import *

"""
def factorial(n):
	return tf.exp(tf.math.lgamma(float(n+1)))

def binomial(a,b):
	return factorial(a)/factorial(b)/factorial(a-b)
"""


class Zlm:
	def __str__(self):
		lines ="Coefs of Z("
		lines = lines+str(self.l)+","+str(self.m)+")\n"
		lines = lines+"============================================\n"
		lines = lines+"Coefs\t\tlx\tly\tlz\n"
		for i in range (self.coefs.shape[0]):
			lines = lines+str(self.coefs[i].numpy())+"\t"+str(self.lx[i].numpy())+"\t"+str(self.ly[i].numpy())+"\t"+str(self.lz[i].numpy())+"\n"
		return lines

	def __init__(self, l, m, dtype=tf.float32):
		self._dtype=dtype
		if l==0 and m==0:
			self._set00()
		elif l<0:
			self._set00()
		else:
			self._l=l	
			self._m=m	
			self._setCoefs()

	def _set00(self):
		self._l=0
		self._m=0
		self._numberOfCoefficients=1
		self._coefs= tf.ones([1], dtype=self.dtype, name="ZlmCoefs")
		v = tf.constant(0.25/pi,dtype=self.dtype)
		self._coefs *= tf.math.sqrt(v)
		self._lx= tf.zeros([1], dtype=tf.int32, name="ZlmClx")
		self._ly= tf.zeros([1], dtype=tf.int32, name="ZlmCly")
		self._lz= tf.zeros([1], dtype=tf.int32, name="ZlmClz")
		return

	def _setCoefs(self):
		absm=tf.math.abs(self.m)
		factlpabs=tf.constant(factorial(self.l+absm),dtype=self.dtype)
		factlmabs=tf.constant(factorial(self.l-absm), dtype=self.dtype)
		factl=tf.constant(factorial(self.l), dtype=self.dtype)
		v = tf.constant((2*self.l+1)/(4*pi),dtype=self.dtype)
		fabsm=tf.cast(absm, dtype=self.dtype)
		Norm = tf.math.sqrt(v)*tf.math.sqrt(factlpabs*factlmabs)/factl/tf.math.pow(2.0,fabsm);
		if self.m != 0:
			 Norm *= tf.cast(tf.math.sqrt(2.0), dtype=self.dtype)
		coefs = []
		lx = []
		ly = []
		lz = []
		for t in range((self.l - absm)//2+1):
			for u in range(t+1):
				v2m=1
				if self.m>=0:
					v2m=0
				for v2 in range(v2m,absm+1,2):
					sign=1
					if (t + (v2-v2m)//2)%2:
						sign=-1
					tmp =  binomial(self.l,t)*binomial(self.l-t,absm+t)*binomial(t,u)*binomial(absm,v2)
					tmp = tmp/tf.math.pow(4.0,t)
					tmp = tf.cast(tmp, dtype=self.dtype)
					v = Norm*tmp*sign
					v=v.numpy()
					coefs.append(v)
					llx= (2*t + absm - 2*u - v2).numpy()
					lly= 2*u + v2
					llz= self.l-llx-lly
					lx.append(llx)
					ly.append(lly)
					lz.append(llz)
		# remove equivalent zlm : same lx,ly,lz
		ncoefs=len(coefs)
		ndel=[0]*ncoefs
		lxx =[]
		lyy =[]
		lzz =[]
		coefsxyz=[]
		for i in range(ncoefs):
			if ndel[i]==1:
				continue
			lxx.append(lx[i])
			lyy.append(ly[i])
			lzz.append(lz[i])
			c=coefs[i]
			for k in range(i+1,ncoefs):
				ok =1 # les 2 sont identiques
				if lx[i] != lx[k] or ly[i] != ly[k] or lz[i] != lz[k]:
					ok=0
					break
				if ok==1:
					c = c + coefs[k]
					ndel[k] = 1
			coefsxyz.append(c)

		self._coefs=tf.constant(coefsxyz,dtype=self._dtype)
		self._lx=tf.constant(lxx)
		self._ly=tf.constant(lyy)
		self._lz=tf.constant(lzz)
		return
		
	@property
	def l(self):
		return self._l
	@property
	def m(self):
		return self._m
	@property
	def coefs(self):
		return self._coefs
	@property
	def lx(self):
		return self._lx
	@property
	def ly(self):
		return self._ly

	@property
	def lz(self):
		return self._lz
	@property
	def dtype(self):
		return self._dtype
    
       
