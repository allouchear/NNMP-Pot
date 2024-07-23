import tensorflow as tf
import numpy as np
import math 
from tensorflow.keras.layers import Layer
from Utils.UtilsFunctions import *

#inverse softplus transformation
def softplus_inverse(x):
    return x + tf.math.log(-tf.math.expm1(-x))

# basis function expansion
class BFLayer(Layer):
	def __str__(self):
		return "basis_function_layer"+super().__str__()

	def __init__(self, K, cutoff, Lmax=2, beta=0.2, name=None, basis_type="Default", dtype=tf.float32):
		super().__init__(dtype=dtype)
		#basis_type="Bessel"
		#basis_type="Slater"
		#basis_type="GaussianNet"
		self._K = K
		self._Lmax = Lmax
		self._cutoff = cutoff
		self._dtype = dtype
		if basis_type=="Default":
			basis_type="Gaussian"
		self._basis_type = basis_type
		#initialize centers
		#print("K=",K)
		#print("cuttof=",cutoff)
		#print("dtype=",dtype)
		#print("linspace=",tf.linspace(tf.constant(0.0,dtype=dtype),cutoff,K))
		if self.basis_type=="Gaussian":
			centers = tf.cast(tf.linspace(tf.constant(0.0,dtype=dtype),cutoff,K),dtype=dtype)
			self._centers = tf.Variable(centers, name=name+"centers", dtype=dtype, trainable=False)
			tf.summary.histogram("rbf_centers", self.centers) 
			delta_rs =  tf.constant(cutoff,dtype=self.dtype)/(K-1)
			alpha = beta/(delta_rs**2)
			widths = [alpha]*K
			self._widths = tf.Variable(widths,  name=name+"widths",  dtype=dtype, trainable=False)
			tf.summary.histogram("rbf_widths", self.widths)
		elif self.basis_type=="GaussianNet":
			centers = softplus_inverse(tf.linspace(tf.constant(1.0,dtype=dtype),tf.math.exp(-cutoff),K))
			self._centers = tf.nn.softplus(tf.Variable(centers, name=name+"centers", dtype=dtype, trainable=False))
			tf.summary.histogram("rbf_centers", self.centers) 
			#initialize widths (inverse softplus transformation is applied, such that softplus can be used to guarantee positive values)
			widths = [softplus_inverse((0.5/((1.0-tf.math.exp(-cutoff))/K))**2)]*K
			self._widths = tf.nn.softplus(tf.Variable(widths,  name=name+"widths",  dtype=dtype, trainable=False))
			tf.summary.histogram("rbf_widths", self.widths)

		elif self.basis_type=="Slater": # centers at origin 
			K=int(tf.sqrt(K*1.0).numpy())
			alphas=tf.linspace(tf.constant(1.0,dtype=dtype),K,K)
			alphas=alphas.numpy().tolist()*K
			n=tf.linspace(tf.constant(1.0,dtype=dtype),K,K)
			n=tf.repeat(n,repeats=K)
			ccut=tf.cast(cutoff,dtype=dtype)
			self._alphas = tf.Variable(alphas, name=name+"alphas", dtype=dtype, trainable=False) # r**(n-1)*exp(-alphas*(rij/rc))
			self._n = tf.Variable(n, name=name+"n", dtype=dtype, trainable=False) 
			K=K*K
			self._K=K
		else: # bessel, radial = sin ( Johannes Klicpera et al., https://arxiv.org/pdf/2003.03123.pdf)
			n=tf.linspace(tf.constant(1.0,dtype=dtype),K,K)
			#print("n=",n)
			ccut=tf.cast(cutoff,dtype=dtype)
			alphas = n*math.pi/ccut # n pi/cutoff => normc*sin(alpha*rij)/rij
			self._alphas = tf.Variable(alphas, name=name+"alphas", dtype=dtype, trainable=False)
			normc=tf.math.sqrt(2.0/ccut)
			self._normc = tf.Variable(normc, name=name+"normc", dtype=dtype, trainable=False)

		L =[]
		lx =[]
		ly =[]
		lz =[]
		fL =[]
		for l in range(Lmax+1):
			for x in range(l+1):
				for y in range(l-x+1):
					z = l -x -y
					L.append(l)
					lx.append(x)
					ly.append(y)
					lz.append(z)
					factL=factorial(l)/factorial(x)/factorial(y)/factorial(z)
					factL=tf.math.sqrt(factL) # sqrt => included in the sum over j
					fL.append(factL)
		self._L = tf.Variable(L,  name=name+"L",  dtype=tf.int32, trainable=False)
		self._lx = tf.Variable(lx,  name=name+"lx",  dtype=tf.int32, trainable=False)
		self._ly = tf.Variable(ly,  name=name+"ly",  dtype=tf.int32, trainable=False)
		self._lz = tf.Variable(lz,  name=name+"lz",  dtype=tf.int32, trainable=False)
		self._fL = tf.Variable(fL,  name=name+"sqrt(Factorial_L/lx/ly/lz)",  dtype=dtype, trainable=False)
		#atom embeddings (we go up to Pu(94), 95 because indices start with 0)
		#coefs = np.ones((95,95,Lmax+1,K))
		coefs = np.ones((95,95,len(L),K))
		coefs = coefs*1e-2
		#print("coefs=",coefs)
		self._coefs = tf.Variable(coefs, name=name+"expansion coefficients", dtype=dtype,trainable=True)
		#self._coefs = tf.Variable(tf.ones([95],dtype=dtype) , name=name+"expansion coefficients", dtype=dtype,trainable=True)
		#self._coefs = tf.Variable(tf.cast(range(95),dtype=dtype) , name=name+"expansion coefficients", dtype=dtype,trainable=False)
		"""
		print("L=",self.L)
		print("lx=",self.lx)
		print("ly=",self.ly)
		print("lz=",self.lz)
		print("fL=",self.fL)
		"""


	@property
	def K(self):
		return self._K

	@property
	def cutoff(self):
		return self._cutoff
    
	@property
	def centers(self):
		return self._centers   

	@property
	def widths(self):
		return self._widths  

	@property
	def Lmax(self):
		return self._Lmax 

	@property
	def L(self):
		return self._L  

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
	def fL(self):
		return self._fL

	@property
	def coefs(self):
		return self._coefs

	@property
	def alphas(self):
		return self._alphas

	@property
	def n(self):
		return self._n

	@property
	def normc(self):
		return self._normc

	@property
	def basis_type(self):
		return self._basis_type

	#cutoff function that ensures a smooth cutoff
	def cutoff_fn(self, D):
		x = D/self.cutoff
		x3 = x**3
		x4 = x3*x
		x5 = x4*x
		return tf.where(x < 1, 1 - 6*x5 + 15*x4 - 10*x3, tf.zeros_like(x))

	def cutoff_fncos(self, D):
		x = D/self.cutoff
		return tf.where(x < 1, (0.5*(1.0+tf.math.cos(math.pi*x)))**2, tf.zeros_like(x))

	def radial(self, rij):
		if self.basis_type=="Gaussian":
			v = tf.exp(-self.widths*(rij-self.centers)**2) # Gaussian
			#v = tf.exp(-self.widths*tf.sqrt((rij-self.centers)**2)) # Slater
			v *= self.cutoff_fncos(rij)
			return v;
		elif self.basis_type=="GaussianNet":
			v = tf.exp(-self.widths*(tf.exp(-rij)-self.centers)**2)
			v *= self.cutoff_fn(rij)
			return v;
		elif self.basis_type=="Slater":
			x = rij/self.cutoff
			v = x**self.n*tf.exp(-self.alphas*x) # Gaussian
			v *= self.cutoff_fncos(rij)
			return v;
		else:
			v = self.normc*tf.math.sin(self.alphas*rij)/rij #  From Bessel
			v *= self.cutoff_fncos(rij)
			return v;

	def _computeEABF(self, Z, R, idx_i, idx_j, offsets=None):
		Ri = tf.gather(R, idx_i)
		Rj = tf.gather(R, idx_j)
		if offsets is not None:
                        Rj += offsets
		rij = tf.sqrt(tf.reduce_sum((Ri-Rj)**2, -1))
		Rit=tf.transpose(Ri)
		Rjt=tf.transpose(Rj)
		xij = Rjt[0]-Rit[0]
		yij = Rjt[1]-Rit[1]
		zij = Rjt[2]-Rit[2]
		Zi = tf.gather(Z, idx_i)
		Zj = tf.gather(Z, idx_j)

		rij = tf.expand_dims(rij, -1)
		xij = tf.expand_dims(xij, -1)
		yij = tf.expand_dims(yij, -1)
		zij = tf.expand_dims(zij, -1)
		radialv = self.radial(rij)
		"""
		print("Ri=",Ri)
		print("Rj=",Rj)
		print("xij=",xij)
		print("yij=",yij)
		print("zij=",zij)
		print("rij=",rij)
		print("radialv=",radialv)
		"""
		ss = tf.stack ([Zi, Zj], axis = 0) # configuration [[0,1,2,3], [2,3,4,5]]
		#print("ss=",ss)
		indexs = tf.unstack (ss, axis = 1) # configuration [[0,2], [1,3], [2,4], [3,5]]
		#print("idx=",indexs)
		coefs_ij=tf.gather_nd(self.coefs,indexs)
		#print("coefsi=",coefs_ij)
		coefs_ij=tf.transpose(coefs_ij,perm=[1,0,2])
		#print("tcoefsi=",coefs_ij)
		#print("coefsij=",coefs_ij)
		bf = []
		for i in range(self.L.shape[0]):
			ex = xij**tf.cast(self.lx[i],dtype=self.dtype)
			ey = yij**tf.cast(self.ly[i],dtype=self.dtype)
			ez = zij**tf.cast(self.lz[i],dtype=self.dtype)
			"""
			print("ex=",ex)
			print("ey=",ey)
			print("ez=",ez)
			print("fL=",self.fL[i])
			print("l=",tf.cast(self.L[i],dtype=self.dtype))
			print("lx=",tf.cast(self.lx[i],dtype=self.dtype))
			print("ly=",tf.cast(self.ly[i],dtype=self.dtype))
			print("lz=",tf.cast(self.lz[i],dtype=self.dtype))
			"""
			
			#print("radialv=",radialv)
			#print("coeds=",coefs_ij[self.L[i]])
			#v = self.fL[i]*ex*ey*ez*radialv*coefs_ij[self.L[i]]
			v = self.fL[i]*ex*ey*ez*radialv*coefs_ij[i]
			#print("v=",v)
			v=tf.math.segment_sum(v, idx_i)
			#print("vrsum=",v)
			vv=v*v
			#print("v2=",vv)
			#print("v2t=",tf.transpose(vv))
			bf.append(tf.transpose(vv))
		bf = tf.convert_to_tensor(bf, dtype=self.dtype)
		#bf = tf.Variable(bf, dtype=self.dtype)
		#print("bf=",bf)
		bfred=tf.math.segment_sum(bf, self.L)
		#print("bfred=",bfred)
		bfred=tf.reshape(bfred,[-1,bfred.shape[2]])
		#print("bfredrefsape=",bfred)
		bfred=tf.transpose(bfred)
		#print("bftranspose=",bfred)
		#print("transpose redusum, seg L =", bfred)
		rij = tf.reshape(rij,-1)
		#print("rij=",rij)
		return bfred,rij

    
	def __call__(self, Z, R, idx_i, idx_j, offsets=None):
		# Phi_i(L,alpha) : One by atom
		bfByAtom, rij = self._computeEABF(Z, R, idx_i, idx_j, offsets=offsets)
		#print("bfByAtom=",bfByAtom)
		bfi = tf.gather(bfByAtom, idx_i)
		bfj = tf.gather(bfByAtom, idx_j)
		#print("bfi=",bfi)
		#print("bfj=",bfj)

		bfij=bfi*bfj # one basis for each (i,j), Size = len(idx_i)*number of basis of each atom
		#print("bfij=",bfij)
		return bfij,rij
