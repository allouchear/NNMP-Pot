from __future__ import absolute_import
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class EEM(Layer):
	xi=[0,
	1.23695041, 1.26590957, 0.54341808, 0.99666991, 1.26691604, 
	1.40028282, 1.55819364, 1.56866440, 1.57540015, 1.15056627, 
	0.55936220, 0.72373742, 1.12910844, 1.12306840, 1.52672442, 
	1.40768172, 1.48154584, 1.31062963, 0.40374140, 0.75442607, 
	0.76482096, 0.98457281, 0.96702598, 1.05266584, 0.93274875, 
	1.04025281, 0.92738624, 1.07419210, 1.07900668, 1.04712861, 
	1.15018618, 1.15388455, 1.36313743, 1.36485106, 1.39801837, 
	1.18695346, 0.36273870, 0.58797255, 0.71961946, 0.96158233, 
	0.89585296, 0.81360499, 1.00794665, 0.92613682, 1.09152285, 
	1.14907070, 1.13508911, 1.08853785, 1.11005982, 1.12452195, 
	1.21642129, 1.36507125, 1.40340000, 1.16653482, 0.34125098, 
	0.58884173, 0.68441115, 0.56999999, 0.56999999, 0.56999999, 
	0.56999999, 0.56999999, 0.56999999, 0.56999999, 0.56999999, 
	0.56999999, 0.56999999, 0.56999999, 0.56999999, 0.56999999, 
	0.56999999, 0.87936784, 1.02761808, 0.93297476, 1.10172128, 
	0.97350071, 1.16695666, 1.23997927, 1.18464453, 1.14191734, 
	1.12334192, 1.01485321, 1.12950808, 1.30804834, 1.33689961, 
	1.27465977]

	J=[ 0,
	-0.35015861, 1.04121227, 0.09281243, 0.09412380, 0.26629137, 
	0.19408787, 0.05317918, 0.03151644, 0.32275132, 1.30996037, 
	0.24206510, 0.04147733, 0.11634126, 0.13155266, 0.15350650, 
	0.15250997, 0.17523529, 0.28774450, 0.42937314, 0.01896455, 
	0.07179178,-0.01121381,-0.03093370, 0.02716319,-0.01843812, 
	-0.15270393,-0.09192645,-0.13418723,-0.09861139, 0.18338109, 
	0.08299615, 0.11370033, 0.19005278, 0.10980677, 0.12327841, 
	0.25345554, 0.58615231, 0.16093861, 0.04548530,-0.02478645, 
	0.01909943, 0.01402541,-0.03595279, 0.01137752,-0.03697213, 
	0.08009416, 0.02274892, 0.12801822,-0.02078702, 0.05284319, 
	0.07581190, 0.09663758, 0.09547417, 0.07803344, 0.64913257, 
	0.15348654, 0.05054344, 0.11000000, 0.11000000, 0.11000000, 
	0.11000000, 0.11000000, 0.11000000, 0.11000000, 0.11000000, 
	0.11000000, 0.11000000, 0.11000000, 0.11000000, 0.11000000, 
	0.11000000,-0.02786741, 0.01057858,-0.03892226,-0.04574364, 
	-0.03874080,-0.03782372,-0.07046855, 0.09546597, 0.21953269, 
	0.02522348, 0.15263050, 0.08042611, 0.01878626, 0.08715453, 
	0.10500484]
	alpg=[0, 
	0.55159092, 0.66205886, 0.90529132, 1.51710827, 2.86070364, 
	1.88862966, 1.32250290, 1.23166285, 1.77503721, 1.11955204, 
	1.28263182, 1.22344336, 1.70936266, 1.54075036, 1.38200579, 
	2.18849322, 1.36779065, 1.27039703, 1.64466502, 1.58859404, 
	1.65357953, 1.50021521, 1.30104175, 1.46301827, 1.32928147, 
	1.02766713, 1.02291377, 0.94343886, 1.14881311, 1.47080755, 
	1.76901636, 1.98724061, 2.41244711, 2.26739524, 2.95378999, 
	1.20807752, 1.65941046, 1.62733880, 1.61344972, 1.63220728, 
	1.60899928, 1.43501286, 1.54559205, 1.32663678, 1.37644152, 
	1.36051851, 1.23395526, 1.65734544, 1.53895240, 1.97542736, 
	1.97636542, 2.05432381, 3.80138135, 1.43893803, 1.75505957, 
	1.59815118, 1.76401732, 1.63999999, 1.63999999, 1.63999999, 
	1.63999999, 1.63999999, 1.63999999, 1.63999999, 1.63999999, 
	1.63999999, 1.63999999, 1.63999999, 1.63999999, 1.63999999, 
	1.63999999, 1.47055223, 1.81127084, 1.40189963, 1.54015481, 
	1.33721475, 1.57165422, 1.04815857, 1.78342098, 2.79106396, 
	1.78160840, 2.47588882, 2.37670734, 1.76613217, 2.66172302, 
	2.82773085]

	def __str__(self):
		return str(self._alpha)+" \n" +str(self._J)+"\n"

	def __init__(self,
		fit_parameters=0,             # fit eem parameters (0 no, 1 fit J, 2 fit alp, 3 fit J & alp
		alpha=alpg,
		J=J,
		xi=xi,
		dtype=tf.float32            #single or double precision
		):
		super().__init__(dtype=dtype, name="EEM")
		#self._alpha = tf.Variable(alpha, name="atomicRadius", dtype=dtype,trainable=trainable)
		self._alpha = tf.Variable(alpha, name="atomicRadius", dtype=dtype,trainable=(fit_parameters>=2))
		self._J = tf.Variable(J, name="J", dtype=dtype,trainable=(fit_parameters==1 or fit_parameters==3))
		self._Xi = tf.Variable(xi, name="Xi", dtype=dtype,trainable=False)
		#print("alpha=",self.alpha)
		#print("J=",self.J)

	#calculates charge per atom using EEM method
	def compute_atomic_charges(self, Z, R, Xia, Q_tot=None, batch_seg=None):
		#Xia =  tf.gather(self.Xi, Z)
		Ja =  tf.gather(self.J, Z)
		Alfa = tf.gather(self.alpha, Z)
		pi=tf.constant(math.pi,dtype=self.dtype)
		f=2.0/tf.math.sqrt(pi)
		index=tf.constant(range(len(batch_seg)),dtype=tf.int32)
		if batch_seg is None:
			batch_seg = tf.zeros_like(Z)

		#print("Ja=",Ja)
		#print("Alfa=",Alfa)

		#number of atoms per batch
		Na_per_batch = tf.math.segment_sum(tf.ones_like(batch_seg, dtype=self.dtype), batch_seg)
		if Q_tot is None: #assume desired total charge zero if not given
			Q_tot = tf.zeros_like(Na_per_batch, dtype=self.dtype)
		Qa = None
		nb=0
		for imol in range(len(Q_tot)):
			nAtoms=int(Na_per_batch[imol])
			ne=nb+nAtoms
			Rmol = tf.gather(R, index[nb:ne])
			Ximol = tf.gather(Xia, index[nb:ne])
			Jmol = tf.gather(Ja, index[nb:ne])
			Alfmol = tf.gather(Alfa, index[nb:ne])

			#print("ximol=",Ximol)
			rhs = tf.concat([-Ximol,[Q_tot[imol]]],0)
			#print("rhs=",rhs)

			idxi = np.ones(nAtoms*nAtoms)
			idxj = np.ones(nAtoms*nAtoms)
			k=0
			for i in range(int(nAtoms)):
				for j in range(int(nAtoms)):
					idxi[k]=i
					idxj[k]=j
					k = k + 1
			idxi=tf.constant(idxi,dtype=tf.int32)
			idxj=tf.constant(idxj,dtype=tf.int32)
			Ri = tf.gather(Rmol, idxi)
			Rj = tf.gather(Rmol, idxj)
			#Rij = tf.sqrt(1e-14+tf.nn.relu(tf.reduce_sum((Ri-Rj)**2, -1))) #relu prevents negative numbers in sqrt
			Rij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri-Rj)**2, -1))) #relu prevents negative numbers in sqrt
			Rij += 1e-20 # if not nan in forces
			Ai = tf.gather(Alfmol, idxi)
			Aj = tf.gather(Alfmol, idxj)
			G=1.0/tf.math.sqrt(Ai*Ai+Aj*Aj)
			A=tf.where(Rij > 1e-14, tf.math.erf(G*Rij)/Rij, 0)
			A=tf.reshape(A,shape=[nAtoms,nAtoms])
			line = tf.ones(nAtoms,dtype=self.dtype)
			line=tf.reshape(line,shape=[1,nAtoms])
			Amatrix=tf.concat([A,line],0)
			line = tf.concat([tf.ones(nAtoms,dtype=self.dtype) ,[0.0] ],0)
			col=tf.reshape(line,shape=[nAtoms+1,1])
			Amatrix=tf.concat([Amatrix,col],1)

			Gii=1/(tf.math.sqrt(tf.constant(2.0,dtype=self.dtype))*Alfmol)
			Ad = Jmol+f*Gii
			diagonal = tf.concat([Ad,[0.0]],0)
			diagonalMatrix=tf.linalg.diag(diagonal)
			Amatrix = Amatrix + diagonalMatrix;

			rhs=tf.reshape(rhs,shape=[Amatrix.shape[0],1])
			Qmol=tf.linalg.solve(Amatrix,rhs)
			qs =  tf.gather(tf.reshape(Qmol,shape=Amatrix.shape[0]), list(range(nAtoms)))
			if imol==0:
				Qa = qs
			else:
				Qa = tf.concat([Qa, qs], 0)
			nb=ne
			
		return Qa

	#calculates the electrostatic energy per atom 
	#for very small distances, the 1/r law is shielded to avoid singularities
	def electrostatic_energy_per_atom(self, Z, Dij, Xia, Qa, idx_i, idx_j):
		#Xia =  tf.gather(self.Xi, Z)
		#gather charges
		Ja =  tf.gather(self.J, Z)
		Alfa = tf.gather(self.alpha, Z)
		pi=tf.constant(math.pi,dtype=self.dtype)
		Qi = tf.gather(Qa, idx_i)
		Qj = tf.gather(Qa, idx_j)
		Xi = tf.gather(Xia, idx_i)
		Ji = tf.gather(Ja, idx_i)
		Alfi = tf.gather(Alfa, idx_i)
		Alfj = tf.gather(Alfa, idx_j)
		Gammaii=1.0/tf.math.sqrt(tf.constant(2.0, dtype=self.dtype))/Alfi
		Gammaij=1.0/tf.math.sqrt(Alfi*Alfi+Alfj*Alfj)
		Eelei=Xi*Qi+(0.5*Ji+1.0/tf.math.sqrt(pi)*Gammaii)*Qi*Qi
		Eeleij=0.5*Qi*Qj*tf.math.erf(Gammaij*Dij)/Dij

		Eleci=tf.math.segment_mean(Eelei, idx_i)
		Eeleij=tf.math.segment_sum(Eeleij, idx_i) 

		#print("Elec",Eleci+Eeleij)
		return Eleci+Eeleij

	#calculates charge & the electrostatic energy per atom using EEM method
	def compute_atomic_charges_and_electrostatic_energies(self, Z, R, Xia, Q_tot=None, batch_seg=None):
		Ja =  tf.gather(self.J, Z)
		Alfa = tf.gather(self.alpha, Z)
		pi=tf.constant(math.pi,dtype=self.dtype)
		f=2.0/tf.math.sqrt(pi)
		index=tf.constant(range(len(batch_seg)),dtype=tf.int32)
		if batch_seg is None:
			batch_seg = tf.zeros_like(Z)

		#print("Ja=",Ja)
		#print("Alfa=",Alfa)

		#number of atoms per batch
		Na_per_batch = tf.math.segment_sum(tf.ones_like(batch_seg, dtype=self.dtype), batch_seg)
		if Q_tot is None: #assume desired total charge zero if not given
			Q_tot = tf.zeros_like(Na_per_batch, dtype=self.dtype)
		Qa = None
		Ea = None
		nb=0
		for imol in range(len(Q_tot)):
			nAtoms=int(Na_per_batch[imol])
			ne=nb+nAtoms
			Rmol = tf.gather(R, index[nb:ne])
			Ximol = tf.gather(Xia, index[nb:ne])
			Jmol = tf.gather(Ja, index[nb:ne])
			Alfmol = tf.gather(Alfa, index[nb:ne])

			#print("ximol=",Ximol)
			rhs = tf.concat([-Ximol,[Q_tot[imol]]],0)
			#print("rhs=",rhs)

			idxi = np.ones(nAtoms*nAtoms)
			idxj = np.ones(nAtoms*nAtoms)
			k=0
			for i in range(int(nAtoms)):
				for j in range(int(nAtoms)):
					idxi[k]=i
					idxj[k]=j
					k = k + 1
			idxi=tf.constant(idxi,dtype=tf.int32)
			idxj=tf.constant(idxj,dtype=tf.int32)
			Ri = tf.gather(Rmol, idxi)
			Rj = tf.gather(Rmol, idxj)
			#Rij = tf.sqrt(1e-14+tf.nn.relu(tf.reduce_sum((Ri-Rj)**2, -1))) #relu prevents negative numbers in sqrt
			Rij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri-Rj)**2, -1))) #relu prevents negative numbers in sqrt
			Rij += 1e-20
			#print("Rij=",Rij)
			Ai = tf.gather(Alfmol, idxi)
			Aj = tf.gather(Alfmol, idxj)
			#print("Ai=",Ai)
			G=tf.math.sqrt(Ai*Ai+Aj*Aj)
			G += 1e-20 # prevents nan when fit Alp
			G=1.0/G
			#print("G=",G)
			A=tf.where(Rij > 1e-14, tf.math.erf(G*Rij)/Rij, 0)
			A=tf.reshape(A,shape=[nAtoms,nAtoms])
			#print("A=",A)
			line = tf.ones(nAtoms,dtype=self.dtype)
			line=tf.reshape(line,shape=[1,nAtoms])
			Amatrix=tf.concat([A,line],0)
			#print("line=",line)
			#print("Amatrix=",Amatrix)
			line = tf.concat([tf.ones(nAtoms,dtype=self.dtype) ,[0.0] ],0)
			#print("line=",line)
			col=tf.reshape(line,shape=[nAtoms+1,1])
			#print("col=",col)
			Amatrix=tf.concat([Amatrix,col],1)
			#print("Amatrix=",Amatrix)

			Gii=(tf.math.sqrt(tf.constant(2.0,dtype=self.dtype)*Alfmol*Alfmol))
			Gii += 1e-20  # prevents nan when fit Alp
			Gii=1/Gii
			Ad = Jmol+f*Gii
			diagonal = tf.concat([Ad,[0.0]],0)
			diagonalMatrix=tf.linalg.diag(diagonal)
			#print("diag=",diagonalMatrix)
			Amatrix = Amatrix + diagonalMatrix;

			rhs=tf.reshape(rhs,shape=[Amatrix.shape[0],1])
			Qmol=tf.linalg.solve(Amatrix,rhs)
			#print("A=",Amatrix)
			# compute energies : q^T(0.5Aq-X)
			As =  tf.slice(Amatrix, [0, 0], [nAtoms, nAtoms])
			qs =  tf.gather(tf.reshape(Qmol,shape=Amatrix.shape[0]), list(range(nAtoms)))
			qs=tf.reshape(qs,shape=[As.shape[0],1])
			Emol = tf.reshape(0.5*qs*tf.matmul(As, qs),shape=As.shape[0])+tf.gather(tf.reshape(-rhs*Qmol,shape=Amatrix.shape[0]), list(range(nAtoms)))
			#print("Emol=",Emol)
			#print("Qmol=",Qmol)
			qs =  tf.gather(tf.reshape(Qmol,shape=Amatrix.shape[0]), list(range(nAtoms)))
			if imol==0:
				Ea = Emol
				Qa = qs
			else:
				Ea = tf.concat([Ea, Emol], 0)
				Qa = tf.concat([Qa, qs], 0)
			nb=ne
			
		return Qa , Ea


	def print_parameters(self):
		#print("Alpha=",end='')
		#[ print("(",i,",", self.alpha.numpy()[i],"), ",end='') for i in range(1,len(self.alpha.numpy())) ]
		#print("")
		print("Alpha=",self.alpha.numpy()[1:])
		print("J=",self.J.numpy()[1:])
		print("Alpha & J for Z=",[ i for i in range(1,len(self.alpha.numpy())+1) ])

	@property
	def alpha(self):
		return self._alpha
	@property
	def J(self):
		return self._J

	@property
	def Xi(self):
		return self._Xi
