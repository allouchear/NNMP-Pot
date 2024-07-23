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
		trainable=True,
		alpha=alpg,
		J=J,
		xi=xi,
		dtype=tf.float32            #single or double precision
		):
		super().__init__(dtype=dtype, name="EEM")
		self._alpha = tf.Variable(alpha, name="atomicRadius", dtype=dtype,trainable=trainable)
		self._J = tf.Variable(J, name="J", dtype=dtype,trainable=trainable)
		self._Xi = tf.Variable(xi, name="Xi", dtype=dtype,trainable=False)
		print("alpha=",self.alpha)
		print("J=",self.J)

	#calculates the electrostatic energy per atom 
	#for very small distances, the 1/r law is shielded to avoid singularities
	def compute_atomic_charges(self, Z, R, Xia, Q_tot=None, batch_seg=None):
		#Xia =  tf.gather(self.Xi, Z)

		Ja =  tf.gather(self.J, Z)
		Alfa = tf.gather(self.alpha, Z)
		#gather charges
		pi=tf.constant(math.pi,dtype=self.dtype)
		f=2.0/tf.math.sqrt(pi)
		#print("Alpfa=",Alfa)
		#print("R=",R)
		index=tf.constant(range(len(batch_seg)),dtype=tf.int32)
		#print("index=",index)

		if batch_seg is None:
			batch_seg = tf.zeros_like(Z)
		#number of atoms per batch
		Na_per_batch = tf.math.segment_sum(tf.ones_like(batch_seg, dtype=self.dtype), batch_seg)
		if Q_tot is None: #assume desired total charge zero if not given
			Q_tot = tf.zeros_like(Na_per_batch, dtype=self.dtype)
		#print("Na_per_batch=",Na_per_batch)
		#print("Q_tot=",Q_tot)
		#print("len batch seq=",len(batch_seg))
		Qa = np.zeros(len(batch_seg))
		nb=0
		for imol in range(len(Q_tot)):
			nAtoms=int(Na_per_batch[imol])
			ne=nb+nAtoms
			#print("mol # ",imol, "nb=",nb, "ne=",ne)
			#print("idx # ",index[nb:ne])
			Rmol = tf.gather(R, index[nb:ne])
			Ximol = tf.gather(Xia, index[nb:ne])
			Jmol = tf.gather(Ja, index[nb:ne])
			Alfmol = tf.gather(Alfa, index[nb:ne])
			#print("Rmol # ",Rmol)
			#print("Alfmol # ",Alfmol)
			Amatrix = np.zeros( (nAtoms+1, nAtoms+1) )
			rhs = np.zeros( (nAtoms+1) )
			for i in range(int(nAtoms)):
				rhs[i]=-Ximol[i]
			rhs[nAtoms]=Q_tot[imol]

			for i in range(int(nAtoms)):
				Amatrix[nAtoms][i]=Amatrix[i][nAtoms]=1.0
			Amatrix[nAtoms][nAtoms]=0.0

			for i in range(int(nAtoms)):
				#print('i:')
				#print(Rmol[i])
				G=tf.math.sqrt(2*Alfmol[i]*Alfmol[i])
				if G>0:
					G=1/G
				else:
					G=1e30
				#print('GAA:',G)
				#print('Jmol:',Jmol[i])
				#print('f:',f)
				Ai = Jmol[i]+f*G
				#print('Ai:',Ai)
				Amatrix[i][i] =Ai

				for j in range(i+1,int(nAtoms)):
					#print("\tj:")
					#print("\t",Rmol[j])
					G=tf.math.sqrt(Alfmol[i]*Alfmol[i]+Alfmol[j]*Alfmol[j])
					if G>0:
						G=1/G
					else:
						G=1e30
					#print('\tGAB:',G)
					Rij= tf.sqrt(tf.nn.relu(tf.reduce_sum((Rmol[i]-Rmol[j])**2, -1)))
					A = tf.math.erf(G*Rij)/Rij
					#print('\tA:',A)
					Amatrix[i][j]=Amatrix[j][i]=A
			#print("Amatrix=",Amatrix)
			#print("rhs=",rhs)
			Amatrix = tf.constant(Amatrix)
			rhs = tf.constant(rhs)
			rhs=tf.reshape(rhs,shape=[rhs.shape[0],1])
			#print("Amatrix=",Amatrix)
			#print("rhs=",rhs)
			Qmol=tf.linalg.solve(tf.constant(Amatrix),tf.constant(rhs))
			#print("Qmol=",tf.reshape(Qmol,shape=Amatrix.shape[0]))
			Qa[nb:ne]=tf.reshape(Qmol,shape=Amatrix.shape[0]).numpy()[0:nAtoms]
			nb=ne
			
		#print("Qa=",Qa)
		#return scaled charges (such that they have the desired total charge)
		return Qa 

	#calculates the electrostatic energy per atom 
	#for very small distances, the 1/r law is shielded to avoid singularities
	def electrostatic_energy_per_atom(self, Z, Dij, Xia, Qa, idx_i, idx_j):
		#Xia =  tf.gather(self.Xi, Z)
		#gather charges
		Ja =  tf.gather(self.J, Z)
		Alfa = tf.gather(self.alpha, Z)
		pi=tf.constant(math.pi,dtype=self.dtype)
		f=1.0/tf.math.sqrt(pi)
		Qi = tf.gather(Qa, idx_i)
		Qj = tf.gather(Qa, idx_j)
		Xi = tf.gather(Xia, idx_i)
		Xj = tf.gather(Xia, idx_j)
		Ji = tf.gather(Ja, idx_i)
		Alfi = tf.gather(Alfa, idx_i)
		Alfj = tf.gather(Alfa, idx_j)
		Gammaii=1.0/tf.math.sqrt(Alfi*Alfi+Alfi*Alfi)
		Gammaij=1.0/tf.math.sqrt(Alfi*Alfi+Alfj*Alfj)
		n_per_idx = tf.math.segment_sum(tf.ones_like(idx_i, dtype=self.dtype), idx_i)
		#Eele=Xi*Qi+(0.5*Ji+f*Gammaii)*Qi*Qi+0.5*Qi*Qj*tf.math.erf(Gammaij*Dij)/Dij
		Eelei=Xi*Qi+(0.5*Ji+f*Gammaii)*Qi*Qi
		Eeleij=0.5*Qi*Qj*tf.math.erf(Gammaij*Dij)/Dij

		Eleci=tf.math.segment_sum(Eelei, idx_i)/(1.0*n_per_idx)
		Eeleij=tf.math.segment_sum(Eeleij, idx_i) 
		return Eleci+Eeleij

	#calculates charge & the electrostatic energy per atom using EEM method
	def compute_atomic_charges_and_electrostatic_energies(self, Z, R, Dij, idx_i, idx_j, Xia, Q_tot=None, batch_seg=None):
		Qa = self.compute_atomic_charges(Z, R, Xia, Q_tot=Q_tot, batch_seg= batch_seg)
		Ea =self.electrostatic_energy_per_atom(Z, Dij, Xia, Qa, idx_i, idx_j)
		return Qa, Ea


	@property
	def alpha(self):
		return self._alpha
	@property
	def J(self):
		return self._J

	@property
	def Xi(self):
		return self._Xi
