import numpy  as np
from .PeriodicTable import *

class DataContainer:
	def __repr__(self):
		return "DataContainer"
	def __init__(self, filename, convDistanceToBohr=1.0, convEnergyToHartree=1.0, convDipoleToAU=1.0, sr_cutoff=None, lr_cutoff=None,num_struct=0,seed=None):
		#read in data
		dictionary = np.load(filename)
		nAll=len(dictionary['N'])
		ns=-1
		if num_struct>0:
			ns=num_struct
		else:
			ns=nAll
		if seed is not None:
			random_state = np.random.RandomState(seed=seed)
			idx = random_state.permutation(np.arange(nAll))[0:ns]
		else:
			idx = np.arange(ns)

		#number of atoms
		if 'N' in dictionary: 
			self._N = dictionary['N'][idx]
		else:
			self._N = None

		# masses
		if 'M' in dictionary: 
			self._M = dictionary['M'][idx] 
		else:
			self._M = None


		#atomic numbers/nuclear charges
		if 'Z' in dictionary: 
			self._Z = dictionary['Z'][idx]
			if self.M is None:
				periodicTable=PeriodicTable()
				self._M = [] 
				for zMol in self.Z:
					mMol = []
					for z in zMol:
						if int(z)==0:
							mMol.append(0)
						else:
							mass=periodicTable.elementZ(int(z)).isotopes[0].rMass # mass of first istotope
							mMol.append(mass)
					self._M.append(mMol)
				self._M = np.array(self._M)
		else:
			self._Z = None


	   #reference dipole moment vector
		if 'D' in dictionary: 
			self._D = dictionary['D'][idx]*convDipoleToAU
		else:
			self._D = None
		#reference total charge
		if 'Q' in dictionary: 
			self._Q = dictionary['Q'][idx]
		else:
			self._Q = np.zeros(self.N.shape[0], dtype=float)

		#reference atomic charges
		if 'Qa' in dictionary: 
			self._Qa = dictionary['Qa'][idx]
		else:
			self._Qa = None

		#maximum number of atoms per molecule
		self._N_max	= self.Z.shape[1] 
		#print("Nmax=",self._N_max)

		# Spin multiplicity (2S+1 not S)
		if 'Mult' in dictionary: 
			self._Mult = dictionary['Mult'][idx]
		else:
			self._Mult = np.ones(self.N.shape[0],dtype=float)
			# if number of electrons is odd , set multiplicity to 2
			if self.Z is not None and self.Q is not None:
				for im, zMol in enumerate(self.Z):
					ne = np.sum(zMol)
					ne = ne - self.Q[im]
					ne = int(ne+0.5)
					if ne%2==1:
						self._Mult[im] = 2

		#reference total charge by spin alpha, beta
		if 'QAlpha' in dictionary: 
			self._QAlpha = dictionary['QAlpha'][idx]
		else:
			self._QAlpha = 0.5*(self.Q-self.Mult+1)

		if 'QBeta' in dictionary: 
			self._QBeta = dictionary['QBeta'][idx]
		else:
			self._QBeta = 0.5*(self.Q+self.Mult-1)


		#reference Alpha atomic charges
		if 'QaAlpha' in dictionary: 
			self._QaAlpha = dictionary['QaAlpha'][idx]
		else:
			self._QaAlpha = []
			for im in range(self.N.shape[0]):
				m = []
				N = self.N[im]
				m = [self._QAlpha[im]/N]*N
				nres =  self.N_max-N
				if nres>0:
					m = m + [0.0]*nres
				self._QaAlpha.append(m)
		self._QaAlpha = np.asarray(self.QaAlpha)

		#reference Beta atomic charges
		if 'QaBeta' in dictionary: 
			self._QaBeta = dictionary['QaBeta'][idx]
		else:
			self._QaBeta = []
			for im in range(self.N.shape[0]):
				m = []
				N = self.N[im]
				m = [self._QBeta[im]/N]*N
				nres =  self.N_max-N
				if nres>0:
					m = m + [0.0]*nres
				self._QaBeta.append(m)
		self._QaBeta = np.asarray(self.QaBeta)

		#positions (cartesian coordinates)
		if 'R' in dictionary:	 
			self._R = dictionary['R'][idx]*convDistanceToBohr 
		else:
			self._R = None
		#reference energy
		if 'E' in dictionary:
			self._E = dictionary['E'][idx]*convEnergyToHartree 
		else:
			self._E = None
		#reference atomic energies
		if 'Ea' in dictionary:
			self._Ea = dictionary['Ea'][idx]*convEnergyToHartree
		else:
			self._Ea = None
		#reference forces
		if 'F' in dictionary:
			self._F = dictionary['F'][idx]*convEnergyToHartree/convDistanceToBohr 
		else:
			self._F = None

	
		if 'Cell' in dictionary: 
			self._cells = dictionary['Cell'][idx]*convDistanceToBohr 
			self._sr_cutoff = sr_cutoff
			self._lr_cutoff = lr_cutoff
			if self._sr_cutoff is not None:
				self._sr_cutoff *= convDistanceToBohr 
			if self._lr_cutoff is not None:
				self._lr_cutoff *= convDistanceToBohr 
			self.set_idx_periodic()
		else:
			self._sr_cutoff = sr_cutoff
			self._lr_cutoff = lr_cutoff
			self._cells = None
			#self.set_idx_mol()
			self.set_idx_periodic()

#####################################################################
	def set_idx_periodic(self):
		import ase
		from ase import Atoms
		from ase.neighborlist import neighbor_list
		self._idx_i =[]
		self._idx_j =[]
		self._sr_idx_i =[]
		self._sr_idx_j =[]
		self._offsets =[]
		self._sr_offsets =[]
		lrcal=True
		srcal=True
		n=self.N.shape[0]
		ns=n//100
		if ns<1:
			ns=1
		print("Setting of neighbor lists.......",flush=True);
		for im in range(self.N.shape[0]):
			if im%ns==0:
				print("\t {:<5d} / {:>5d}   .......".format(im,n),flush=True)
			#print("im=",im,"/",self.N.shape[0])
			N = self.N[im]
			z = self.Z[im][:N]
			if N != len(z):
				print("N=",N)
				print("lz=",len(z))
				
			pos=[]
			for ia in range(N):
				pos.append(self.R[im][ia])
			cell=[(0,0,0),(0,0,0),(0,0,0)]
			pbc=[False,False,False]
			#print("cells=",self.cells)
			if self.cells is not None and ~np.isnan(self.cells).all() and ~np.isnan(self.cells[im]).any(): 
				for k in range(np.array(self._cells[im]).shape[0]):
					if self._cells[im][k] is not None:
						cell[k]= self._cells[im][k]
						pbc[k] = True

			atoms=Atoms(numbers=z, positions=pos,cell=cell,pbc=pbc)
			#atoms.set_positions(pos)
			#atoms.set_cell(cell)
			#atoms.set_pbc(pbc)
			if any(atoms.get_pbc()):
				if self._lr_cutoff is None and self._sr_cutoff is None:
					print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
					print("lr_cutoff & sr_cutoff are needed for periodic system")
					print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
					sys.exit(1)

			if self._lr_cutoff is not None and self._sr_cutoff is not None:
				idx_i, idx_j, S = neighbor_list('ijS', atoms, self._lr_cutoff, self_interaction=False)
				offsets = np.dot(S, atoms.get_cell())
				self._idx_i.append(idx_i)
				self._idx_j.append(idx_j)
				self._offsets.append(offsets)
				sr_idx_i, sr_idx_j, sr_S = neighbor_list('ijS', atoms, self._sr_cutoff, self_interaction=False)
				sr_offsets = np.dot(sr_S, atoms.get_cell())
				self._sr_idx_j.append(sr_idx_j)
				self._sr_idx_i.append(sr_idx_i)
				self._sr_offsets.append(sr_offsets)
			elif self._lr_cutoff is not None or self._sr_cutoff is not None:
				srcal=False
				cutoff= self._lr_cutoff
				if cutoff is None:
					cutoff= self._sr_cutoff
				idx_i, idx_j, S = neighbor_list('ijS', atoms, cutoff, self_interaction=False)
				offsets = np.dot(S, atoms.get_cell())
				self._idx_i.append(idx_i)
				self._idx_j.append(idx_j)
				self._offsets.append(offsets)
				self._sr_idx_j.append(None)
				self._sr_idx_i.append(None)
				self._sr_offsets.append(None)
			else:
				lrcal=False
				srcal=False
				idx_i = []
				idx_j = []
				for ia in range(N):
					for ja in range(N):
						if ja != ia:
							idx_i.append(ia)
							idx_j.append(ja)
				self._idx_i.append(idx_i)
				self._idx_j.append(idx_j)
				self._offsets.append([None]*len(idx_i))
				self._sr_idx_j.append(None)
				self._sr_idx_i.append(None)
				self._sr_offsets.append(None)
		#for i in range(len(self.idx_i)):
		#	print(self.idx_i[i])
			
		self._idx_i =np.asarray(self._idx_i,dtype=object)
		self._idx_j =np.asarray(self._idx_j,dtype=object)
		self._offsets =np.asarray(self._offsets,dtype=object)
		self._sr_idx_i =np.asarray(self._sr_idx_i,dtype=object)
		self._sr_idx_j =np.asarray(self._sr_idx_j,dtype=object)
		self._sr_offsets =np.asarray(self._sr_offsets,dtype=object)
		"""
		print(self.idx_i.shape)
		print(self._offsets.shape)
		print(self._sr_offsets.shape)
		di=self._idx_i-self._idx_j
		for k in range(di.shape[0]):
			ddi=di[k]
			print("ddi=",ddi[(ddi<1) & (ddi>-1)])
			print("off=",self._offsets[k][(ddi<1) & (ddi>-1)])
		di=self._sr_idx_i-self._sr_idx_j
		for ddi in di:
			print("srddi=",ddi[(ddi<1) & (ddi>-1)])
		print("lrlen=",self._idx_i.shape)
		print("srlen=",self._sr_idx_i.shape)
		print("sr=",self._sr_idx_i)
		"""
		if not lrcal and self._offsets.all() is None:
			self._offsets=None
		if not srcal and self._sr_idx_i.all() is None:
			self._sr_idx_i=None
		if not srcal and self._sr_idx_j.all() is None:
			self._sr_idx_j=None
		if not srcal and self._sr_offsets.all() is None:
			self._sr_offsets=None
				
	
	def set_idx_mol(self):
		#construct indices used to extract position vectors to calculate relative positions 
		#(basically, constructs indices for calculating all possible interactions (excluding self interactions), 
		#this is a naive (but simple) O(N^2) approach, could be replaced by something more sophisticated) 
		self._idx_i = np.empty([self.N_max, self.N_max-1],dtype=int)
		for i in range(self.idx_i.shape[0]):
			for j in range(self.idx_i.shape[1]):
				self._idx_i[i,j] = i

		self._idx_j = np.empty([self.N_max, self.N_max-1],dtype=int)
		for i in range(self.idx_j.shape[0]):
			c = 0
			for j in range(self.idx_j.shape[0]):
				if j != i:
					self._idx_j[i,c] = j
					c += 1

	@property
	def N_max(self):
		return self._N_max

	@property
	def N(self):
		return self._N

	@property
	def Z(self):
		return self._Z

	@property
	def M(self):
		return self._M

	@property
	def Q(self):
		return self._Q

	@property
	def Qa(self):
		return self._Qa

	@property
	def Mult(self):
		return self._Mult

	@property
	def QAlpha(self):
		return self._QAlpha

	@property
	def QBeta(self):
		return self._QBeta

	@property
	def QaAlpha(self):
		return self._QaAlpha

	@property
	def QaBeta(self):
		return self._QaBeta

	@property
	def D(self):
		return self._D

	@property
	def R(self):
		return self._R

	@property
	def E(self):
		return self._E

	@property
	def Ea(self):
		return self._Ea
	
	@property
	def F(self):
		return self._F

	#indices for atoms i (when calculating interactions)
	@property
	def idx_i(self):
		return self._idx_i

	#indices for atoms j (when calculating interactions)
	@property
	def idx_j(self):
		return self._idx_j

	@property
	def sr_idx_i(self):
		return self._sr_idx_i

	@property
	def sr_idx_j(self):
		return self._sr_idx_j

	@property
	def offsets(self):
		return self._offsets

	@property
	def sr_offsets(self):
		return self._sr_offsets

	@property
	def cells(self):
		return self._cells

	def __len__(self): 
		return self.Z.shape[0]

	def __getitem__(self, idx):
		if type(idx) is int or type(idx) is np.int64:
			idx = [idx]

		data = {	'E':		 [],
				'Ea':		 [],	
				'F':		 [],
				'Z':		 [],
				'M':		 [],
				'D':		 [],
				'Q':		 [],
				'Qa':	 	 [],
				'N':	 	 [],
				'Mult':	 	 [],
				'QAlpha': 	 [],
				'QBeta': 	 [],
				'QaAlpha': 	 [],
				'QaBeta': 	 [],
				'R':		 [],
				'idx_i':	 [],
				'idx_j':	 [],
				'offsets'  :     [],
				'sr_idx_i':	 [],
				'sr_idx_j':	 [],
				'sr_offsets' :   [],
				'batch_seg':     [],
				'Cell':          [],
			}

		Ntot = 0 #total number of atoms
		Itot = 0 #total number of interactions
		for k, i in enumerate(idx):
			N = self.N[i] #number of atoms
			I = N*(N-1)   #number of interactions
			#append data
			if self.E is not None:
				data['E'].append(self.E[i]/N) # Energy by atom 
			else:
				data['E'].append(np.nan)
			if self.Ea is not None:
				data['Ea'].extend(self.Ea[i,:N].tolist())
			else:
				data['Ea'].extend([np.nan]*N)
			if self.Q is not None:
				data['Q'].append(self.Q[i])
			else:
				data['Q'].append(np.nan)
			if self.Mult is not None:
				data['Mult'].append(self.Mult[i])
			else:
				data['Mult'].append(np.nan)
			if self.QAlpha is not None:
				data['QAlpha'].append(self.QAlpha[i])
			else:
				data['QAlpha'].append(np.nan)
			if self.QBeta is not None:
				data['QBeta'].append(self.QBeta[i])
			else:
				data['QBeta'].append(np.nan)

			if self.Qa is not None:
				data['Qa'].extend(self.Qa[i,:N].tolist())
			else:
				data['Qa'].extend([np.nan]*N)

			if self.QaAlpha is not None:
				data['QaAlpha'].extend(self.QaAlpha[i,:N].tolist())
			else:
				data['QaAlpha'].extend([np.nan]*N)

			if self.QaBeta is not None:
				data['QaBeta'].extend(self.QaBeta[i,:N].tolist())
			else:
				data['QaBeta'].extend([np.nan]*N)

			if self.Z is not None:
				data['Z'].extend(self.Z[i,:N].tolist())
			else:
				data['Z'].append(0)

			if self.M is not None:
				data['M'].extend(self.M[i,:N].tolist())
			else:
				data['M'].append(6.0)

			if self.D is not None:
				data['D'].extend(self.D[i:i+1,:].tolist())
			else:
				data['D'].extend([[np.nan,np.nan,np.nan]])

			if self.cells is not None:
				data['Cell'].extend(self.cells[i:i+1,:,:].tolist())
			else:
				data['Cell'].extend([[(np.nan,np.nan,np.nan),(np.nan,np.nan,np.nan),(np.nan,np.nan,np.nan)]])

			if self.R is not None:
				data['R'].extend(self.R[i,:N,:].tolist())
			else:
				data['R'].extend([[np.nan,np.nan,np.nan]])
			if self.F is not None:
				data['F'].extend(self.F[i,:N,:].tolist())
			else:
				data['F'].extend([[np.nan,np.nan,np.nan]])

			#print(len(self.idx_i),len(self.Z))
			if len(self.idx_i)==len(self.Z): 
				if self.idx_i is not None:
					data['idx_i'].extend(np.reshape(self.idx_i[i]+Ntot,[-1]).tolist())
				if self.idx_j is not None:
					data['idx_j'].extend(np.reshape(self.idx_j[i]+Ntot,[-1]).tolist())
				if self.offsets is not None:
					data['offsets'].extend(self.offsets[i].tolist())
				if self.sr_idx_i is not None:
					data['sr_idx_i'].extend(np.reshape(self.sr_idx_i[i]+Ntot,[-1]).tolist())
				if self.sr_idx_j is not None:
					data['sr_idx_j'].extend(np.reshape(self.sr_idx_j[i]+Ntot,[-1]).tolist())
				if self.sr_offsets is not None:
					data['sr_offsets'].extend(self.sr_offsets[i].tolist())
			else:
				data['idx_i'].extend(np.reshape(self.idx_i[:N,:N-1]+Ntot,[-1]).tolist())
				data['idx_j'].extend(np.reshape(self.idx_j[:N,:N-1]+Ntot,[-1]).tolist())
				data['offsets']=[]
				data['sr_idx_i']=[]
				data['sr_idx_j']=[]
				data['sr_offsets']=[]
			#offsets could be added in case they are need
			data['batch_seg'].extend([k] * N)
			#increment totals
			Ntot += N
			Itot += I

		return data

