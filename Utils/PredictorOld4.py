import tensorflow as tf
import numpy as np
import ase
from ase.neighborlist import neighbor_list
from PhysModel.PhysModel import *
from Utils.UtilsFunctions import *
from Utils.PhysicalConstants import *
from Utils.UtilsModel import *

from timeit import default_timer as timer



'''
Calculator for the atomic simulation environment (ASE)
It computes energies and forces using a NNMol
'''

def getExampleData():
	data = {}
	data['Z'] =  [1,6]
	data['R'] =  [[0.0,0.0,0.0],[1.0,0.0,0.0]]
	data['idx_i'] =  [0]
	data['idx_j'] =  [1]
	data['batch_seg'] =   tf.zeros_like(data['Z'])
	data['offsets'] =   None
	data['sr_offsets'] =   None
	data['sr_idx_i'] =  None
	data['sr_idx_j'] =  None
	data['M'] =  None
	data['QaAlpha'] =  None
	data['QaBeta'] =  None
	return data

def get_idx(conv_distance, atoms, lr_cutoff, sr_cutoff, use_ase):
		#find neighbors and offsets
		if use_ase or any(atoms.get_pbc()):
			if lr_cutoff is not None and sr_cutoff is not None:
				cutoff= (lr_cutoff/conv_distance).numpy()
				idx_i, idx_j, S = neighbor_list('ijS', atoms, cutoff, self_interaction=False)
				offsets = np.dot(S, atoms.get_cell())
				cutoff= (sr_cutoff/conv_distance).numpy()
				sr_idx_i, sr_idx_j, sr_S = neighbor_list('ijS', atoms, cutoff, self_interaction=False)
				sr_offsets = np.dot(sr_S, atoms.get_cell())
			elif lr_cutoff is not None or sr_cutoff is not None:
				srcal=False
				cutoff= lr_cutoff
				if cutoff is None:
					cutoff= sr_cutoff
				"""
				print("cutoff=",cutoff)
				print("cutoff=",cutoff/conv_distance)
				print("---------------------------------------------")
				print(atoms.get_pbc())
				print(atoms.get_cell())
				print(atoms.get_positions())
				print("---------------------------------------------")
				"""
				cutoff= (cutoff/conv_distance).numpy()
				idx_i, idx_j, S = neighbor_list('ijS', atoms, cutoff, self_interaction=False)
				offsets = np.dot(S, atoms.get_cell())
				sr_idx_j = None
				sr_idx_i = None
				sr_offsets = None
			else:
				nAtoms = len(atoms)
				nnAtoms = nAtoms*(nAtoms-1)
				idx_i = np.zeros([nnAtoms], dtype=int)
				idx_j = np.zeros([nnAtoms], dtype=int)
				offsets = np.zeros([nnAtoms,3], dtype=float)
				count = 0
				for i in range(nAtoms):
					for j in range(nAtoms):
						if i != j:
							idx_i[count] = i
							idx_j[count] = j
							count += 1
				sr_offsets = None
				sr_idx_i = None
				sr_idx_j = None
		
		idxData = {}
		idxData['idx_i'] =  idx_i
		idxData['idx_j'] =  idx_j
		idxData['offsets'] =   offsets
		if offsets is not None:
			idxData['offsets'] =   offsets*conv_distance
		idxData['sr_offsets'] =   sr_offsets
		if sr_offsets is not None:
			idxData['sr_offsets'] =   sr_offsets*conv_distance
		idxData['sr_idx_i'] =  sr_idx_i
		idxData['sr_idx_j'] =  sr_idx_j

		return idxData

def get_data_from_atoms(conv_distance, atoms, idxData):
		data = {}
		data['Z'] =  atoms.get_atomic_numbers()
		data['batch_seg'] =   tf.zeros_like(data['Z'])
		data['R'] =  atoms.get_positions()*conv_distance
		data['Cell'] =  atoms.get_cell()*conv_distance
		data['idx_i'] =  idxData['idx_i']
		data['idx_j'] =  idxData['idx_j']
		data['offsets'] =   idxData['offsets']
		data['sr_idx_i'] =  idxData['sr_idx_i']
		data['sr_idx_j'] =  idxData['sr_idx_j']
		data['sr_offsets'] =  idxData['sr_offsets']
		data['M'] =  atoms.get_masses()
		data['QaAlpha'] =  None
		data['QaBeta'] =  None

		return data
	
class Predictor:
	def __init__(self,
		parameters_files,                 # parameters file from which to restore the model (can also be a list for ensembles)
		atoms,                            #ASE atoms object
		charge=0,                         #system charge
		activation_fn=shifted_softplus,   #activation function
		conv_distance=ANG_TO_BOHR,        #coef. conversion of distance from unit of ase in unit of NNMol
		conv_energy=1.0/AUTOEV            #coef. conversion of energy from unit of ase in unit of NNMol
		):

		if(type(parameters_files) is not list):
			self._parameters_files=[parameters_files]
		else:
			self._parameters_files=parameters_files

		self._conv_distance=conv_distance
		self._conv_energy=conv_energy
		
		self._models = []
		self._idxData=None

		n=0
		for fileName in self._parameters_files:
			args = read_model_parameters(fileName)
			directory = args.directory
			best_dir = os.path.join(directory, 'best')
			best_xtbparameters = os.path.join(best_dir, 'best_xtb_parameters.txt')
			best_checkpoint = os.path.join(best_dir, 'best.ckpt')
			physModel=create_model_predict(args, directory, best_xtbparameters)
			data = getExampleData()
			energies, charges, Qa, dipoles, forces, loss, gradients = physModel(data,closs=False) # to set auto shape
			#print("best_checkpoint=",best_checkpoint)
			physModel.load_weights(best_checkpoint)
			if str(physModel) == "PhysModelXTB" :
				physModel.sup_variables.set_xtb_variables_from_trainables()
			self._models.append(physModel)
		self._computeProperties(atoms)

	def reset_neighbor_list(self):
		self._idxData=None

	def calculation_required(self, atoms, quantities=None):
		return atoms != self._last_atoms

	def _computeProperties(self, atoms):
		#find neighbors and offsets
		data = None
		n=0
		nModel=len(self._models)
		for i in range(nModel):
			if i==0 or self._models[i].lr_cut != self._models[i-1].lr_cut  or self._models[i].sr_cut != self._models[i-1].sr_cut:
				use_ase = self._models[i].lr_cut  is not None
				if self._idxData is None:
					start = timer()
					self._idxData=get_idx(self._conv_distance, atoms, self._models[i].lr_cut, self._models[i].sr_cut, use_ase)
					end = timer()
					print("idx time = ", end-start)
				data = get_data_from_atoms(self._conv_distance, atoms, self._idxData)
			start = timer()
			energy, molcharges, atomcharges, dipoles, forces, nhloss = self._models[i].computeProperties(data)
			end = timer()
			print("prop time = ", end-start)
			#print("energy=",energy)
			#print("atomcharges=",atomcharges)
			energy *= len(atoms) # total energy. returned energy of model is energy by atom

			if i == 0:
				self._energy  = energy
				self._molcharges = molcharges
				self._atomcharges  = atomcharges
				self._dipole  = dipoles
				self._forces  = forces
				self._energy_stdev = 0
			else:
				n = i+1
				if energy is not None:
					delta = energy-self.energy
					self._energy += delta/n
					self._energy_stdev += delta*(energy-self.energy)

				if atomcharges is not None:
					self._atomcharges += (atomcharges-self._atomcharges)/n
				if forces is not None:
					self._forces +=  (forces-self._forces)/n 
		
				if dipoles is not None:
					self._dipole +=  (dipoles-self._dipole)/n 
				if molcharges is not None:
					self._molcharges += (molcharges-self._molcharges)/n

		self._energy_stdev = np.sqrt(self.energy_stdev/nModel)

		#print("E=",self._energy)

		if self._energy_stdev is not None:
			self._energy_stdev *= 1.0/self.conv_energy # conversion in ase unit
		if self._energy is not None:
			self._energy *= 1.0/self.conv_energy # conversion in ase unit
			self._energy = self._energy.numpy()
		if self._forces is not None:
			self._forces *= self.conv_distance/self.conv_energy # conversion in ase unit
			self._forces = self._forces.numpy()
		if self._dipole is not None:
			self._dipole *= 1.0/self.conv_distance # conversion in ase unit ( e Ang)
			self._dipole = self._dipole.numpy()
		#store copy of atoms
		self._last_atoms = atoms.copy()

	def computeHessian(self, atoms):
		nModel=len(self._models)
		data = None
		hessian = None
		n=0
		for i in range(nModel):
			if i==0 or self._models[i].lr_cut != self._models[i-1].lr_cut  or self._models[i].sr_cut != self._models[i-1].sr_cut:
				use_neighborlist = self._models[i].lr_cut  is not None
				data = get_data_from_atoms(self._conv_distance, atoms, use_neighborlist, self._models[i].lr_cut, self._models[i].sr_cut)
			lhessian = self._models[i].computeHessian(data)

			if i == 0:
				hessian = lhessian
			else:
				n = i+1
				hessian += (lhessian-hessian)/n

		return hessian

	def computedDipole(self, atoms):
		nModel=len(self._models)
		data = None
		dDipole = None
		n=0
		for i in range(nModel):
			if i==0 or self._models[i].lr_cut != self._models[i-1].lr_cut  or self._models[i].sr_cut != self._models[i-1].sr_cut:
				use_neighborlist = self._models[i].lr_cut  is not None
				data = get_data_from_atoms(self._conv_distance, atoms, use_neighborlist, self._models[i].lr_cut, self._models[i].sr_cut)
			ldDipole = self._models[i].computedDipole(data)

			if i == 0:
				dDipole = ldDipole
			else:
				n = i+1
				dDipole += (ldDipole-dDipole)/n

		return dDipole

	def computeHessianAnddDipoles(self, atoms):
		nModel=len(self._models)
		data = None
		hessian = None
		n=0
		for i in range(nModel):
			if i==0 or self._models[i].lr_cut != self._models[i-1].lr_cut  or self._models[i].sr_cut != self._models[i-1].sr_cut:
				use_neighborlist = self._models[i].lr_cut  is not None
				data = get_data_from_atoms(self._conv_distance, atoms, use_neighborlist, self._models[i].lr_cut, self._models[i].sr_cut)
			lhessian, ldDipole = self._models[i].computeHessianAnddDipoles(data)

			if i == 0:
				hessian = lhessian
				dDipole = ldDipole
			else:
				n = i+1
				hessian += (lhessian-hessian)/n
				dDipole += (ldDipole-dDipole)/n

		return hessian, dDipole

	def computeHessianAnddDipole(self, atoms):
		hessian = self.computeHessian(atoms)
		dDipole = self.computedDipole(atoms)
		return hessian, dDipole

	def _str_frequencies(self, omega):
		summary_lines = ['-----------------',
				 ' Mode   Frequency',
				 ' #      cm^-1 ',
				 '-----------------']
		for n, e in enumerate(omega.numpy()):
			summary_lines.append('{index:3d} {cm:13.3f}'.format(index=n, cm=e))
		summary_lines.append('-----------------')

		return summary_lines

	def _str_ir(self, omega, intensities):
		summary_lines = ['-------------------------------',
				 ' Mode   Frequency     Intensity',
				 ' #      cm^-1         km/mol',
				 '-------------------------------']
		for n, e in enumerate(omega.numpy()):
			summary_lines.append('{index:3d} {cm:13.3f} {intensity:13.3f}'.format(index=n, cm=e, intensity=intensities[n]))
		summary_lines.append('-------------------------------')

		return summary_lines
				
	def _computeNormalizedModes(self, invmasses, modes):
		# compute reduced masses  & normalized modes
		mmodes=tf.reshape(tf.transpose(modes)*invmasses,[modes.shape[0],-1,3])
		rmasses = tf.reduce_sum(mmodes*mmodes,axis=2)
		rmasses = tf.reduce_sum(rmasses,axis=1)
		rmasses = 1.0/rmasses
		mmodes=tf.reshape(mmodes,[mmodes.shape[0],-1])
		mmodes =tf.transpose(mmodes)*tf.math.sqrt(rmasses)
		mmodes =tf.reshape(tf.transpose(mmodes), [modes.shape[0],-1,3])
		rmasses /= AMU_TO_AU
		#print("rmasses=",rmasses)
		#print("mmodes=",mmodes)
		return rmasses, mmodes

	def computeHarmonicFrequencies(self, atoms):
		hessian = self.computeHessian(atoms)
		masses = tf.constant(atoms.get_masses(),dtype=hessian.dtype)
		#print("masses =",masses)
		if any(masses == 0):
			raise RuntimeError('Zero mass encountered in one or more of '
                               'the vibrated atoms. Use Atoms.set_masses()'
                               ' to set all masses to non-zero values.')

		masses *= AMU_TO_AU 
		im = tf.repeat(1.0/tf.math.sqrt(masses),3)
		mHm =tf.transpose([im])*hessian*im
		mHm = (mHm+tf.transpose(mHm))/2.0
		omega2, modes = tf.linalg.eigh(mHm)
		omega = tf.where(omega2>0, tf.math.sqrt(omega2)*AU_TO_CM1, -tf.math.sqrt(-omega2)*AU_TO_CM1)
		summary_lines = self._str_frequencies(omega)
		log_text = '\n'.join(summary_lines) + '\n'
		reduced_masses, normalized_modes = self._computeNormalizedModes(im, modes)
		return omega, modes, log_text,  reduced_masses, normalized_modes

	def computeIR(self, atoms):
		omega, modes, log_text, reduced_masses, normalized_modes = self.computeHarmonicFrequencies(atoms)
		dDipole = self.computedDipole(atoms)
		#modes = tf.transpose(modes)
		masses = tf.constant(atoms.get_masses(),dtype=omega.dtype)
		masses *= AMU_TO_AU 
		im = tf.repeat(1.0/tf.math.sqrt(masses),3)
		dDipoledq = dDipole *im
		dDipoledQ = tf.tensordot(dDipoledq,modes, axes = 1)
		intensities = tf.reduce_sum(dDipoledQ**2, axis=0)
		convToD2Ang2amum1 = (AUTODEB/BOHR_TO_ANG)**2*AMU_TO_AU   
		convTokmmolm1 = convToD2Ang2amum1*42.255
		intensities *= convTokmmolm1
		summary_lines = self._str_ir(omega, intensities)
		log_text = '\n'.join(summary_lines) + '\n'
		return omega, modes, intensities, log_text, reduced_masses, normalized_modes



	def get_potential_energy(self, atoms, force_consistent=False):
		if self.calculation_required(atoms):
			self._computeProperties(atoms)
		return self._energy

	def get_forces(self, atoms):
		if self.calculation_required(atoms):
			self._computeProperties(atoms)
		return self._forces

	def get_molcharges(self, atoms):
		if self.calculation_required(atoms):
			self._computeProperties(atoms)
		return self._molcharges

	def get_dipole(self, atoms):
		if self.calculation_required(atoms):
			self._computeProperties(atoms)
		return self._dipole

	def get_dipole_moment(self, atoms):
		if self.calculation_required(atoms):
			self._computeProperties(atoms)
		return self._dipole

	def get_atomcharges(self, atoms):
		if self.calculation_required(atoms):
			self._computeProperties(atoms)
		return self._atomcharges

	def get_charges(self, atoms):
		return self.get_atomcharges(atoms)

	@property
	def last_atoms(self):
		return self._last_atoms

	@property
	def energy(self):
		return self._energy

	@property
	def forces(self):
		return self._forces

	@property
	def charges(self):
		return self.get_atomcharges()

	@property
	def energy_stdev(self):
		return self._energy_stdev

	@property
	def sr_cutoff(self):
		return self._sr_cutoff

	@property
	def lr_cutoff(self):
		return self._lr_cutoff

	@property
	def use_neighborlist(self):
		return self._use_neighborlist

	@property
	def model(self):
        	return self._models

	@property
	def checkpoint(self):
		return self._checkpoint

	@property
	def Z(self):
		return self._Z

	@property
	def Q_tot(self):
		return self._Q_tot

	@property
	def R(self):
		return self._R

	@property
	def offsets(self):
		return self._offsets

	@property
	def idx_i(self):
		return self._idx_i

	@property
	def idx_j(self):
		return self._idx_j

	@property
	def sr_offsets(self):
		return self._sr_offsets

	@property
	def sr_idx_i(self):
		return self._sr_idx_i

	@property
	def sr_idx_j(self):
		return self._sr_idx_j

	@property
	def conv_distance(self):
		return self._conv_distance

	@property
	def conv_energy(self):
		return self._conv_energy
