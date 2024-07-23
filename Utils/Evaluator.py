import tensorflow as tf
import os
from Utils.DataContainer import *
from Utils.DataProvider import *
from PhysModel.PhysModel import *

def getExampleData():
	data = {}
	data['Z'] =  [1,6]
	data['R'] =  [[0.0,0.0,0.0],[1.0,0.0,0.0]]
	data['Q'] =  [0]
	data['idx_i'] =  [0]
	data['idx_j'] =  [1]
	data['batch_seg'] =   tf.zeros_like(data['Z'])
	data['offsets'] =   None
	data['sr_offsets'] =   None
	data['sr_idx_i'] =  None
	data['sr_idx_j'] =  None
	return data

class Evaluator:
	def __init__(self, 
		models_directories,               # directories containing fitted models (can also be a list for ensembles)
		dataFile=os.path.join("Data","sn2_reactions.npz"), 
		nvalues=-1,
		batch_size=1, 
		convDistanceToBohr=1.0, 
		convEnergyToHartree=1.0, 
		activation_fn=shifted_softplus,
		convDipoleToAU=1.0):
		self._data=DataContainer(dataFile,convDistanceToBohr=convDistanceToBohr, convEnergyToHartree=convEnergyToHartree, convDipoleToAU=convDipoleToAU)
		print("data shape = ",self.data.N.shape)
		if nvalues<0:
			ntrain = self.data.N.shape[0]
		else:
			ntrain = nvalues
		print("nvalues=",ntrain)
		self._nvalues=ntrain
		nvalid=0
		ntest=0
		valid_batch_size=0
		self._dataProvider=DataProvider(self.data,ntrain, nvalid, ntest=ntest, batch_size=batch_size,valid_batch_size=valid_batch_size)

		if(type(models_directories) is not list):
			self._models_directories=[models_directories]
		else:
			self._models_directories=models_directories


		self._models = []
		n=0
		for directory in self._models_directories:
			args = read_model_parameters(directory)
			dispersionParameters=getDispersionParameters(args)
			best_dir = os.path.join(args.directory, 'best')
			best_checkpoint = os.path.join(best_dir, 'best.ckpt')
			best_xtbparameters = os.path.join(best_dir, 'best_xtb_parameters.txt')

			physModel = PhysModel (
			F=args.num_features,
			K=args.num_basis,
			sr_cut=args.cutoff,
			dtype=tf.float64 if args.dtype=='float64' else tf.float32, 
			num_blocks=args.num_blocks, 
			num_residual_atomic=args.num_residual_atomic,
			num_residual_interaction=args.num_residual_interaction,
			num_residual_output=args.num_residual_output,
			model=args.model,
			activation_fn=activation_fn,
			energy_weight=args.energy_weight,
			force_weight=args.force_weight,
			charge_weight=args.charge_weight,
			atomic_charge_weight=args.atomic_charge_weight,
			dipole_weight=args.dipole_weight,
			use_scaled_charges=(args.use_scaled_charges==1),
			use_electrostatic=(args.use_electrostatic==1),
			use_dispersion=(args.use_dispersion==1),
			dispersionParameters=dispersionParameters,
			orbfile=args.orbfile,
			xtb_file_parameters=best_xtbparameters, 
			xtb_file_best_parameters=best_xtbparameters,
			xtb_working_directory="/tmp", 
			)
			data = getExampleData()
			energies, charges, Qa, dipoles, forces, loss, gradients = physModel(data,closs=False) # to set auto shape
			if str(physModel) == "PhysModelXTB" :
				physModel.physModel.sup_variables.set_xtb_variables_from_trainables()
			#print("best_checkpoint=",best_checkpoint)
			physModel.load_weights(best_checkpoint)
			self._models.append(physModel)
			self._set_weights()

	def _set_weights(self):
		self._energy_weight = -1
		self._force_weight = -1
		self._charge_weight = -1
		self._atomic_charge_weight = -1
		self._dipole_weight = -1
		nModel=len(self._models)
		for i in range(nModel):
			self._energy_weight = max(self._models[i].energy_weight, self._energy_weight)
			self._force_weight = max(self._models[i].force_weight, self._force_weight)
			self._charge_weight = max(self._models[i].charge_weight, self._charge_weight)
			self._atomic_charge_weight = max(self._models[i].atomic_charge_weight, self._atomic_charge_weight)
			self._dipole_weight = max(self._models[i].dipole_weight, self._dipole_weight)

	def computeProperties(self, data):
		n=0
		nModel=len(self._models)
		energies = None
		molcharges = None
		atomcharges = None
		dipoles = None
		forces = None
		nhloss =None
		for i in range(nModel):
			lenergies, lmolcharges, latomcharges, ldipoles, lforces, lnhloss = self._models[i].computeProperties(data)
			#print("energy=",energy)
			#print("atomcharges=",atomcharges)

			if i == 0:
				energies  = lenergies
				molcharges = lmolcharges
				atomcharges  = latomcharges
				dipoles  = ldipoles
				forces  = lforces
				nhloss = lnhloss
			else:
				n = i+1
				if lenergies is not None: 
					energies = energies + (lenergies-energies)/n
				if latomcharges is not None: 
					atomcharges = atomcharges + (latomcharges-atomcharges)/n
				if lmolcharges is not None: 
					molcharges = molcharges+ (lmolcharges-molcharges)/n
				if lforces is not None: 
					forces =  forces + (lforces-forces)/n 
				if ldipoles is not None: 
					dipoles =  dipoles + (ldipoles-dipoles)/n 
				if lnhloss is not None: 
					nhloss = nhloss+ (lnhloss-nhloss)/n 

		return energies, molcharges, atomcharges, dipoles, forces, nhloss


	def computeSums(self, data):
		energies, charges, Qa, dipoles, forces, nhloss = self.computeProperties(data)
		sums= {}
		coefs = { 'E':self.energy_weight, 'F':self.force_weight, 'Q':self.charge_weight, 
			  'Qa':self.atomic_charge_weight, 'D':self.dipole_weight
			}
		values = { 'E':energies, 'F':forces, 'Q':charges, 'Qa':Qa, 'D':dipoles}
		for key in coefs:
			if coefs[key] > 0:
				sData=tf.reduce_sum(tf.reshape(tf.constant(data[key],dtype=self.model.physModel.dtype),[-1]))
				sPredict=tf.reduce_sum(tf.reshape(values[key],[-1]))
				s2Data=tf.reduce_sum(tf.square(tf.reshape(tf.constant(data[key],dtype=self.model.physModel.dtype),[-1])))
				s2Predict=tf.reduce_sum(tf.square(tf.reshape(values[key],[-1])))
				sDataPredict=tf.reduce_sum((tf.reshape(tf.constant(data[key],dtype=self.model.physModel.dtype),[-1]))*tf.reshape(values[key],[-1]))
				s2DataPredict=tf.reduce_sum(tf.square((tf.reshape(tf.constant(data[key],dtype=self.model.physModel.dtype),[-1]))*tf.reshape(values[key],[-1])))
				sAbsDataMPredict=tf.reduce_sum(tf.math.abs((tf.reshape(tf.constant(data[key],dtype=self.model.physModel.dtype),[-1]))-tf.reshape(values[key],[-1])))
				d = {}
				d['sData']         =  sData
				d['sPredict']      =  sPredict
				d['s2Data']        =  s2Data
				d['s2Predict']     =  s2Predict
				d['sDataPredict']  =  sDataPredict
				d['s2DataPredict'] =  s2DataPredict
				d['sAbsDataMPredict'] = sAbsDataMPredict
				d['n'] =  tf.reshape(values[key],[-1]).shape[0]
				sums[key] = d

		return sums

	def addSums(self, data, sums=None):
		coefs = { 'E':self.energy_weight, 'F':self.force_weight, 'Q':self.charge_weight, 
			  'Qa':self.atomic_charge_weight, 'D':self.dipole_weight
			}
		if sums is None:
			sums= {}
			lis=[ 'sData', 'sPredict', 's2Data', 's2Predict', 'sDataPredict', 's2DataPredict', 'sAbsDataMPredict','n']
			for key in coefs:
				if coefs[key] > 0:
					d = {}
					for name in lis:
						d[name]  =  0
					sums[key] = d
		s=self.computeSums(data)
		for keys in sums:
			for key in sums[keys]:
				sums[keys][key] +=  s[keys][key]
		return sums

	def addTrainSums(self, sums=None):
		sums=self.addSums(self.dataProvider.current_batch(), sums=sums)
		return sums


	def computeAccuraciesFromSums(self, sums, verbose=False):
		coefs = { 'E':self.energy_weight, 'F':self.force_weight, 'Q':self.charge_weight, 
			  'Qa':self.atomic_charge_weight, 'D':self.dipole_weight
			}
		acc = {}
		mae = {}
		ase = {}
		rmse = {}
		R2 = {}
		rr = {}
		for keys in sums:
			mae[keys] =  sums[keys]['sAbsDataMPredict']/sums[keys]['n']
			m =  sums[keys]['sData']/sums[keys]['n']
			residual =  sums[keys]['s2Data'] +sums[keys]['s2Predict'] -2*sums[keys]['sDataPredict']
			if tf.math.abs(residual)<1e-14:
				R2[keys] = tf.Variable(1.0,dtype=sums[keys]['sData'].dtype)
			else:
				total =  sums[keys]['s2Data']-2*m*sums[keys]['sData']+sums[keys]['n']*m*m
				R2[keys] = 1.0-residual/(total+1e-14)

			ymean = m
			ypredmean = sums[keys]['sPredict']/sums[keys]['n']
			ase[keys] = ypredmean-ymean 
			rmse[keys] = tf.math.sqrt(tf.math.abs(residual/sums[keys]['n']))

			yvar =  sums[keys]['s2Data']-2*ymean*sums[keys]['sData']+sums[keys]['n']*ymean*ymean
			ypredvar =  sums[keys]['s2Predict']-2*ypredmean*sums[keys]['sPredict']+sums[keys]['n']*ypredmean*ypredmean
			cov =  sums[keys]['sDataPredict']-ypredmean*sums[keys]['sData']-ymean*sums[keys]['sPredict']+sums[keys]['n']*ymean*ypredmean
			den = tf.sqrt(yvar*ypredvar)
			if den<1e-14:
				corr = tf.Variable(1.0,dtype=sums[keys]['sData'].dtype)
			else:
				corr = cov/tf.sqrt(yvar*ypredvar+1e-14)
			rr[keys] = corr*corr

		loss=0.0
		for key in coefs:
			#if mae[key] in locals()  and coefs[key]>0:
			if coefs[key]>0:
				loss += mae[key]*coefs[key]
		lossDic ={ 'L':loss}

		acc['mae'] = mae
		acc['ase'] = ase
		acc['rmse'] = rmse
		acc['R2'] = R2
		acc['r2'] = rr
		acc['Loss'] = lossDic
		if verbose is True:
			for keys in acc:
				#print(keys,":")
				[ print(keys,"[", key,"]=", acc[keys][key].numpy()) for key in acc[keys] ]
				print("")
		return acc

	def computeLossFromSums(self, sums):
		coefs = { 'E':self.energy_weight, 'F':self.force_weight, 'Q':self.charge_weight, 
			  'Qa':self.atomic_charge_weight, 'D':self.dipole_weight
			}
		mae = {}
		for keys in sums:
			mae[keys] =  sums[keys]['sAbsDataMPredict']/sums[keys]['n']

		loss=0.0
		for key in coefs:
			if coefs[key]>0:
				loss += mae[key]*coefs[key]
		return loss.numpy()

	def computeAccuracies(self, verbose=True):
		sums= None
		nsteps = self.dataProvider.get_nsteps_batch()
		for i in range(nsteps):
			if verbose is True:
				print("Step : ", i+1, " / ", nsteps)
			dt = self.dataProvider.next_batch()
			sums=self.addSums(dt, sums)
		acc = self.computeAccuraciesFromSums(sums,verbose)
		return acc

	"""
		Save reference and predicted values in text file, as 2 columns
	"""
	def saveAnalysisOneData(self, ref, pred, fileOut):
		rref = tf.reshape(tf.convert_to_tensor(ref,dtype=self.model.physModel.dtype),[-1])
		rpred = tf.reshape(tf.convert_to_tensor(pred,dtype=self.model.physModel.dtype),[-1])
		for i in range(rref.shape[0]):
			fileOut.write(" "+'{:20.14e}'.format(rref.numpy()[i])+" "+'{:20.14e}'.format(rpred.numpy()[i])+"\n")

	def saveAnalysisData(self, data, files):
		energies, charges, Qa, dipoles, forces, nhloss = self.computeProperties(data)
		coefs = { 'E':self.energy_weight, 'F':self.force_weight, 'Q':self.charge_weight, 
			  'Qa':self.atomic_charge_weight, 'D':self.dipole_weight
			}
		values = { 'E':energies, 'F':forces, 'Q':charges, 'Qa':Qa, 'D':dipoles}
		for key in coefs:
			if coefs[key] > 0:
				self.saveAnalysisOneData(data[key], values[key], files[key])

	def addTitleAnalysis(self, fileOut, t):
		fileOut.write("###########################################################################\n")
		fileOut.write("#"+'{:20s}'.format(t)+"\n")
		fileOut.write("###########################################################################\n")

	def saveAnalysis(self, metrics_dir, verbose=True, uid=None):
		if not os.path.exists(metrics_dir):
			os.makedirs(metrics_dir)
		prefix=os.path.join(metrics_dir,"Evaluation")
		if uid is not None:
			prefix=prefix+"_"+str(uid)

		fileNames= {}
		titles = {}
		if self.energy_weight > 0:
			fileNames['E'] = prefix+"_energies.txt"
			s="energies"
			titles['E'] ="#"+'{:20s}'.format("Reference "+s)+" "+'{:20s}'.format("Predicted "+s)

		if self.force_weight > 0:
			fileNames['F'] = prefix+"_forces.txt"
			s="forces"
			titles['F'] ="#"+'{:20s}'.format("Reference "+s)+" "+'{:20s}'.format("Predicted "+s)

		if self.charge_weight > 0:
			fileNames['Q'] = prefix+"_mol_charges.txt"
			s="molecular charges"
			titles['Q'] ="#"+'{:20s}'.format("Reference "+s)+" "+'{:20s}'.format("Predicted "+s)

		if self.atomic_charge_weight > 0:
			fileNames['Qa'] = prefix+"_atomic_charges.txt"
			s="atomic charges"
			titles['Qa'] ="#"+'{:20s}'.format("Reference "+s)+" "+'{:20s}'.format("Predicted "+s)

		if self.dipole_weight > 0:
			fileNames['D'] = prefix+"_dipoles.txt"
			s="dipoles"
			titles['D'] ="#"+'{:20s}'.format("Reference "+s)+" "+'{:20s}'.format("Predicted "+s)

		files= {}
		for key in fileNames:
			files[key] = open(fileNames[key],"w")
			self.addTitleAnalysis(files[key], titles[key])

		nsteps = self.dataProvider.get_nsteps_batch()

		for i in range(nsteps):
			if verbose is True:
				print("Step : ", i+1, " / ", nsteps)
			dt = self.dataProvider.next_batch()
			self.saveAnalysisData(dt, files)

		for key in files:
			files[key].close()

		return fileNames


	@property
	def data(self):
		return self._data
	@property
	def dataProvider(self):
		return self._dataProvider

	@property
	def models(self):
		return self._models

	@property
	def model(self):
		return self._models[0]
    
	@property
	def energy_weight(self):
		return self._energy_weight

	@property
	def force_weight(self):
		return self._force_weight

	@property
	def charge_weight(self):
		return self._charge_weight

	@property
	def atomic_charge_weight(self):
		return self._atomic_charge_weight

	@property
	def dipole_weight(self):
		return self._dipole_weight

	@property
	def nvalues(self):
		return self._nvalues
