from __future__ import absolute_import
import os
import math
import tensorflow as tf
from Utils.UtilsFunctions import *
from Utils.XTBInterface import *
#from EEM.EEMOld import *

from myXTB.interface import Calculator
from myXTB.utils import get_method
from myXTB.libxtb import VERBOSITY_MUTED, VERBOSITY_FULL

from tensorflow.keras.layers import Layer
import multiprocessing as mp
import random
import tensorflow_probability as tfp
import tensorflow_addons as tfa

# using xtb-python tools, it does not work because  parameters are not reloaded, if I change the parameters
def computePropertiesOneMol(xtbmethod, Z, R, molCharge=None):
	#calc = Calculator(get_method("gfn2-xtb"),Z,R)
	calc = Calculator(get_method(xtbmethod),Z,R,charge=molCharge)
	calc.set_verbosity(VERBOSITY_MUTED)
	#calc.set_verbosity(VERBOSITY_FULL)
	res = calc.singlepoint()
	energie=res.get_energy()
	Qa=res.get_charges()
	gradients=res.get_gradient()
	dipole =res.get_dipole()
	return energie, Qa, dipole, gradients


class ModelVariables(Layer):
	def __str__(self):
		return "Model Variables"

	def __init__(self,
		fileName,
		atomic_energies_filename=None,
		seed=None,
		dtype=tf.float32            #single or double precision
		):
		super().__init__(dtype=dtype, name="ModelVariables")
		self._read_parameters(fileName)
		aenergies=self._set_atomic_energies(atomic_energies_filename)
		self._atomic_energies = tf.Variable(aenergies, name="atomic_energies",dtype=dtype,trainable=True)
		self.add_xtb_variables_to_trainables()

	@property
	def methodName(self):
		return self._methodName

	@property
	def blocks(self):
		return self._blocks

	@property
	def names(self):
		return self._names
	@property
	def values(self):
		return self._values
	@property
	def steps(self):
		return self._steps
	@property
	def varnumlines(self):
		return self._varnumlines
	@property
	def varnumcols(self):
		return self._varnumcols

	@property
	def atomic_energies(self):
		return self._atomic_energies

	@property
	def xtb_parameters(self):
		return self._xtb_parameters

	def get_step(self, iv):
		i=self.varnumlines[iv]
		j=self.varnumcols[iv]
		return self.steps[i][j]

	def get_value(self, iv):
		i=self.varnumlines[iv]
		j=self.varnumcols[iv]
		return self.values[i][j]


	def add_step(self, iv):
		i=self.varnumlines[iv]
		j=self.varnumcols[iv]
		self.values[i][j] =  self.values[i][j] + self.steps[i][j]

	def sub_step(self, iv):
		i=self.varnumlines[iv]
		j=self.varnumcols[iv]
		self.values[i][j] =  self.values[i][j] - self.steps[i][j]

	def set_xtb_variables_from_trainables(self):
		xtbvar = None
		for var in self.trainable_weights:
			if var.name.find("xtb_parameters") != -1:
				xtbvar = var
				break
		k=0
		if xtbvar is not None:
			for iv in range(len(self.varnumlines)):
				i=self.varnumlines[iv]
				j=self.varnumcols[iv]
				self.values[i][j]= xtbvar[k].numpy()
				k = k + 1

	def add_xtb_variables_to_trainables(self):
		v=[]
		for iv in range(len(self.varnumlines)):
			i=self.varnumlines[iv]
			j=self.varnumcols[iv]
			v.append(self.values[i][j])
		if len(v)>0:
			print("v=",v)
			self._xtb_parameters = tf.Variable(v, name="xtb_parameters",dtype=self.dtype,trainable=True)
		else:
			self._xtb_parameters = None


	def _set_atomic_energies(self, fileName):
		nmax=95
		energies=np.zeros(nmax)
		if fileName is None:
			return energies

		if not os.path.isfile(fileName):
			print("??????????????????????????????????????????????")
			print("Sorry, I cannot locate ", fileName," file")
			print("??????????????????????????????????????????????")
			exit(1)

		f=open(fileName,"r")
		lines=f.readlines()
		for line in lines:
			ll = line.split()
			if len(ll)>=2 and int(ll[0])>0 and int(ll[0])<nmax:
				energies[int(ll[0])]=float(ll[1])
		f.close()
		return energies

	def _read_parameters(self, fileName):
		if not os.path.isfile(fileName):
			print("Sorry, I cannot locate ", fileName," file")
			exit(1)

		f=open(fileName,"r")
		lines=f.readlines()
		names = []
		values = []
		steps = []
		varnumlines=[]
		varnumcols=[]
		blocks=[]
		iline=0
		blk="$UNK"
		methodName="UNK"
		for line in lines:
			if line.find('$') != -1:
				ll = line.split()
				if len(ll)>=2:
					blk = ll[0]+ll[1]
				else:
					blk = ll[0]
			blocks.append(blk)
			lu = line.upper()
			if lu.find('NAME') != -1:
				ll = lu.split()
				if len(ll)==2 and ll[0]=='NAME':
					methodName= line.split()[1]

			if lu.find('OPT') == -1:
				names.append(line.rstrip())
				values.append(None)
				steps.append(None)
			else:
				ll = line.split()
				names.append(ll[0])
				iopt = lu.find('OPT')
				lv=line[0:lu.find('OPT')]
				llv= lv.split()
				nvalues = len(llv)-1

				ls=line[lu.find('OPT'):]
				lls= ls.split()
				ns = len(lls)-1
				if ns != nvalues:
					print("Error in line ", ll, " : Nomber of values must be = number of step")
					exit(1)
				v=[]
				s=[]
				for i in range(0,nvalues):
					v.append(float(llv[i+1]))
					s.append(float(lls[i+1]))
					varnumlines.append(iline)
					varnumcols.append(i)
				values.append(v)
				steps.append(s)
			iline=iline+1
			
			f.close()
		self._names = names
		self._values = values
		self._steps = steps
		self._varnumlines = varnumlines 
		self._varnumcols = varnumcols 
		self._blocks = blocks 
		self._methodName = methodName 
		if len(varnumlines) <1:
			print("WARNING : number of fitted parameters = 0")

	def save_parameters(self, filename):
		with open(filename, "w") as f:
			nlines = len(self.names)
			for i in range(0,nlines):
				f.write(self.names[i])
				if self.values[i] is not None:
					nvalues = len(self.values[i])
					for j in range(0,nvalues):
						f.write(" "+str(self.values[i][j]))
					f.write(" opt")
					for j in range(0,nvalues):
						f.write(" "+str(self.steps[i][j]))
				f.write("\n")
			f.close()

	def create_xtbfile(self, fileName):
		f = open(fileName, "w")
		nlines = len(self.names)
		for i in range(0,nlines):
			f.write(self.names[i])
			if self.values[i] is not None:
				nvalues = len(self.values[i])
				for j in range(0,nvalues):
					f.write(" "+str(self.values[i][j]))
			f.write("\n")
		f.close()

	def print_variables(self):
		self.set_xtb_variables_from_trainables()
		for var in self.trainable_weights:
			if var.name.find("atomic_energies") != -1:
				print("---------------------------------------------------------")
				print("Atomic energies: ",var[1].numpy(), var[6].numpy())
				break
		print("---------------------------------------------------------")
		print("Liste of XTB fitted paremeters Method =",self.methodName)
		print("---------------------------------------------------------")
		for iv in range(len(self.varnumlines)):
			i=self.varnumlines[iv]
			j=self.varnumcols[iv]
			print(self.blocks[i], "/ ", self.names[i],  " Values ", self.values[i][j], "\t Steps=",  self.steps[i][j])
		print("==========================================================")

	def increment_variable(self, num):
		if num <0 or num >= len(self.varnumlines):
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			print("Error, total number of variables = ", len(self.varnumlines))
			print("       index must be between 0 and ", len(self.varnumlines)-1)
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			exit(1)
		i=self.varnumlines[num]
		j=self.varnumcols[num]
		self._values[i][j] = self._values[i][j] + self.steps[i][j]

	def decrement_variable(self, num):
		if num <0 or num >= len(self.varnumlines):
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			print("Error, total number of variables = ", len(self.varnumlines))
			print("       index must be between 0 and ", len(self.varnumlines)-1)
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			exit(1)
		i=self.varnumlines[num]
		j=self.varnumcols[num]
		self._values[i][j] = self._values[i][j] - self.steps[i][j]

class PhysModelXTB(tf.keras.Model):

	def __str__(self):
		return "PhysModelXTB"

	def __init__(self,
		xtb_file_parameters,              # xtb file parameters, required for xtb model
		xtb_file_best_parameters,         # xtb file best parameters, required for xtb model
		xtb_working_directory,            # save xtb file in working directory
		dtype=tf.float32,                 # single or double precision
		energy_weight=1,   		  # energy contribution to the loss function
		force_weight=100,   		  # force contribution to the loss function
		charge_weight=1.0, 		  # charge contribution to the loss function
		atomic_charge_weight=0.0,	  # atomic charge contribution to the loss function
		dipole_weight=1.0, 		  # dipole contribution to the loss function
		nhlambda=0,			  # lambda multiplier for non-hierarchicality loss (regularization)
		loss_type=0,			  # loss type (0=> Mean, 1=>RMSE)
		atomic_energies_filename=None,    # atomic_energies_filename=None, starting atomic energies, if not atomic energies=0
		seed=None):
		super().__init__(dtype=dtype, name="PhysModelXTB")


		self._dtype=dtype
		self._xtb_file_parameters=xtb_file_parameters
		self._xtb_file_best_parameters=xtb_file_best_parameters
		if xtb_working_directory is not None:
			os.environ['XTBPATH']=xtb_working_directory
			print("XTBPATH=",os.environ['XTBPATH'])

		self.sup_variables=ModelVariables(xtb_file_parameters, atomic_energies_filename=atomic_energies_filename, dtype=dtype, seed=seed)
		self._fileXTB=xtb_working_directory+"/param_"+self.sup_variables.methodName.lower()+".txt"
		self._xtb_working_directory=xtb_working_directory

		self._energy_weight=tf.constant(energy_weight,dtype=dtype)
		self._force_weight=tf.constant(force_weight,dtype=dtype)
		self._charge_weight=tf.constant(charge_weight,dtype=dtype)
		self._atomic_charge_weight=tf.constant(atomic_charge_weight,dtype=dtype)
		self._dipole_weight=tf.constant(dipole_weight, dtype=dtype)
		self._nhlambda=nhlambda
		self._loss_type=loss_type

		self.create_xtbfile()

	# using xtb-python tools, it does not work because  parameters are not reloaded, if I change the parameters
	def computePropertiesOneMolXTBLIB(self,xtbmethod, Z, R, molCharge=None):
		workingDirectory="/tmp/"+id_generator()
		filextb=workingDirectory+"/param_"+self.sup_variables.methodName.lower()+".txt"
		if not os.path.exists(workingDirectory):
			os.makedirs(workingDirectory)
		self.sup_variables.create_xtbfile(filextb)
		calc = Calculator(get_method(xtbmethod),Z,R, filename=filextb, charge=molCharge)

		# it does not work
		#os.environ['XTBPATH']=workingDirectory
		#calc = Calculator(get_method(xtbmethod),Z,R)

		calc.set_verbosity(VERBOSITY_MUTED)
		#calc.set_verbosity(VERBOSITY_FULL)
		res = calc.singlepoint()
		energie=res.get_energy()
		Qa=res.get_charges()
		gradients=res.get_gradient()
		dipole =res.get_dipole()
		remove_directory(workingDirectory)
		return energie, Qa, dipole, gradients

	def computePropertiesOneMol(self,xtbmethod, Z, R, molCharge=None):
		xtb_nterface = XTBInterface(xtbmethod,dtype=self.dtype)
		#print("getmethod=",get_method(xtbmethod))
		# change directory w if not parametres are not reloaded !
		workingDirectory="/tmp/"+id_generator()
		filextb=workingDirectory+"/param_"+self.sup_variables.methodName.lower()+".txt"
		if not os.path.exists(workingDirectory):
			os.makedirs(workingDirectory)
		self.sup_variables.create_xtbfile(filextb)
		#energy, gradients, Qa, outputFile = xtb_nterface.run(Z,R,workingDirectory, self.fileXTB)
		energy, gradients, Qa, outputFile = xtb_nterface.run(Z,R,workingDirectory, filextb, charge=molCharge)
		dipole = tf.zeros(3,dtype=self.dtype)
		QR = tf.stack([Qa*R[:,0], Qa*R[:,1], Qa*R[:,2]],1)
		batch_seg = tf.zeros_like(Z)
		#print("QR=",QR)
		dipole = tf.math.segment_sum(QR, batch_seg)

		#print("energy=",energy)
		#print("Qa=",Qa)
		#print("gradients=",gradients)
		#print("dipole=",dipole)
		remove_directory(workingDirectory)
		return energy, Qa, dipole, gradients


	def computeProperties(self, data):
		#print(data)
		Q_tot=data['Q']
		Z=data['Z']
		R=tf.Variable(data['R'],dtype=self.dtype)
		batch_seg = data['batch_seg']
		idx_i=data['idx_i']
		idx_j=data['idx_j']
		if batch_seg is None:
			batch_seg = tf.zeros_like(Z)
		index=tf.constant(range(len(batch_seg)),dtype=tf.int32)
		#number of atoms per batch
		Na_per_batch = tf.math.segment_sum(tf.ones_like(batch_seg, dtype=self.dtype), batch_seg)
		if Q_tot is None: #assume desired total charge zero if not given
			Q_tot = tf.zeros_like(Na_per_batch, dtype=self.dtype)
		
		energies = []
		forces = []
		charges = []
		dipoles = []
		nhloss = 0
		nb=0
		#pool = mp.Pool(mp.cpu_count())
		#print("#procs=",mp.cpu_count())
		xtbmethod=self.xtb_method()
		for imol in range(len(Q_tot)):
			nAtoms=int(Na_per_batch[imol])
			ne=nb+nAtoms
			Rmol = tf.gather(R, index[nb:ne])
			Zmol = tf.gather(Z, index[nb:ne])
			ea = tf.reduce_sum(tf.gather(self.sup_variables._atomic_energies, Zmol))
			#print("imol=",imol," ea=",ea)
			#print("aener =", tf.gather(self.sup_variables._atomic_energies, Zmol))
			#if self.sup_variables._xtb_parameters is None:
				#print("xttrainable =", self.sup_variables._xtb_parameters)
			#energieMol, chargesMol, dipoleMol, gradMol = pool.apply(computePropertiesOneMol, args=(xtbmethod, Zmol.numpy(), Rmol.numpy(),molCharge=Q_tot[imol]))

			# using myXTB interface :using files, so more timing consoming
			#energieMol, chargesMol, dipoleMol, gradMol = self.computePropertiesOneMol(xtbmethod, Zmol.numpy(), Rmol.numpy(), molCharge=Q_tot[imol])

			# using xtb-python : warning problem is you want to change parameters
			#energieMol, chargesMol, dipoleMol, gradMol = computePropertiesOneMol(xtbmethod, Zmol.numpy(), Rmol.numpy(), molCharge=Q_tot[imol])

			# using xtb-python : work only with my XTB-python modified version
			energieMol, chargesMol, dipoleMol, gradMol = self.computePropertiesOneMolXTBLIB(xtbmethod, Zmol.numpy(), Rmol.numpy(), molCharge=Q_tot[imol])

			energieMol += ea
			energies.append(energieMol)
			charges.extend(chargesMol)
			forces.extend(gradMol)
			dipoles.append(dipoleMol)
			nb=ne
		#pool.close()

		#print("charges=",charges)
		Qa=tf.Variable(charges, dtype=self.dtype)
		charges = tf.squeeze(tf.math.segment_sum(Qa, batch_seg))
		"""
		cast => probelem with gradient
		energies=tf.cast(tf.Variable(energies), self.dtype)
		charges=tf.cast(tf.Variable(charges),self.dtype)
		dipoles=tf.cast(tf.Variable(dipoles), self.dtype)
		forces=tf.cast(tf.Variable(forces), self.dtype)
		"""
		dipoles=tf.Variable(dipoles, self.dtype)
		forces=tf.Variable(forces,dtype=self.dtype)
		forces=-forces
		nhloss=tf.Variable(nhloss, dtype=self.dtype)
		energies=tf.Variable(energies, dtype=self.dtype)
		return energies, charges, Qa, dipoles, forces, nhloss

	def computeAtomicEnergiesAndForces(self, data):
		print("====================== Error ==============================");
		print("computeProperties not yet implemented in PhysModelXTB.py");
		print("====================== Error ==============================");
		sys.exit()
		return None, None, None, None

	def getLoss(self, weight, values):
		if self.loss_type==0:
			return  weight*tf.reduce_mean(values)
		else:
			return weight*tf.math.sqrt(tf.reduce_mean(values*values))

	def computeOnlyLoss(self, data):
		energies, charges, Qa, dipoles, forces, nhloss = self.computeProperties(data)
		loss = 0
		"""
		if self.energy_weight > 0:
			de = tf.abs(energies-tf.constant(data['E'],dtype=self.dtype))
			loss +=  self.energy_weight*tf.reduce_mean(de)
		#print("charges=",charges)
		if self.charge_weight > 0:
			dq = tf.abs(charges-tf.constant(data['Q'],dtype=self.dtype))
			loss += self.charge_weight*tf.reduce_mean(dq)
		if self.atomic_charge_weight > 0:
			dqa = tf.abs(Qa-tf.constant(data['Qa'],dtype=self.dtype))
			loss += self.atomic_charge_weight*tf.reduce_mean(dqa)
		if self.dipole_weight > 0:
			dd = tf.abs(dipoles-tf.constant(data['D'],dtype=self.dtype))
			loss +=self.dipole_weight*tf.reduce_mean(dd)
		if self.force_weight > 0:
			df = tf.abs(forces-tf.constant(data['F'],dtype=self.dtype))
			loss += self.force_weight*tf.reduce_mean(df)
		if self.nhlambda>0:
			loss += self.nhlambda*nhloss
		"""

		if self.energy_weight > 0:
			de = tf.abs(energies-tf.constant(data['E'],dtype=self.dtype))
			loss += self.getLoss(self.energy_weight,de)
		if self.charge_weight > 0:
			dq = tf.abs(charges-tf.constant(data['Q'],dtype=self.dtype))
			loss += self.getLoss(self.charge_weight,dq)
		if self.atomic_charge_weight > 0:
			dqa = tf.abs(Qa-tf.constant(data['Qa'],dtype=self.dtype))
			loss += self.getLoss(self.atomic_charge_weight,dqa)
		if self.dipole_weight > 0:
			dd = tf.abs(dipoles-tf.constant(data['D'],dtype=self.dtype))
			loss += self.getLoss(self.dipole_weight,dd)
		if self.force_weight > 0:
			df = tf.abs(forces-tf.constant(data['F'],dtype=self.dtype))
			loss += self.getLoss(self.force_weight,df)
		if self.nhlambda>0:
			loss += self.nhlambda*nhloss

		return energies, charges, Qa, dipoles, forces , loss

	def computeGradientAtomicEnergies(self, data):
		with tf.GradientTape() as g:
			g.watch(self.trainable_weights)
			energies, charges, Qa, dipoles, forces, loss = self.computeOnlyLoss(data)
		gradients = g.gradient(loss, self.trainable_weights)
		for var in self.trainable_weights:
			if var.name.find("atomic_energies") != -1:
				#print("trained atomic energies: ",var[1].numpy(), var[6].numpy())
				lis = []
				for i in range(var.shape[0]):
					if tf.math.abs(var[i])>1e-10:
						lis.append(var[i].numpy())
				if len(lis)>0:
					print("trained atomic energies: ",lis)

		return energies, charges, Qa, dipoles, forces , loss, gradients

	def computeGradientXTBParameters(self, data):
		self.sup_variables.set_xtb_variables_from_trainables()
		self.create_xtbfile()
		energies0, charges0, Qa0, dipoles0, forces0 , loss0 = self.computeOnlyLoss(data)
		#print("v0=",self.sup_variables.get_value(0))
		#print("loss0=",loss0, "e0=",energies0)
		for var in self.trainable_weights:
			if var.name.find("xtb_parameters") != -1:
				print("trained xtb parameters: ",var.numpy())
		grads =[]
		for iv in range(len(self.sup_variables.varnumlines)):
			self.sup_variables.add_step(iv)
			#self.create_xtbfile()
			#print("vp=",self.sup_variables.get_value(iv))
			energiesp, chargesp, Qap, dipolesp, forcesp , lossp = self.computeOnlyLoss(data)
			# g = (lossp-loss0)/self.sup_variables.get_step(iv) # one point
			self.sup_variables.sub_step(iv)
			self.sup_variables.sub_step(iv)
			#self.create_xtbfile()
			#print("vm=",self.sup_variables.get_value(iv))
			energiesm, chargesm, dipolesm, Qam, forcesm , lossm = self.computeOnlyLoss(data)
			self.sup_variables.add_step(iv)
			#self.create_xtbfile()
			g = (lossp-lossm)/(2*self.sup_variables.get_step(iv)) # 2 points
			#print("lossp=",lossp, "ep=",energiesp)
			#print("lossm=",lossm, "em=",energiesm)
			grads.append(g)
		grads = tf.reshape(grads,shape=[len(grads)])
		return grads


	def computeLoss(self, data):
		energies, charges, Qa, dipoles, forces , loss, gradients =  self.computeGradientAtomicEnergies(data)
		#print("trainable_weigths=", self.trainable_weights)
		#print("gradients=",gradients)
		gg=[]
		for g in gradients:
			if isinstance(g, tf.IndexedSlices) is True:
				g=convert_indexed_slices_to_tensor(g)
				#print("g=",g)
				gg.append(g)
		# grads of xtb parameters
		if self.sup_variables.xtb_parameters is not None:
			gradxtb = self.computeGradientXTBParameters(data)
			#print("gradxtb=",gradxtb)
			gg.append(gradxtb)
		#print("gg=",gg)
		#return energies, charges, Qa, dipoles, forces , loss, gradients
		return energies, charges, Qa, dipoles, forces , loss, gg

	def __call__(self, data, closs=True):
		if closs is not True:
			energies, charges, Qa, dipoles, forces, nhloss = self.computeProperties(data)
			loss=None
			gradients=None
		else:
			energies, charges, Qa, dipoles, forces , loss, gradients = self.computeLoss(data)
			
		return energies, charges, Qa, dipoles, forces, loss, gradients

	def print_parameters(self):
		self.print_variables()

	def save_parameters(self):
		self.sup_variables.save_parameters(self.xtb_file_best_parameters)

	def save_weights(self,fname):
		self.sup_variables.save_parameters(self.xtb_file_best_parameters)
		supper.save_weights(fname)

	def create_xtbfile(self):
		self.sup_variables.create_xtbfile(self.fileXTB)

	def set_xtb_variables_from_trainables(self):
		self.sup_variables.set_xtb_variables_from_trainables()

	def print_variables(self):
		self.sup_variables.print_variables()

	def xtb_method(self):
		return self.sup_variables.methodName

	def increment_variable(self, num):
		self.sup_variables.increment_variable(num)

	def decrement_variable(self, num):
		self.sup_variables.decrement_variable(num)

	def load_weights(self,fname):
		supper.load_weights(fname)
		self.sup_variables.set_xtb_variables_from_trainables()

	@property
	def xtb_working_directory(self):
		return self._xtb_working_directory

	@property
	def fileXTB(self):
		return self._fileXTB
	@property
	def xtb_file_parameters(self):
		return self._xtb_file_parameters
	@property
	def xtb_file_best_parameters(self):
		return self._xtb_file_best_parameters

	@property
	def dtype(self):
		return self._dtype

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
	def nhlambda(self):
		return self._nhlambda

	@property
	def loss_type(self):
		return self._loss_type

	@property
	def lr_cut(self):
		return None

	@property
	def sr_cut(self):
		return None

