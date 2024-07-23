from __future__ import absolute_import
import os
import sys
import tensorflow as tf
from Utils.UtilsFunctions import *
from PhysModel.PhysModelStandard import *
from PhysModel.PhysModelEEM import *
from PhysModel.PhysModelEEMPot import *
from PhysModel.PhysModelGFN1 import *
#from PhysModel.PhysModelXTB import *

from tensorflow.keras.layers import Layer

class PhysModel(tf.keras.Model):
	def __str__(self):
		return str(self._physModel)

	def __init__(self,
		F=None,                          #dimensionality of feature vector
		K=None,                          #number of radial basis functions
		kmax=[2,2,2],                     # kmax for ewald sum
		sr_cut=None,                     #cutoff distance for short range interactions
		lr_cut = None,                    #cutoff distance for long range interactions (default: no cutoff)
		num_scc=0,                        #number of cycles of updating of spin atomic charges (0 => no cycle)
		em_type = 0,       		  #  Elemental modes type : 0(default) => only Z, 1=> Z + Masses, 2=> Z+Masses+QaAlpha+QaBeta (so we use charge &multiplicity)
		num_hidden_nodes_em = None,       # number of nodes on each hidden layer in elemental modes block
		num_hidden_layers_em = 2,         #  number of hidden layers in elemental modes block
		num_blocks=5,                     #number of building blocks to be stacked
		num_residual_atomic=2,            #number of residual layers for atomic refinements of feature vector
		num_residual_interaction=3,       #number of residual layers for refinement of message vector
		num_residual_output=1,            #number of residual layers for the output blocks
		electrostatic_model=None,         # electrostatic model None, Ewald/Coulomb, Coulomb, EEM, EEMPot, GFN1, XTB
                drop_rate=None,                   #initial value for drop rate (None=No drop)
		activation_fn=shifted_softplus,   #activation function
		dtype=tf.float32,                 #single or double precision
		energy_weight=1.0,   		  # force contribution to the loss function
		force_weight=100.0,   		  # force contribution to the loss function
		charge_weight=1.0, 		  #charge contribution to the loss function
		atomic_charge_weight=0.0,	  # atomic charge contribution to the loss function
		dipole_weight=1.0, 		  # dipole contribution to the loss function
		scale_shift_output=False,         # scale & shift output
		use_scaled_charges=True,          # use scaled charges in electrostatic energy & dipole
		use_dispersion=True,		  # use dispersion in energy prediction 
		dispersionParameters=None,        # dispersion parameters. If None => trainable
		nhlambda=0,			  # lambda multiplier for non-hierarchicality loss (regularization)
		loss_type=0,			  # loss type (0=> Mean, 1=>RMSE)
		eem_fit_parameters=0,		  # fit eem parameters (0 no, 1 fit J, 2 fit alp, 3 fit J & alp
		gfn_fit_parameters=0,		  # fit gfn parameters (0 no, 1 fit eta, 2 fit gamma, 3 fit eta & gamma
		orbfile=None,      	  	  # read orbital for each atom type : Format : Z l1 l2 ...lmax for each Z

		xtb_file_parameters=None,         # xtb file parameters, required for xtb model
		xtb_file_best_parameters=None,    # xtb file best parameters, required for xtb model
		xtb_working_directory=None,            # save xtb file in workingDirectory, required for xtb model
		atomic_energies_filename=None,      # atomicEnergiesfileName=None, starting atomic energies, if not atomic energies=0

		basis_type="Default",           # radial basis type : GaussianNet (Default for MPNN), Gaussian(Default for EANN), Bessel, Slater, 
		# EAM specific parameters

		nn_model="MPNN",			# MPNN (Message-Passing Neural network), EANN (Embedded Atom Neural Network), EAMP (Embedded Atom Message-Passing Neural network), EANNP (Embedded Atom Pairs Neural Network)
		# Specific parameters for EANN
		Lmax=2,                           	# Lmax (same value for all atoms)
		beta=0.2,                       	# beta value for basis functions
		num_interaction_layers=1,     		# number of hidden layers for interaction block (1 at least)
		num_output_layers=1,          		# number of hidden layers for the output blocks (can be =0 !)
		num_interaction_nodes=10,     		# number of nodes in each hidden layer of interaction block (1 at least)
		num_output_nodes=10,          		# number of nodes in each hidden layer in the output block 
		repulsion_model="None",             # None, ZBL or GFN1
		repulsion_fit_parameters=1,       # 0, 1 , 2   or 3
		type_output=0,                  # 0=> average, 1=> Wx+b (modules outputs)
		seed=None):
		super().__init__(dtype=dtype, name="PhysModel")

		self._physModel=None
		if electrostatic_model=="XTB" or nn_model is None:
			self._physModel=PhysModelXTB(
						xtb_file_parameters, 
						xtb_file_best_parameters,
						xtb_working_directory, 
						atomic_energies_filename=atomic_energies_filename,
						dtype=dtype, 
						energy_weight=energy_weight,
						force_weight=force_weight,
						charge_weight=charge_weight,
						atomic_charge_weight=atomic_charge_weight,
						dipole_weight=dipole_weight,
						nhlambda=nhlambda,
						loss_type=loss_type,
						seed=seed)

		elif electrostatic_model=="Coulomb" or electrostatic_model=="Ewald/Coulomb" or  electrostatic_model is None or electrostatic_model=="None":
			self._physModel=PhysModelStandard (
						nn_model=nn_model,
						F=F,
						K=K,
						num_scc =  num_scc,
						em_type =  em_type,
						num_hidden_nodes_em =  num_hidden_nodes_em,
						num_hidden_layers_em = num_hidden_layers_em,
						sr_cut=sr_cut,
						dtype=dtype, 
						num_blocks=num_blocks, 
						num_residual_atomic=num_residual_atomic,
						num_residual_interaction=num_residual_interaction,
						num_residual_output=num_residual_output,
						num_outputs=2, # Ea and Qa
						activation_fn=activation_fn,
						scale_shift_output=scale_shift_output,
						energy_weight=energy_weight,
						force_weight=force_weight,
						charge_weight=charge_weight,
						atomic_charge_weight=atomic_charge_weight,
						dipole_weight=dipole_weight,
						drop_rate=drop_rate,
						use_scaled_charges=use_scaled_charges,
						electrostatic_model=electrostatic_model,
						use_dispersion=use_dispersion,
						dispersionParameters=dispersionParameters,
						nhlambda=nhlambda,
						loss_type=loss_type,
						basis_type=basis_type,
						lr_cut= lr_cut,
						Lmax=Lmax,
						beta=beta,
						num_interaction_layers=num_interaction_layers,
						num_output_layers=num_output_layers,
						num_interaction_nodes=num_interaction_nodes,
						num_output_nodes=num_output_nodes,
						repulsion_model=repulsion_model,
						repulsion_fit_parameters=repulsion_fit_parameters,
						type_output=type_output,
						kmax=kmax,
						seed=seed)

		elif electrostatic_model=="EEM":
			self._physModel=PhysModelEEM (
						nn_model=nn_model,
						F=F,
						K=K,
						num_scc =  num_scc,
						em_type =  em_type,
						num_hidden_nodes_em =  num_hidden_nodes_em,
						num_hidden_layers_em = num_hidden_layers_em,
						sr_cut=sr_cut,
						dtype=dtype, 
						num_blocks=num_blocks, 
						num_residual_atomic=num_residual_atomic,
						num_residual_interaction=num_residual_interaction,
						num_residual_output=num_residual_output,
						num_outputs=2, # Ea and Xi
						activation_fn=activation_fn,
						scale_shift_output=scale_shift_output,
						energy_weight=energy_weight,
						force_weight=force_weight,
						charge_weight=charge_weight,
						atomic_charge_weight=atomic_charge_weight,
						dipole_weight=dipole_weight,
						drop_rate=drop_rate,
						use_electrostatic=True,
						use_dispersion=use_dispersion,
						dispersionParameters=dispersionParameters,
						eem_fit_parameters=eem_fit_parameters,
						nhlambda=nhlambda,
						loss_type=loss_type,
						basis_type=basis_type,
						lr_cut= lr_cut,
						Lmax=Lmax,
						beta=beta,
						num_interaction_layers=num_interaction_layers,
						num_output_layers=num_output_layers,
						num_interaction_nodes=num_interaction_nodes,
						num_output_nodes=num_output_nodes,
						repulsion_model=repulsion_model,
						repulsion_fit_parameters=repulsion_fit_parameters,
						type_output=type_output,
						seed=seed)

		elif electrostatic_model=="EEMPot":
			self._physModel=PhysModelEEMPot (
						nn_model=nn_model,
						F=F,
						K=K,
						num_scc =  num_scc,
						em_type =  em_type,
						num_hidden_nodes_em =  num_hidden_nodes_em,
						num_hidden_layers_em = num_hidden_layers_em,
						sr_cut=sr_cut,
						dtype=dtype, 
						num_blocks=num_blocks, 
						num_residual_atomic=num_residual_atomic,
						num_residual_interaction=num_residual_interaction,
						num_residual_output=num_residual_output,
						num_outputs=3, # Ea ,Xi and Qa
						activation_fn=activation_fn,
						scale_shift_output=scale_shift_output,
						energy_weight=energy_weight,
						force_weight=force_weight,
						charge_weight=charge_weight,
						atomic_charge_weight=atomic_charge_weight,
						dipole_weight=dipole_weight,
						drop_rate=drop_rate,
						use_scaled_charges=use_scaled_charges,
						use_electrostatic=True,
						use_dispersion=use_dispersion,
						dispersionParameters=dispersionParameters,
						eem_fit_parameters=eem_fit_parameters,
						nhlambda=nhlambda,
						loss_type=loss_type,
						basis_type=basis_type,
						lr_cut= lr_cut,
						Lmax=Lmax,
						beta=beta,
						num_interaction_layers=num_interaction_layers,
						num_output_layers=num_output_layers,
						num_interaction_nodes=num_interaction_nodes,
						num_output_nodes=num_output_nodes,
						repulsion_model=repulsion_model,
						repulsion_fit_parameters=repulsion_fit_parameters,
						type_output=type_output,
						seed=seed)

		elif electrostatic_model=="GFN1" or electrostatic_model=="GFN":
			self._physModel=PhysModelGFN1 (
						nn_model=nn_model,
						F=F,
						K=K,
						num_scc =  num_scc,
						em_type =  em_type,
						num_hidden_nodes_em =  num_hidden_nodes_em,
						num_hidden_layers_em = num_hidden_layers_em,
						sr_cut=sr_cut,
						dtype=dtype, 
						num_blocks=num_blocks, 
						num_residual_atomic=num_residual_atomic,
						num_residual_interaction=num_residual_interaction,
						num_residual_output=num_residual_output,
						activation_fn=activation_fn,
						energy_weight=energy_weight,
						force_weight=force_weight,
						charge_weight=charge_weight,
						atomic_charge_weight=atomic_charge_weight,
						dipole_weight=dipole_weight,
						drop_rate=drop_rate,
						use_scaled_charges=use_scaled_charges,
						use_electrostatic=True,
						scale_shift_output=scale_shift_output,
						use_dispersion=use_dispersion,
						dispersionParameters=dispersionParameters,
						gfn_fit_parameters=gfn_fit_parameters,
						nhlambda=nhlambda,
						loss_type=loss_type,
						orbfile=orbfile,
						basis_type=basis_type,
						lr_cut= lr_cut,
						Lmax=Lmax,
						beta=beta,
						num_interaction_layers=num_interaction_layers,
						num_output_layers=num_output_layers,
						num_interaction_nodes=num_interaction_nodes,
						num_output_nodes=num_output_nodes,
						repulsion_model=repulsion_model,
						repulsion_fit_parameters=repulsion_fit_parameters,
						type_output=type_output,
						seed=seed)
		else:
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
			print("Fatal error, Unknown model", file=sys.stderr)
			print("Check electrostatic_model &nd nn_model parameters", file=sys.stderr)
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
			exit(1)


	def computeMyPearsonCorrelation(self, data, energies=None, charges=None, Qa = None, dipoles=None, forces=None):
		r = {}
		coefs = { 'E':self.energy_weight, 'F':self.force_weight, 'Q':self.charge_weight, 
			  'Qa':self.atomic_charge_weight, 'D':self.dipole_weight
			}
		values = { 'E':energies, 'F':forces, 'Q':charges, 'Qa':Qa, 'D':dipoles}
		for key in coefs:
			if coefs[key] > 0:
				if values[key] is None:
					energies, charges, Qa, dipoles, forces, nhloss = self.computeProperties(data)
					values = { 'E':energies, 'F':forces, 'Q':charges, 'Qa':Qa, 'D':dipoles}
				r[key] = pearson_correlation_coefficient(tf.reshape(tf.constant(data[key],dtype=self.dtype),[-1]), tf.reshape(values[key],[-1]))

		return r

	def computeMyPearsonCorrelationSquared(self, data, energies=None, charges=None, Qa=None, dipoles=None, forces=None):
		r2 = {}
		coefs = { 'E':self.energy_weight, 'F':self.force_weight, 'Q':self.charge_weight, 
			  'Qa':self.atomic_charge_weight, 'D':self.dipole_weight
			}
		values = { 'E':energies, 'F':forces, 'Q':charges, 'Qa':Qa, 'D':dipoles}
		for key in coefs:
			if coefs[key] > 0:
				if values[key] is None:
					energies, charges, Qa, dipoles, forces, nhloss = self.computeProperties(data)
					values = { 'E':energies, 'F':forces, 'Q':charges, 'Qa':Qa, 'D':dipoles}
				r2[key] = pearson_correlation_coefficient_squared(tf.reshape(tf.constant(data[key],dtype=self.dtype),[-1]), tf.reshape(values[key],[-1]))
		return r2


	def computePearsonCorrelation(self, data, energies=None, charges=None, Qa = None, dipoles=None, forces=None):
		r = {}
		coefs = { 'E':self.energy_weight, 'F':self.force_weight, 'Q':self.charge_weight, 
			  'Qa':self.atomic_charge_weight, 'D':self.dipole_weight
			}
		values = { 'E':energies, 'F':forces, 'Q':charges, 'Qa':Qa, 'D':dipoles}
		for key in coefs:
			if coefs[key] > 0:
				if values[key] is None:
					energies, charges, Qa, dipoles, forces, nhloss = self.computeProperties(data)
					values = { 'E':energies, 'F':forces, 'Q':charges, 'Qa':Qa, 'D':dipoles}
				r[key] = tfp.stats.correlation(tf.reshape(tf.constant(data[key],dtype=self.dtype),[-1]), tf.reshape(values[keys],[-1]), sample_axis=0, event_axis=None)

		return r



	def computecoefficient_of_determination(self, data, energies=None, charges=None, Qa=None, dipoles=None, forces=None):
		R2 = {}
		coefs = { 'E':self.energy_weight, 'F':self.force_weight, 'Q':self.charge_weight, 
			  'Qa':self.atomic_charge_weight, 'D':self.dipole_weight
			}
		values = { 'E':energies, 'F':forces, 'Q':charges, 'Qa':Qa, 'D':dipoles}
		for key in coefs:
			if coefs[key] > 0:
				if values[key] is None:
					energies, charges, Qa, dipoles, forces, nhloss = self.computeProperties(data)
					values = { 'E':energies, 'F':forces, 'Q':charges, 'Qa':Qa, 'D':dipoles}
				R2[key] = coefficient_of_determination(tf.reshape(tf.constant(data[key],dtype=self.dtype),[-1]), tf.reshape(values[key],[-1]))

		return R2

	def computeAverages(self, values):
		ase = tf.reduce_mean(values)
		mae = tf.reduce_mean(tf.abs(values))
		rmse = tf.math.sqrt(tf.reduce_mean(values*values))
		return ase, mae, rmse

	def computeDeviations(self, data, energies=None, charges=None, Qa=None, dipoles=None, forces=None):
		
		mae ={}
		ase ={}
		rmse ={}
		coefs = { 'E':self.energy_weight, 'F':self.force_weight, 'Q':self.charge_weight, 
			  'Qa':self.atomic_charge_weight, 'D':self.dipole_weight
			}
		values = { 'E':energies, 'F':forces, 'Q':charges, 'Qa':Qa, 'D':dipoles}
		for key in coefs:
			if coefs[key] > 0:
				if values[key] is None:
					energies, charges, Qa, dipoles, forces, nhloss = self.computeProperties(data)
					values = { 'E':energies, 'F':forces, 'Q':charges, 'Qa':Qa, 'D':dipoles}
				dif = values[key]-tf.constant(data[key],dtype=self.dtype)
				ase[key], mae[key], rmse[key] = self.computeAverages(dif)

		return ase, mae, rmse

	def computeAccuracies(self, data):
		if str(self.physModel) == "PhysModelXTB" :
			self.physModel.sup_variables.set_xtb_variables_from_trainables()
		energies, charges, Qa, dipoles, forces, nhloss = self.physModel.computeProperties(data)
		ase, mae, rmse = self.computeDeviations(data, energies=energies, charges=charges, Qa=Qa, dipoles=dipoles, forces=forces)
		coefs = { 'E':self.energy_weight, 'F':self.force_weight, 'Q':self.charge_weight, 
			  'Qa':self.atomic_charge_weight, 'D':self.dipole_weight
			}
		"""
		loss = 0
		for key in coefs:
			loss += coefs[key]*mae[key]

		"""
		loss=0.0
		for key in coefs:
			if coefs[key]>0:
				if self.loss_type==0:
					v = mae[key]
				else:
					v = rmse[key]
				loss += v*coefs[key]
		if self.nhlambda>0:
			loss += self.nhlambda*nhloss


		lossDic = {}
		lossDic['L'] = loss

		R2 = self.computecoefficient_of_determination(data, energies=energies, charges=charges, Qa=Qa, dipoles=dipoles, forces=forces)
		r2 = self.computeMyPearsonCorrelationSquared(data, energies=energies, charges=charges, Qa=Qa, dipoles=dipoles, forces=forces)
		acc = {}
		acc["Loss"] = lossDic
		acc["mae"] = mae
		acc["ase"] = ase
		acc["rmse"] =rmse
		acc['R2'] = R2
		acc['r2'] = r2

		return acc

	def getLoss(self, weight, values):
		if self.loss_type==0:
			return  weight*tf.reduce_mean(values)
		else:
			return weight*tf.math.sqrt(tf.reduce_mean(values*values))

	def computeAccuracy(self, data):
		if str(self.physModel) == "PhysModelXTB" :
			self.physModel.sup_variables.set_xtb_variables_from_trainables()
		energies, charges, Qa, dipoles, forces, nhloss = self.physModel.computeProperties(data)
		loss = 0
		if self.energy_weight > 0:
			de = tf.abs(energies-tf.constant(data['E'],dtype=self.dtype))
			#loss += self.energy_weight*tf.reduce_mean(de)
			loss += self.getLoss(self.energy_weight,de)
		if self.charge_weight > 0:
			dq = tf.abs(charges-tf.constant(data['Q'],dtype=self.dtype))
			#loss += self.charge_weight*tf.reduce_mean(dq)
			loss += self.getLoss(self.charge_weight,dq)
		if self.atomic_charge_weight > 0:
			dqa = tf.abs(Qa-tf.constant(data['Qa'],dtype=self.dtype))
			#loss += self.atomic_charge_weight*tf.reduce_mean(dqa)
			loss += self.getLoss(self.atomic_charge_weight,dqa)
		if self.dipole_weight > 0:
			dd = tf.abs(dipoles-data['D'])
			dd = tf.abs(dipoles-tf.constant(data['D'],dtype=self.dtype))
			#loss +=self.dipole_weight*tf.reduce_mean(dd)
			loss += self.getLoss(self.dipole_weight,dd)
		if self.force_weight > 0:
			df = tf.abs(forces-tf.constant(data['F'],dtype=self.dtype))
			#loss += self.force_weight*tf.reduce_mean(df)
			loss += self.getLoss(self.force_weight,df)

		if self.nhlambda>0:
			loss += self.nhlambda*nhloss
		means = {}
		if self.energy_weight > 0:
			means['E'] = tf.reduce_mean(de)
		if self.charge_weight > 0:
			means['Q'] = tf.reduce_mean(dq)
		if self.atomic_charge_weight > 0:
			means['Qa'] = tf.reduce_mean(dqa)
		if self.dipole_weight > 0:
			means['D'] = tf.reduce_mean(dd)
		if self.force_weight > 0:
			means['F'] = tf.reduce_mean(df)

		"""
		r = self.computePearsonCorrelation(data)
		R2 = self.computecoefficient_of_determination(data)
		mr = self.computeMyPearsonCorrelation(data)
		r2 = self.computeMyPearsonCorrelationSquared(data)
		r = self.computePearsonCorrelation(data, energies, charges, Qa, dipoles, forces)
		R2 = self.computecoefficient_of_determination(data, energies, charges, Qa, dipoles, forces)
		mr = self.computeMyPearsonCorrelation(data, energies, charges, Qa, dipoles, forces)
		r2 = self.computeMyPearsonCorrelationSquared(data, energies, charges, Qa, dipoles, forces)
		acc = self.computeAccuracies(data)
		for keys in acc:
			print(keys,":")
			[ print("\t",keys,"[", key,"]=", acc[keys][key].numpy()) for key in acc[keys] ]
		"""

		return means,loss

	#calculates the electrostatic energy per atom 
	#for very small distances, the 1/r law is shielded to avoid singularities
	def electrostatic_energy_per_atom(self, Z, Dij, Qa, idx_i, idx_j):
		return self._physModel.electrostatic_energy_per_atom(Z, Dij, Qa, idx_i, idx_j)

	def computeProperties(self, data):
		return self._physModel.computeProperties(data)

	def computeAtomicEnergiesAndForces(self, data):
		return self._physModel.computeAtomicEnergiesAndForces(data)

	def computeHessian(self, data):
		return self._physModel.computeHessian(data)

	def computedDipole(self, data):
		return self._physModel.computedDipole(data)

	def computeHessianAnddDipoles(self, data):
		return self._physModel.computeHessianAnddDipoles(data)

	def computeLoss(self, data):
		return self._physModel.computeLoss(data)

	def print_parameters(self):
		self._physModel.print_parameters()

	def __call__(self, data, closs=True):
		return self._physModel(data,closs=closs)

	@property
	def physModel(self):
		return self._physModel

	@property
	def dtype(self):
		return self._physModel.dtype

	@property
	def neuralNetwork(self):
		return self._physModel.neuralNetwork

	@property
	def energy_weight(self):
		return self._physModel.energy_weight

	@property
	def force_weight(self):
		return self._physModel.force_weight

	@property
	def charge_weight(self):
		return self._physModel.charge_weight

	@property
	def atomic_charge_weight(self):
		return self._physModel.atomic_charge_weight

	@property
	def dipole_weight(self):
		return self._physModel.dipole_weight

	@property
	def use_scaled_charges(self):
		return self._physModel.use_scaled_charges

	@property
	def use_electrostatic(self):
		return self._physModel.use_electrostatic

	@property
	def use_dispersion(self):
		return self._physModel.use_dispersion
	@property
	def nhlambda(self):
		return self._physModel.nhlambda

	@property
	def lr_cut(self):
		return self._physModel.lr_cut

	@property
	def loss_type(self):
		return self._physModel.loss_type

	@property
	def sr_cut(self):
		return self._physModel.sr_cut

	@property
	def repulsion_model(self):
		return self._physModel.repulsion_model

