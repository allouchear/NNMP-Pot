from __future__ import absolute_import
import os
import sys
import math
import tensorflow as tf
from MessagePassingNN.MessagePassingNeuralNetwork import *
from EmbeddedAtomNN.EmbeddedAtomNeuralNetwork import *
from Utils.ActivationFunctions import *
from Utils.UtilsFunctions import *
from Utils.PeriodicTable import *
from Dispersion.Dispersion import *
from Electrostatic.Electrostatic import *
from NuclearRepulsion.NuclearRepulsion import *
from .NeuralNetwork import *


from tensorflow.keras.layers import Layer

class PhysModelEEMPot(tf.keras.Model):
	def __str__(self):
		st = str(self.neuralNetwork)
		if self.neuralNetwork.activation_fn is None:
			st += "\nNo activation function"
		else:
			st += "\n"+str(self.neuralNetwork.activation_fn.__name__) +" activation function"
		st += "\nEEM Electrostatic energy, NN charges"
		if self.repulsion_model is None:
			st += "\nNo repulsion pot."
		else:
			st += "\n"+str(self.repulsion_model)

		return st

	def __init__(self,
		F=128,                            #dimensionality of feature vector
		K=64,                             #number of radial basis functions
		sr_cut=8.0,                       #cutoff distance for short range interactions (atomic unit)
		lr_cut = None,                    #cutoff distance for long range interactions (default: no cutoff)
		num_scc=0,                        #number of cycles of updating of spin atomic charges (0 => no cycle)
		em_type = 0,       		  #  Elemental modes type : 0(default) => only Z, 1=> Z + Masses, 2=> Z+Masses+QaAlpha+QaBeta (so we use charge &multiplicity)
		num_hidden_nodes_em = None,       # number of nodes on each hidden layer in elemental modes block
		num_hidden_layers_em = 2,         #  number of hidden layers in elemental modes block
		num_blocks=5,                     #number of building blocks to be stacked
		num_residual_atomic=2,            #number of residual layers for atomic refinements of feature vector
		num_residual_interaction=3,       #number of residual layers for refinement of message vector
		num_residual_output=1,            #number of residual layers for the output blocks
		num_outputs=3,        		  #number of outputs by atom
		scale_shift_output=False,         # scale & shift output
                drop_rate=None,                   #initial value for drop rate (None=No drop)
		activation_fn=shifted_softplus,   #activation function
		dtype=tf.float32,                 #single or double precision
		energy_weight=1.0,   		  # force contribution to the loss function
		force_weight=100.0,   		  # force contribution to the loss function
		charge_weight=1.0, 		  # charge contribution to the loss function
		atomic_charge_weight=0.0, 	  # atomic charge contribution to the loss function
		dipole_weight=1.0, 		  # dipole contribution to the loss function
		use_scaled_charges=True,          # use scaled charges for energy
		use_electrostatic=True,           # use electrostatics in energy prediction
		use_dispersion=True,		  # use dispersion in energy prediction 
		dispersionParameters=None,        # dispersion parameters. If None => trainable
		nhlambda=0,			  # lambda multiplier for non-hierarchicality loss (regularization)
		loss_type=0,			  # loss type (0=> Mean, 1=>RMSE)
		basis_type="Default",           # radial basis type : GaussianNet (Default for MPNN), Gaussian(Default for EANN), Bessel, Slater, 
		eem_fit_parameters=0,		  # fit eem parameters (0 no, 1 fit J, 2 fit alp, 3 fit J & alp
		nn_model="MPNN",			# MPNN (Message-Passing Neural network), EANN (Embedded Atom Neural Network), EAMP (Embedded Atom Message-Passing Neural network), EANNP (Embedded Atom Pairs Neural Network)
		# Specific parameters for EANN
		Lmax=2,                           	# Lmax (same value for all atoms)
		beta=0.2,                       	# beta value for basis functions
		num_interaction_layers=1,     		# number of hidden layers for interaction block (1 at least)
		num_output_layers=1,          		# number of hidden layers for the output blocks (can be =0 !)
		num_interaction_nodes=10,     		# number of nodes in each hidden layer of interaction block (1 at least)
		num_output_nodes=10,          		# number of nodes in each hidden layer in the output block 
		repulsion_model = None,           # None, ZBL or GFN1
		repulsion_fit_parameters=1,       # 0, 1 , 2   or 3
		type_output=0,                  # 0=> average, 1=> Wx+b (modules outputs)
		seed=None):
		super().__init__(dtype=dtype, name="PhysModelEEMPot")

		if scale_shift_output is True:
			shifts=[0.0]*num_outputs
			scales=[1.0]*num_outputs
		else:
			shifts=None
			scales=None

		self._lr_cut = lr_cut

		self._neuralNetwork = neuralNetwork(
			nn_model=nn_model,
			F=F,
			K=K,
			basis_type=basis_type,
			sr_cut=sr_cut, 
			num_scc =  num_scc,
			em_type = em_type,
			num_hidden_nodes_em = num_hidden_nodes_em,
			num_hidden_layers_em = num_hidden_layers_em,
			num_blocks=num_blocks, 
			num_residual_atomic=num_residual_atomic, 
			num_residual_interaction=num_residual_interaction,
			num_residual_output=num_residual_output, 
			num_outputs=num_outputs,
			shifts=shifts, 
			scales=scales,
			drop_rate=drop_rate,
			activation_fn=activation_fn,
			Lmax=Lmax,
			beta=beta,
			num_interaction_layers=num_interaction_layers,
			num_output_layers=num_output_layers,
			num_interaction_nodes=num_interaction_nodes,
			num_output_nodes=num_output_nodes,
			type_output=type_output,
			dtype=dtype,
			seed=seed) 

		self._nhlambda=nhlambda
		self._loss_type=loss_type
		self._dtype=dtype
		self._trainable_weights=None
		self._energy_weight=energy_weight
		self._force_weight=force_weight
		self._charge_weight=charge_weight
		self._atomic_charge_weight=atomic_charge_weight
		self._dipole_weight=dipole_weight
		self._use_electrostatic=use_electrostatic
		self._use_scaled_charges=use_scaled_charges
		self._use_dispersion=use_dispersion

		self._electrostatic_model = None
		if ((self.use_electrostatic and (self.energy_weight > 0 or self.force_weight > 0))) or (self.charge_weight>0) or (self.atomic_charge_weight) or (self.dipole_weight):
			self._electrostatic_model = Electrostatic(model_type="EEMPot",  
				sr_cut=sr_cut, lr_cut=lr_cut, eem_fit_parameters=eem_fit_parameters,
				use_scaled_charges=use_scaled_charges, 	dtype=dtype)

		self._dispersion_model=None
		if self.use_dispersion and (self.energy_weight > 0 or self.force_weight > 0):
			self._dispersion_model=Dispersion(dP=dispersionParameters,dtype=dtype, cutoff=self.lr_cut)

		if repulsion_model !="None":
			model_type=0
			if repulsion_model=="GFN1":
				model_type=1
			self._repulsion_model = NuclearRepulsion(model_type=model_type, fit_parameters=repulsion_fit_parameters, dtype=dtype)
		else:
			self._repulsion_model = None



	def computeAtomicEnergiesAndForces(self, data):
		print("====================== Error ==============================");
		print("computeProperties not yet implemented in PhysModelEEMPot.py");
		print("====================== Error ==============================");
		sys.exit()
		return None, None, None, None

	def computeProperties(self, data):
		M=data['M']
		QaAlpha=data['QaAlpha']
		QaBeta=data['QaBeta']

		#print(data)
		Z=data['Z']
		R=tf.Variable(data['R'],dtype=self.dtype)
		idx_i=data['idx_i']
		idx_j=data['idx_j']
		offsets=data['offsets']
		sr_idx_i=data['sr_idx_i']
		sr_idx_j=data['sr_idx_j']
		sr_offsets=data['sr_offsets']
		#print("-------outputs------------------\n",outputs,"\n----------------------------\n")

		energies = None
		with tf.GradientTape() as g:
			g.watch(R)
			outputs, Dij , nhloss = self.neuralNetwork.atomic_properties(Z, R, idx_i, idx_j, M=M, QaAlpha=QaAlpha, QaBeta=QaBeta, offsets=offsets, sr_idx_i=sr_idx_i, sr_idx_j=sr_idx_j, sr_offsets=sr_offsets, batch_seg=data['batch_seg'])
			if (self.energy_weight > 0 or self.force_weight > 0):
				Ea = outputs[:,0]
			Xia = outputs[:,1]
			Qa  = outputs[:,2]
			charges = tf.squeeze(tf.math.segment_sum(Qa, data['batch_seg']))
			#print("Qtot=",data['Q'])
			#print("charges=",charges)
			if self.use_scaled_charges:
				Qa = scaled_charges(Z, Qa, Q_tot=data['Q'], batch_seg=data['batch_seg'])

			if self.use_electrostatic and (self.energy_weight > 0 or self.force_weight > 0):
				Elec,QaScaled = self._electrostatic_model.energy_per_atom(Z=Z, Dij=Dij, Xia=Xia, Qa=Qa, idx_i=idx_i, idx_j=idx_j, Q_tot=data['Q'], batch_seg=data['batch_seg'])
				if QaScaled is not None:
					Qa = QaScaled
				Ea += Elec
				#print("Ea apres electrstatic energy=",Ea)

			if self.use_dispersion and (self.energy_weight > 0 or self.force_weight > 0):
				Ea += self._dispersion_model.energy(Z, Dij, idx_i, idx_j)

			if self.repulsion_model is not None:
				Ea += self._repulsion_model.energy_per_atom(Z, Dij, idx_i, idx_j)

			if (self.energy_weight > 0 or self.force_weight > 0):
				energies = tf.squeeze(tf.math.segment_sum(Ea, data['batch_seg']))
				energy=tf.reduce_sum(energies)
			#print("Ea=",Ea)
			#print("energies=",energies)
			#ener=energies[0]
		#print("-------Energies-----------------\n",energies,"\n----------------------------\n")
		dipoles=None
		if self.dipole_weight > 0:
			QR = tf.stack([Qa*R[:,0], Qa*R[:,1], Qa*R[:,2]],1)
			dipoles = tf.math.segment_sum(QR, data['batch_seg'])
		if self.force_weight > 0:
			#forces = -convert_indexed_slices_to_tensor(g.gradient(energies,R))
			#forces = -convert_indexed_slices_to_tensor(g.gradient(energies,R))
			gradients=g.gradient(energy, R)
			#print("-------Gradients--------------\n",gradients,"\n----------------------------\n")
			forces = -tf.convert_to_tensor(gradients)
			#print("-------Forces-----------------\n",forces,"\n----------------------------\n")
		else:
			forces = None

		if self._trainable_weights is None:
			self._trainable_weights=self.neuralNetwork.trainable_weights

		return energies, charges, Qa, dipoles, forces, nhloss

	def getLoss(self, weight, values):
		if self.loss_type==0:
			return  weight*tf.reduce_mean(values)
		else:
			return weight*tf.math.sqrt(tf.reduce_mean(values*values))

	def computeLoss(self, data):
		with tf.GradientTape() as tape:
			energies, charges, Qa, dipoles, forces, nhloss = self.computeProperties(data)
			loss = 0
			"""
			if self.energy_weight > 0:
				de = tf.abs(energies-tf.constant(data['E'],dtype=self.dtype))
				loss +=  self.energy_weight*tf.reduce_mean(de)
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

		gradients = tape.gradient(loss, self.trainable_weights)
		#print("Loss=",loss)
		#print("-------Loss-----------------\n",loss,"\n----------------------------\n")
		#print("-------Gradients------------\n",gradients,"\n----------------------------\n")
		return energies, charges, Qa, dipoles, forces , loss, gradients

	def __call__(self, data, closs=True):
		
		if closs is not True:
			energies, charges, Qa, dipoles, forces, nhloss = self.computeProperties(data)
			loss=None
			gradients=None
		else:
			energies, charges, Qa, dipoles, forces , loss, gradients = self.computeLoss(data)
			
		if self._trainable_weights is None:
			self._trainable_weights=self.neuralNetwork.trainable_weights
		return energies, charges, Qa, dipoles, forces, loss, gradients

	def print_parameters(self):
		if self._dispersion_model is not None:
			self._dispersion_model.print_parameters()
		if self._electrostatic_model is not None:
			self._electrostatic_model.print_parameters()
		if self._repulsion_model is not None:
			self._repulsion_model.print_parameters()



	@property
	def dtype(self):
		return self._dtype

	@property
	def eem(self):
		return self._eem

	@property
	def neuralNetwork(self):
		return self._neuralNetwork

#	@property
#	def trainable_weights(self):
#		return self._trainable_weights

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
	def use_electrostatic(self):
		return self._use_electrostatic

	@property
	def use_dispersion(self):
		return self._use_dispersion

	@property
	def use_scaled_charges(self):
		return self._use_scaled_charges

	@property
	def nhlambda(self):
		return self._nhlambda

	@property
	def lr_cut(self):
		return self._lr_cut

	@property
	def sr_cut(self):
		return self.neuralNetwork.sr_cut

	@property
	def repulsion_model(self):
		return self._repulsion_model

	@property
	def loss_type(self):
		return self._loss_type


