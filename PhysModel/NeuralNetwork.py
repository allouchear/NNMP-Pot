from __future__ import absolute_import
import os
import tensorflow as tf
from MessagePassingNN.MessagePassingNeuralNetwork import *
from ElementalModesMessagePassingNN.ElementalModesMessagePassingNeuralNetwork import *
from EmbeddedAtomNN.EmbeddedAtomNeuralNetwork import *
from EmbeddedAtomNN.EmbeddedAtomPairsNeuralNetwork import *
from EmbeddedAtomMessagePassingNN.EmbeddedAtomMessagePassingNeuralNetwork import *
from Utils.ActivationFunctions import *

def neuralNetwork(
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
		num_outputs=2,        		  #number of outputs by atom
                shifts=None,                      #initial value for output shift (makes convergence faster)
                scales=None,                      #initial value for output scale (makes convergence faster)
                drop_rate=None,                   #initial value for drop rate (None=No drop)
		activation_fn=shifted_softplus,   #activation function
		dtype=tf.float32,                 #single or double precision
		energy_weight=1.0,   		  # force contribution to the loss function
		force_weight=1.0,   		  # force contribution to the loss function
		charge_weight=1.0, 		  #charge contribution to the loss function
		atomic_charge_weight=1.0, 	  # atomic charge contribution to the loss function
		dipole_weight=1.0, 		  # dipole contribution to the loss function
		use_scaled_charges=True,          # use scaled charges in electrostatic energy & dipole
		use_electrostatic=True,           # use electrostatics in energy prediction
		use_dispersion=True,		  # use dispersion in energy prediction 
		dispersionParameters=None,        # dispersion parameters. If None => trainable
		nhlambda=0,			  # lambda multiplier for non-hierarchicality loss (regularization)
		basis_type="Default",           # radial basis type : GaussianNet (Default for MPNN), Gaussian(Default for EANN), Bessel, Slater, 
		nn_model="MPNN",		# MPNN (Message-Passing Neural network), EANN (Embedded Atom Neural Network), EAMP (Embedded Atom Message-Passing Neural network), EANNP (Embedded Atom Pairs Neural Network)
		# Specific parameters for EANN
		Lmax=2,                           	# Lmax (same value for all atoms)
		beta=0.2,                       	# beta value for basis functions
		num_interaction_layers=1,     		# number of hidden layers for interaction block (1 at least)
		num_output_layers=1,          		# number of hidden layers for the output blocks (can be =0 !)
		num_interaction_nodes=10,     		# number of nodes in each hidden layer of interaction block (1 at least)
		num_output_nodes=10,          		# number of nodes in each hidden layer in the output block 
		type_output=0,                  # 0=> average, 1=> Wx+b (modules outputs)
		seed=None):

	neuralNetwork = None
	if nn_model=="MPNN":
		neuralNetwork = MessagePassingNeuralNetwork(F,K,sr_cut, num_blocks=num_blocks, num_residual_atomic=num_residual_atomic, 
		num_residual_interaction=num_residual_interaction,
		num_residual_output=num_residual_output, num_outputs=num_outputs,
		shifts=shifts, scales=scales,
		drop_rate=drop_rate,
		beta=beta,
		basis_type=basis_type,
		type_output=type_output,
		activation_fn=activation_fn,dtype=dtype,seed=seed) 
	elif nn_model=="EMMPNN":
		neuralNetwork = ElementalModesMessagePassingNeuralNetwork(
		F,
		K,sr_cut, 
		num_scc =  num_scc,
		em_type = em_type,
		num_hidden_nodes_em=num_hidden_nodes_em,
		num_hidden_layers_em=num_hidden_layers_em,
		num_blocks=num_blocks, num_residual_atomic=num_residual_atomic, 
		num_residual_interaction=num_residual_interaction,
		num_residual_output=num_residual_output, num_outputs=num_outputs,
		shifts=shifts, scales=scales,
		drop_rate=drop_rate,
		beta=beta,
		basis_type=basis_type,
		type_output=type_output,
		activation_fn=activation_fn,dtype=dtype,seed=seed) 
	elif nn_model=="EANN":
		neuralNetwork = EmbeddedAtomNeuralNetwork(K,Lmax, sr_cut, beta=beta, num_blocks=num_blocks, 
		num_interaction_layers=num_interaction_layers,
		num_output_layers=num_output_layers,
		num_interaction_nodes=num_interaction_nodes,
		num_output_nodes=num_output_nodes,
		num_outputs=num_outputs,
		shifts=shifts, 
		scales=scales,
		drop_rate=drop_rate,
		basis_type=basis_type,
		type_output=type_output,
		activation_fn=activation_fn,dtype=dtype,seed=seed) 
	elif nn_model=="EANNP":
		neuralNetwork = EmbeddedAtomPairsNeuralNetwork(K,Lmax, sr_cut, beta=beta, num_blocks=num_blocks, 
		num_interaction_layers=num_interaction_layers,
		num_output_layers=num_output_layers,
		num_interaction_nodes=num_interaction_nodes,
		num_output_nodes=num_output_nodes,
		num_outputs=num_outputs,
		shifts=shifts, 
		scales=scales,
		drop_rate=drop_rate,
		basis_type=basis_type,
		type_output=type_output,
		activation_fn=activation_fn,dtype=dtype,seed=seed) 
	else:
		neuralNetwork = EmbeddedAtomMessagePassingNeuralNetwork(F,K,sr_cut, num_blocks=num_blocks, num_residual_atomic=num_residual_atomic, 
		num_residual_interaction=num_residual_interaction,
		num_residual_output=num_residual_output, num_outputs=num_outputs,
		shifts=shifts, scales=scales,
		drop_rate=drop_rate,
		beta=beta,
		basis_type=basis_type,
		type_output=type_output,
		activation_fn=activation_fn,dtype=dtype,seed=seed) 
	return neuralNetwork
