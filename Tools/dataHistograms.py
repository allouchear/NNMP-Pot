#!/home/theochem/allouche/Softwares/anaconda3/bin/python -u
import tensorflow as tf

import numpy as np
from GrimmeD3.GrimmeD3 import *
from Utils.DataContainer import *
from Utils.DataProvider import *
from PhysModel.PhysModel import *
from PhysModel.PhysModelEEM import *
from PhysModel.PhysModelEEMPot import *
from Utils.UtilsFunctions import *
from Utils.Trainer import *
from PhysModel.PhysModelNet import *

args = getArguments()

directory, log_dir, best_dir, average_dir,  metrics_dir, best_checkpoint, average_checkpoint, step_checkpoint, best_loss_file, best_xtbfile =setOutputLocationFiles(args)

dispersionParameters = getDispersionParameters(args)

logging.info("Creation of physical model")
physModel=   PhysModel (F=args.num_features,
			K=args.num_basis,
			sr_cut=args.cutoff,
			lr_cut=args.lr_cutoff,
			dtype=tf.float64 if args.dtype=='float64' else tf.float32, 
			num_blocks=args.num_blocks, 
			num_residual_atomic=args.num_residual_atomic,
			num_residual_interaction=args.num_residual_interaction,
			num_residual_output=args.num_residual_output,
			activation_fn=activation_deserialize(args.activation_function),
			energy_weight=args.energy_weight,
			force_weight=args.force_weight,
			charge_weight=args.charge_weight,
			atomic_charge_weight=args.atomic_charge_weight,
			dipole_weight=args.dipole_weight,
			drop_rate=args.drop_rate,
			use_scaled_charges=(args.use_scaled_charges==1),
			use_dispersion=(args.use_dispersion==1),
			dispersionParameters=dispersionParameters,
			eem_fit_parameters=args.eem_fit_parameters,
			gfn_fit_parameters=args.gfn_fit_parameters,
			nhlambda=args.nhlambda,
			orbfile=args.orbfile,
			xtb_file_parameters=args.xtb_file_parameters, 
			xtb_file_best_parameters=best_xtbfile,
			xtb_working_directory=directory, 
			atomic_energies_filename=args.atomic_energies,
			nn_model=args.nn_model,
			basis_type=args.basis_type,
			electrostatic_model=args.electrostatic_model,
			Lmax=args.Lmax,
			beta=args.beta,
			num_interaction_layers=args.num_interaction_layers,
			num_output_layers=args.num_output_layers,
			num_interaction_nodes=args.num_interaction_nodes,
			num_output_nodes=args.num_output_nodes,
			seed=args.seed)


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=args.ema_decay, epsilon=1e-10,amsgrad=False)

print("Creation of trainer")
trainer=Trainer(physModel, optimizer, dataFile=args.dataset, 
		ntrain=args.num_train, nvalid=args.num_valid, batch_size=args.batch_size, valid_batch_size=args.valid_batch_size, seed=1,
		convDistanceToBohr=args.conv_distance_to_bohr, convEnergyToHartree=args.conv_energy_to_hartree, convDipoleToAU=args.conv_dipole_to_au)

trainer.dataProvider.create_tensorboard_histograms(log_dir)

