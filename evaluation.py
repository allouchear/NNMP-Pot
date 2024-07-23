#!/home/theochem/allouche/Softwares/anaconda3/bin/python -u
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(40)
tf.config.threading.set_inter_op_parallelism_threads(40)

import numpy as np
from Utils.Evaluator import *
#from Utils.UtilsTrain import *

evaluator = Evaluator(
		[
			#"trainingXTB0NatChemQE_parameters.txt",
			"trainingXTB1NatChemQE_parameters.txt",
			"trainingXTB2NatChemQE_parameters.txt",
		], # list of models
		dataFile=os.path.join("DataBehlerNatureCom","input.npz"), 
		#nvalues=-1,
		nvalues=100,
		batch_size=100,
		convDistanceToBohr=1.8897261278504418, # conv data to unit of NNMol
		convEnergyToHartree=0.03674932427703664, 
		convDipoleToAU=1.8897261278504418
		)

print("Accuraties for :", evaluator.nvalues, "values")
print("---------------------------------------------")
acc=evaluator.computeAccuracies(verbose=True)
#print_all_acc(acc)
metrics_dir="metricsEvaluation"
print("Save validation data metrics in :", metrics_dir)
fileNames=evaluator.saveAnalysis(metrics_dir)

