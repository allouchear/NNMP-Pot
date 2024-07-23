#!/home/csanz/bin/anaconda3/bin/python -u
import tensorflow as tf
import sys
print("GPUs Available: ", (tf.config.list_physical_devices('GPU')))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#sys.exit()

import tensorflow_addons as tfa

sys.path.append("/home/csanz/bin/NNMol-Per/")

#tf.config.threading.set_intra_op_parallelism_threads(2)
#tf.config.threading.set_inter_op_parallelism_threads(2)

import numpy as np
from Utils.DataContainer import *
from Utils.DataProvider import *
from PhysModel.PhysModel import *
from Utils.UtilsFunctions import *
from Utils.Trainer import *
from Utils.UtilsTrain import *
from Utils.UtilsModel import *
from PhysModel.PhysModel import *
import os


args = getArguments()


directory, log_dir, best_dir, average_dir,  metrics_dir, best_checkpoint, average_checkpoint, step_checkpoint, best_loss_file, best_xtbfile =setOutputLocationFiles(args)


save_model_parameters(args)

print("activation_fn=",activation_deserialize(args.activation_function))
print("activation_fn=",activation_deserialize(args.activation_function).__name__)

logging.info("Creation of physical model")
physModel=   create_model(args, directory, best_xtbfile)
print("========================================= Model =================================================================",flush=True)
print(physModel)
print("=================================================================================================================",flush=True)

#********************* Creation of trainer model ***************
logging.info("Creation of trainer")
trainer = create_trainer(physModel, args)
print("=================================================================================================================",flush=True)

logging.info("Creation of histograms for data")
trainer.dataProvider.create_tensorboard_histograms(log_dir)
print("=================================================================================================================",flush=True)

logging.info("Load best recorded loss")
lossbest, props = load_best_recorded_loss(best_loss_file)
print("=================================================================================================================",flush=True)

logging.info("Trying to load the best check point")
print("Trying to load the best check point")
ok=trainer.load_weights(fname=best_checkpoint)
print("=================================================================================================================",flush=True)

#********************* Open metrics files ***************
validation_metrics_files, validation_metrics_file_names = open_validation_metrics_files(metrics_dir,  args)
train_metrics_files, train_metrics_file_names = open_train_metrics_files(metrics_dir,  args)
test_metrics_files, test_metrics_file_names = open_test_metrics_files(metrics_dir,  args)

nstepsByEpoch = trainer.dataProvider.get_nsteps_batch()
nsteps=args.max_steps

print("=================================================================================================================",flush=True)
#********************* Training ***************
logging.info("Begin training")
print("Begin training")
sums=None
writer_logs_train = tf.summary.create_file_writer(os.path.join(log_dir,  'train'))
writer_logs_validation = tf.summary.create_file_writer(os.path.join(log_dir,  'validation'))
writer_logs_test = tf.summary.create_file_writer(os.path.join(log_dir,  'test'))
for istep in range(nsteps):
	epoch=istep//nstepsByEpoch
	st='Step : {:10d}/{:<10d} ; Epoch :{:10d} '.format(istep, nsteps,  epoch)
	print(st)
	nc=len(st); [ print("-",end='') if i<nc-1 else print("-",flush=True) for i in range(nc) ]
	#dt=trainer.dataProvider.next_batch()
	#print("dt.E=",dt['E'])
	#loss,gradients= trainer.applyOneStep(dt=dt, verbose=False)
	loss,gradients= trainer.applyOneStep(verbose=False)
	if args.verbose>=1:
		print_gradients_norms(gradients,physModel.trainable_weights,details=args.verbose>=2)
	#print_gradients_norms(gradients,physModel.trainable_weights,details=False)

	#sums=trainer.addTrainSums(sums=sums)
	#aloss=trainer.computeLossFromSums(sums)
	#print("Step : ", istep,"/",nsteps, "; Epoch = ", epoch, " ; Loss=",  loss.numpy(), " ; Averaged Loss=", aloss) 
	print("Loss=",  '{:14.8f}'.format(loss.numpy()))
	#if istep%args.validation_interval==1000:
	#if istep==0 or (istep+1)%args.validation_interval==0:
	if (istep+1)%args.validation_interval==0:
		lossbest=validation_test(trainer, lossbest, best_checkpoint, best_loss_file, istep)
		fileNames=trainer.saveAnalysis(metrics_dir,dataType=1,uid=(istep+1))
		acc=add_validation_metrics_to_files(trainer, validation_metrics_files, istep==0)
		add_metrics_to_logs(writer_logs_validation, acc, istep, prefix=None)

	#if istep==0 or (istep+1)%args.summary_interval==0:
	if (istep+1)%args.summary_interval==0:
		#fileNames=trainer.saveAnalysis(metrics_dir,dataType=1,uid=(istep+1))
		#acc=add_validation_metrics_to_files(trainer, validation_metrics_files, istep==0)
		#add_metrics_to_logs(writer_logs_validation, acc, istep, prefix=None)

		acc=add_train_metrics_to_files(trainer, train_metrics_files, istep==0)
		add_metrics_to_logs(writer_logs_train, acc, istep, prefix=None)

		acc=add_test_metrics_to_files(trainer, test_metrics_files, istep==0)
		add_metrics_to_logs(writer_logs_test, acc, istep, prefix=None)
	nc=90; [ print("=",end='') if i<nc-1 else print("=",flush=True) for i in range(nc) ]

#exit(1)
lossbest=validation_test(trainer, lossbest, best_checkpoint, best_loss_file, nsteps-1)
print("====================================================================================================",flush=True)
logging.info("Begin test on all training data & all validation ones")

trainer.save_weights(fname=step_checkpoint)
# Update the weights to their mean before saving
print("Averaged variables")
print("==================")
trainer.save_variable_backups()
if trainer.use_average==1:
        trainer.set_average_vars()
print("Train data")
print("----------")
#means,loss=trainer.computeTrainAccuracy(verbose=False)
#print_means(means)
#print("Loss=",loss.numpy())
acc=trainer.computeTrainAccuracies(verbose=False)
print_all_acc(acc)

print("Validation data")
print("--------------")
#means,loss=trainer.computeValidationAccuracy(verbose=False)
#print_means(means)
#print("Loss=",loss.numpy())
acc=trainer.computeValidationAccuracies(verbose=True)
print_all_acc(acc)
print("Specific parameters of model")
print("----------------------------")
physModel.print_parameters()
trainer.save_averaged_weights(fname=average_checkpoint)
print("----------------------------------------------------------")

print("Best variables")
#trainer.restore_variable_backups()
trainer.load_weights(fname=best_checkpoint)
print("==============")
print("Train data")
print("----------")
acc=trainer.computeTrainAccuracies(verbose=False)
print_all_acc(acc)
#means,loss=trainer.computeTrainAccuracy(verbose=False)
#print_means(means)
#print("Loss=",loss.numpy())
print("Save training data metrics in :", metrics_dir)
fileNames=trainer.saveAnalysis(metrics_dir,dataType=0,uid='final')

print("Validation data")
print("--------------")
#means,loss=trainer.computeValidationAccuracy(verbose=False)
#print_means(means)
#print("Loss=",loss.numpy())
acc=trainer.computeValidationAccuracies(verbose=False)
print_all_acc(acc)
print("Save validation data metrics in :", metrics_dir)
fileNames=trainer.saveAnalysis(metrics_dir,dataType=1, uid='final')

print("Test data")
print("--------------")
print("Save test data metrics in :", metrics_dir)
fileNames=trainer.saveAnalysis(metrics_dir,dataType=2, uid='final')

if args.verbose>=1:
	print("Specific parameters of model")
	print("----------------------------")
	physModel.print_parameters()

print("============================================================================")
logging.info("That'a all. Good bye")
print("That'a all. Good bye")
close_metrics_files(validation_metrics_files)
print("See metrics in files ")
print("---------------------")
for key in validation_metrics_file_names:
	print(validation_metrics_file_names[key])
for key in train_metrics_file_names:
	print(train_metrics_file_names[key])

print("Logs files for tendorboard ")
print("---------------------")
print("logs files are in ", log_dir)
print("To visualise them : tensorboard --logdir ", log_dir)
print(" and run your navigator to show all ")
print("============================================================================")
