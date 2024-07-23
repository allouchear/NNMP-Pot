#!/home/theochem/allouche/Softwares/anaconda3/bin/python -u
from ase import io
from ase.units import Bohr,Hartree
# Electron volts (eV), Ångström (Ang), the atomic mass unit and Kelvin are defined as 1.0.
from ase.vibrations import Infrared

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(40)
tf.config.threading.set_inter_op_parallelism_threads(40)

import numpy as np
from Utils.Predictor import *
import os
import shutil

def remove_directory(dir_path):
	try:
		shutil.rmtree(dir_path)
	except OSError as e:
		print("Error: %s : %s" % (dir_path, e.strerror))

remove_directory("vib")
remove_directory("ir")

atoms = io.read('molecule.xyz')

print("unitDeb=", ase.units.Debye)
print("1/unitDeb=", 1.0/ase.units.Debye)
print("1/unitDeb*Bohr=", 1.0/ase.units.Debye*Bohr)

atoms.calc = Predictor(
		[
"trainingNetNatChemQE1.txt",
#"trainingNetNatChemQE2.txt",
#"trainingNetNatChemQE3.txt",
#"trainingNetNatChemQE4.txt",
#"trainingNetNatChemQE5.txt"
		],
		atoms,
		conv_distance=1/Bohr,
		conv_energy=1/Hartree
		)

from ase.optimize import BFGS

BFGS(atoms).run(fmax=0.001)

print("Position=",atoms.get_positions())
print("Dipole=",atoms.get_dipole_moment())


ir = Infrared(atoms)

ir.run()

ir.summary(intensity_unit='km/mol')
ir.clean()
