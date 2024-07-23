#!/home/theochem/allouche/Softwares/anaconda3/bin/python -u
from ase import io
from ase.units import Bohr,Hartree
# Electron volts (eV), Ångström (Ang), the atomic mass unit and Kelvin are defined as 1.0.

import tensorflow as tf
#tf.config.threading.set_intra_op_parallelism_threads(40)
#tf.config.threading.set_inter_op_parallelism_threads(40)

import numpy as np
from Utils.Predictor import *
from PhysModel.PhysModelStandard import *
import os
import shutil

def remove_directory(dir_path):
	try:
		shutil.rmtree(dir_path)
	except OSError as e:
		print("Error: %s : %s" % (dir_path, e.strerror))

remove_directory("vib")
remove_directory("ir")


#atoms = io.read('molecule.xyz')
atoms = io.read('POSCAR')

atoms.calc = Predictor(
		[
		"Checkpoint",
		#"traingDirArgile24k6k5k2"
		],
		atoms,
		conv_distance=1/Bohr,
		conv_energy=1/Hartree
		)

e = atoms.get_potential_energy()
print(e)
f = atoms.get_forces()
print(f)
c = atoms.get_charges()
print(c)
cm = atoms.calc.get_molcharges(atoms)
print(cm)
d = atoms.calc.get_dipole(atoms)
print(d)

from ase.optimize import BFGS
from ase.vibrations import Vibrations

BFGS(atoms).run(fmax=0.001)

print(atoms.get_positions())

vib = Vibrations(atoms)
vib.run()
vib.summary()
