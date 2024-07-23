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

atoms = io.read('moleculeG.xyz')

from ase.calculators.gaussian import Gaussian

calc = Gaussian(label='calc/gaussian',
                xc='PBEPBE',
                basis='6-31G*',
                scf='maxcycle=100')

atoms.calc = calc

print("Position=",atoms.get_positions())
print("Dipole=",atoms.get_dipole_moment())


ir = Infrared(atoms)

ir.run()

ir.summary(intensity_unit='km/mol')
ir.clean()
