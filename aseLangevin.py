#!/home/theochem/allouche/Softwares/anaconda3/bin/python -u
from ase import io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
# Electron volts (eV), Ångström (Ang), the atomic mass unit and Kelvin are defined as 1.0.
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(40)
tf.config.threading.set_inter_op_parallelism_threads(40)

import numpy as np
from Utils.Predictor import *
from PhysModel.PhysModelNet import *
import os

atoms = io.read('molecule.xyz')


atoms.calc = Predictor(
		[
			"trainingNetNatChemQE.txt",
			"trainingNetNatChemQE.txt",
		],
		atoms,
		conv_distance=1/units.Bohr,
		conv_energy=1/units.Hartree
		)

from ase.optimize import BFGS
from ase.vibrations import Vibrations

BFGS(atoms).run(fmax=0.01)

print(atoms.get_positions())



T = 300
# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 5 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(atoms, 1 * units.fs, T * units.kB, 0.002)


def printenergy(a=atoms):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))



dyn.attach(printenergy, interval=50)

# We also want to save the positions of all atoms after every 50th time step.
traj = Trajectory('moldyn3.traj', 'w', atoms)
dyn.attach(traj.write, interval=50)

# Now run the dynamics
printenergy()
dyn.run(5000)

