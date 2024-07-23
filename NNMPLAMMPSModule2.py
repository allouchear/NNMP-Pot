#!/home/theochem/allouche/Softwares/anaconda3/envs/tf/bin/python3
from ase import io
from ase.cell import Cell
from ase import Atoms
from ase import units
# Electron volts (eV), Angstrom (Ang), the atomic mass unit and Kelvin are defined as 1.0.

import tensorflow as tf
import sys
#tf.config.threading.set_intra_op_parallelism_threads(40)
#tf.config.threading.set_inter_op_parallelism_threads(40)

import numpy as np
from Utils.Predictor import *
from PhysModel.PhysModelStandard import *
import os
import sys

from timeit import default_timer as timer

def computeEnergyAndForces(lmodels, natoms, symbols, X,Y,Z, xBox, yBox, zBox, pbc, cutoff, idx_i, idx_j, offsetsX, offsetsY, offsetsZ):
	computeEnergyAndForces.calculator = getattr(computeEnergyAndForces, 'calculator', None)
	try:
		verbose=0
		if verbose>0:
			print("natoms =", natoms)
			start = timer()
		positions=[]
		for x,y,z in zip(X,Y,Z):
			positions.append((x,y,z))
		cell=[(xBox[0], xBox[1], xBox[2]), (yBox[0], yBox[1], yBox[2]), (zBox[0], zBox[1], zBox[2])]
		atoms = Atoms(symbols, positions=positions, cell=cell, pbc=pbc)
		offsets=[]
		for x,y,z in zip(offsetsX, offsetsY, offsetsZ):
			offsets.append([x,y,z])
		if verbose>0:
			startc = timer()
		
		idx_i=None
		idx_j=None
		offsets=None
		if computeEnergyAndForces.calculator is None:
			computeEnergyAndForces.calculator = Predictor(
				lmodels,
				atoms,
				conv_distance=1/units.Bohr,
				conv_energy=1/units.Hartree,
				verbose=verbose,
				idx_i=idx_i,
				idx_j=idx_j,
				offsets=offsets,
				)
		else:
			computeEnergyAndForces.calculator.set_idx(idx_i, idx_j, offsets)

		if verbose>0:
			endc = timer()
			print("build calculator time", endc-startc)
		energy, forces = computeEnergyAndForces.calculator.get_energy_forces(atoms)
		listRes = [energy[0]]
		for i in range (natoms):
			for k in range(3):
				listRes.append(forces[i][k])
		if verbose>0:
			end = timer()
			print("computeEnergyAndForces time", end-start)
			print("energy =", energy)
		return listRes
	except Exception as Ex:
		print("computeEnergyAndForces Failed.", Ex)
		raise Ex
		return None


#data=setData("test",2, ['O','H'], [0,0],[0,1],[0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,1], 5.0, [0,0], [0,1], [0,1], [1,0], [0,0])
#print(data)

def main(argv):
	if len(argv)<3:
		sys.exit(2) 
	calculator = None

if __name__ == "__main__":
    main(sys.argv[0:])
