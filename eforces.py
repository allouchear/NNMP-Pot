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

def readAtoms(filename):
	print("reading ", filename, " file ....",flush=True)
	atoms = io.read(filename)
	return atoms

def readBox(filename,atoms):
	print("reading ", filename, " file ....",flush=True)
	f = open(filename, "r")
	lines = f.readlines()
	v = []
	for i,line in enumerate(lines):
		data = line.split() 
		if len(data)!=3:
			print("fatal error", file=sys.stderr)
			print("first line of ",filename, "must contain 3 values by line", file=sys.stderr)
			sys.exit()
		if i==0:
			atoms.set_pbc((int(data[0])!=0, int(data[1])!=0, int(data[2])!=0))
		else:
			v.append((float(data[0]), float(data[1]),float(data[2])))

	atoms.set_cell([v[0], v[1], v[2]])
	f.close()
	return atoms

def readNeighbor(filename):
	print("reading neighbor  ", filename, " file ....", flush=True)
	f = open(filename, "r")
	lines = f.readlines()
	cutoff=float(lines[0])
	idx_i=[]
	idx_j=[]
	offsets=[]
	for i,line in enumerate(lines):
		if i==0:
			continue
		data = line.split() 
		if len(data)!=5:
			break
		idx_i.append(int(data[0]))
		idx_j.append(int(data[1]))
		offsets.append([float(data[2]), float(data[3]), float(data[4])])
	offsets=np.array(offsets)
	return cutoff, idx_i, idx_j, offsets
		



def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--input_atoms_file_name', default="atoms.xyz", type=str, help="Input file name atoms coordinates(in Ang)")
	parser.add_argument('--input_box_file_name', default="box.data", type=str, help="Input file name of box (in Ang)")
	parser.add_argument('--input_neighbor_file_name', default="neigh.data", type=str, help="Input file for neghbor list (in Ang)")
	parser.add_argument('--output_energy_file_name', default="energy.out", type=str, help="name of  output energy")
	parser.add_argument('--output_forces_file_name', default="forces.out", type=str, help="name of  output forces")
	parser.add_argument('--list_models', type=str, nargs='+', help="list of directory containing fitted models (at least one file), ....")
	parser.add_argument('--verbose', type=int, default=0, help="verbose (default=0)")

	#if no command line arguments are present, config file is parsed
	config_file='config.txt'
	fromFile=False
	if len(sys.argv) == 1:
		fromFile=False
	if len(sys.argv) == 2 and sys.argv[1].find('--') == -1:
		config_file=sys.argv[1]
		fromFile=True

	if fromFile is True:
		print("Try to read configuration from ",config_file, "file")
		if os.path.isfile(config_file):
			args = parser.parse_args(["@"+config_file])
		else:
			args = parser.parse_args(["--help"])
	else:
		args = parser.parse_args()

	return args


print("get arguments",flush=True)
args = getArguments()
atoms = readAtoms(args.input_atoms_file_name)
atoms = readBox(args.input_box_file_name, atoms)

lmodels=args.list_models
lmodels=lmodels[0].split()
if args.verbose>1:
	print(lmodels)
	print("--------- Input geometry --------------------")
	print("PBC : ",atoms.get_pbc())
	print("Cell : " ,atoms.get_cell())
	print("Z : " , atoms.get_atomic_numbers())
	print("Positions : ",atoms.get_positions())
	print("---------------------------------------------")
cutoff, idx_i, idx_j, offsets = readNeighbor(args.input_neighbor_file_name)
	
if args.verbose>1:
	print("cutoff=",cutoff)
	print("idx_i=",idx_i)
	print("idx_j=",idx_j)
	print("offsets=",offsets)

if len(idx_i)<1:
	idx_i=None
	idx_j=None
	offsets=None
"""
idx_i=None
idx_j=None
offsets=None
"""

atoms.calc = Predictor(
		lmodels,
		atoms,
		conv_distance=1/units.Bohr,
		conv_energy=1/units.Hartree,
		verbose=args.verbose,
		idx_i=idx_i,
		idx_j=idx_j,
		offsets=offsets,
		)

#epot = atoms.get_potential_energy() / len(atoms)
epot = atoms.get_potential_energy()
#print("epot=",epot)
forces = atoms.get_forces()
#print("forces=",forces)
np.savetxt(args.output_energy_file_name,epot)
#with open(args.output_energy_file_name,"w") as f:
#	print(epot[0],file=f)
np.savetxt(args.output_forces_file_name,forces)
#with open(args.output_forces_file_name,"w") as f:
#	forces.tofile(f, sep='', format='%s')

