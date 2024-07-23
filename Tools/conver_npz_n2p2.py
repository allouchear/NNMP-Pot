import os
import sys
import numpy as np
import argparse

sys.path.insert(0,"../Utils")
from PeriodicTable import *

"""
begin
comment by xLogML : From a0.log file, All value are in AU
atom 0.000000000000 5.388582648150 0.000000000000 C -0.073760000000 0.000000000000 0.000000000000 -0.000006363000 0.000000000000
atom 0.000000000000 2.700223176744 0.000000000000 C 0.050820000000 0.000000000000 0.000000000000 0.000009271000 0.000000000000
atom 2.360941613109 6.686180616316 0.000000000000 C -0.215490000000 0.000000000000 -0.000003752000 0.000000796000 0.000000000000
energy  -921.92601237700001
charge  0.00000000000000
dipole 0.00000000000179 -0.00000000000134 0.00000000000002
end
"""
def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--conv_distance', default=1.0, type=float, help="convert coefficient of distance")
	parser.add_argument('--conv_energy', default=1.0, type=float, help="convert coefficient of energy")
	parser.add_argument('--conv_dipole', default=1.0, type=float, help="convert coefficient of dipole")

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

def convert(path, outfile, convDistance=1.0, convEnergy=1.0, convDipole=1.0):
	try:
		convForce = convEnergy/convDistance
		dictionary = np.load(path)
		pt = PeriodicTable()
		numMol = len(dictionary['N'])
		f = open(outfile, "w")
		#f.write("Woops! I have deleted the content!")
		R = dictionary['R']
		R *= convDistance
		F = dictionary['F']
		F *= convForce
		D = dictionary['D']
		D *= convDipole
		E = dictionary['E']
		E *= convEnergy
		Q = dictionary['Q']
		Cell = None
		if 'Cell' in dictionary:
			Cell = dictionary['Cell']
		for idx in range(0,numMol):
			if numMol > 100 and idx%100==0:
				print("mol "+str(idx)+" / "+str(numMol))
			print("mol "+str(idx)+" / "+str(numMol))
			sout ="begin\n"
			sout +='comment by  begin conver_npz_n2p2 idx= {:2}\n'.format(idx)
			print(Cell[idx])
			if Cell is not None and not np.isnan(Cell[idx]).any():
				for il in range(0,Cell[idx].shape[0]):
					sout1 ='lattice {:20.14} {:20.14} {:20.14} '.format(Cell[idx][il][0], Cell[idx][il][1],Cell[idx][il][2])
					sout+=sout1+"\n"

			natoms = (dictionary['N'][idx])
			print("natoms "+str(natoms))
			for ia in range(0,natoms):
				sout1 ='atom {:20.14} {:20.14} {:20.14} '.format(R[idx][ia][0], R[idx][ia][1],R[idx][ia][2])
				z=dictionary['Z'][idx][ia]
				#print("z=",z)
				e = pt.elementZ(z)
				#print("e=",e.symbol)
				sout2 =' {:6s}'.format(e.symbol)
				if 'Qa' in dictionary:
					sout3 =' {:20.14} '.format(dictionary['Qa'][idx][ia])
				else:
					sout3 =' {:20.14} '.format(0.0)
				sout4 =' {:20.14} '.format(0.0)
				sout5 ='{:20.14} {:20.14} {:20.14} '.format(F[idx][ia][0], F[idx][ia][1],F[idx][ia][2])
				sout+=sout1+sout2+sout3+sout4+sout5+"\n"
			sout += 'energy {:20.14}\n'.format(E[idx])
			sout += 'charge {:20.14}\n'.format(Q[idx])
			sout += 'dipole {:20.14} {:20.14} {:20.14}\n'.format(D[idx][0], D[idx][1], D[idx][2])
			sout += "end\n"
			f.write(sout)
			print(sout)
		f.close()
	except Exception as Ex:
		print("Read Failed.", Ex)
		raise Ex
	return

args = getArguments()
npzfile="data.npz" 
npzfile="data_half4.npz"
npzfile="periodic.npz"
print("npz data file : "+npzfile)
#n2p2file="input.data" 
n2p2file="res.data" 
convert(npzfile, n2p2file, convDistance=args.conv_distance, convEnergy=args.conv_energy, convDipole=args.conv_dipole)
print("See "+n2p2file+" file")
#convert("input.data")

