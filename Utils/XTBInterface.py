import os
import subprocess
import sys
import numpy as np
import tensorflow as tf
from Utils.PeriodicTable import *

class XTBInterface:
	def __str__(self):
		return "XTBInterface:"

	def __init__(self,
		xtbMethod,
		dtype=tf.float32            #single or double precision
		):
		self._gfn=0
		#print("xtbMethod=",xtbMethod)
		if len(xtbMethod)>=4:
			self._gfn=int(xtbMethod[3])
		self._dtype=dtype

	@property
	def gfn(self):
		return self._gfn

	@property
	def dtype(self):
		return self._dtype

	def get_step(self, iv):
		i=self.varnumlines[iv]
		j=self.varnumcols[iv]
		return self.steps[i][j]

	def save_xyz(self, Z, R, workingDir):
		fileName=workingDir+"/data.xyz"
		f = open(fileName,"w")
		periodicTable=PeriodicTable()
		nAtoms=Z.shape[0]
		f.write(str(nAtoms)+"\n")
		f.write("Coordinates in Angstrom\n")
		Ra = R*0.52917721
		for ia in range(nAtoms):
			symbol=periodicTable.elementZ(int(Z[ia])).symbol
			f.write('{:<6s} '.format(symbol))
			for c in range(3):
				f.write('{:>20.14f} '.format(Ra[ia][c]))
			f.write("\n")
		f.close()
		return fileName
	def read_xtboutput(self, charge_file, grad_file):
		''' read xtb qmmm output file (charge_file, grad_file)
		return: 
			energy (python float)
			gradients (n * 3D)
			charges (n)
		'''
		# read charge file and get the number of atoms
		charges = []
		with open(charge_file, 'r') as charge:
			charge_lines = charge.readlines()
			for line in charge_lines:
            			if line.strip():
                			charges.append(float(line.strip()))
		# read energy& grad
		nAtoms = len(charges)
		gradients = []
		with open(grad_file, 'r') as grad:
			grad_lines = grad.readlines()
			n = 0
			for line in grad_lines:
				n += 1
				# the first line is useless
				if n == 1:
					continue
				# the second line contains the energy
				if n == 2:
					data = line.strip().split()
					# the unit of energy in xtb is hartrees
					energy = float(data[6])
				if n > nAtoms + 2 and n <= nAtoms + nAtoms + 2:
					data = line.strip().split()
					# force
					g = []
					for i in range(3):
						# the unit of grad in xtb output should be hartrees/bohr
						g.append(float(data[i].replace('D','E')) )
					gradients.append(g)
		return energy, gradients, charges

	def run(self, Z, R, workingDir, xtbFile, charge=None):
		path= os.path.dirname(xtbFile)
		os.environ['XTBPATH']=path
		if not os.path.exists(workingDir):
			os.makedirs(workingDir)
		xtbxyz=self.save_xyz(Z,R,workingDir)
		outputFile=workingDir+"/"+"output.txt"
		xtbpcfile = workingDir + r'/pcharge'
		xtbcharges = workingDir + r'/charges'
		xtbgrad = workingDir + r'/gradient'
		xtbrestart = workingDir + r'/xtbrestart'
		energy = workingDir + r'/energy'
		fo = open(outputFile,"wb")
		xtbbin="xtb"
		if charge is None:
			subprocess.call([xtbbin, xtbxyz, '--grad', '--gfn', str(self.gfn)], stdout = fo, stderr = fo, cwd=workingDir)
		else:
			subprocess.call([xtbbin, xtbxyz, '--grad', '--gfn', str(self.gfn),'--chrg', str(int(charge+0.5))], stdout = fo, stderr = fo, cwd=workingDir)
		fo.close()
		energy, gradients, charges= self.read_xtboutput(xtbcharges, xtbgrad)
		return energy, gradients, charges, outputFile

