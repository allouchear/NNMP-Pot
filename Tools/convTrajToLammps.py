import ase
from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.geometry.cell import cell_to_cellpar 
from math import *
import numpy as np


def getBox(cell):
	abcangles=cell_to_cellpar(cell, radians=True)
	a= abcangles[0]
	b= abcangles[1]
	c= abcangles[2]
	alpha=abcangles[3]
	beta=abcangles[4]
	gamma=abcangles[5]

	lx = a
	xy = b*cos(gamma)
	xz = c*cos(beta)
	ly=sqrt(abs(b*b-xy*xy))
	yz=(b*cos(alpha)-xy*xz)/ly
	lz=sqrt(abs(c*c-xz*xz-yz*yz))
	xlo=0
	ylo=0
	zlo=0
	xhi=xlo+lx
	yhi=ylo+ly
	zhi=zlo+lz
	xlo_bound = xlo + min(0.0,xy,xz,xy+xz)
	xhi_bound = xhi + max(0.0,xy,xz,xy+xz)
	ylo_bound = ylo + min(0.0,yz)
	yhi_bound = yhi + max(0.0,yz)
	zlo_bound = zlo
	zhi_bound = zhi
	V=np.array([[xlo_bound,xhi_bound,xy],[ylo_bound,yhi_bound,xz],[zlo_bound,zhi_bound,yz]])
	return V


traj =  Trajectory("mdNVE.traj")
fname="traj.lammpstrj"
f = open(fname, "w")
timestep=0.5
t=0
for atoms in traj:
	st="ITEM: TIMESTEP"+"\n"
	st+= str(t)+"\n"
	f.write(st)
	t += timestep
	pos = atoms.get_positions()
	z = atoms.get_atomic_numbers()
	cell = atoms.get_cell()
	st="ITEM: NUMBER OF ATOMS"+"\n"
	st+= str(len(pos))+"\n"
	st+="ITEM: BOX BOUNDS xy xz yz pp pp pp"+"\n"
	f.write(st)
	V=getBox(cell)
	lat = ""
	for c in V:
		lat +='{:0.10f} {:0.10f} {:0.10f}'.format(c[0], c[1],c[2])+"\n"
	f.write(lat)
	st="ITEM: ATOMS id type x y z"+"\n"
	f.write(st)
	for ia in range(len(pos)):
		xyz='{:5d} {:5d} {:0.10f} {:0.10f} {:0.10f}'.format(ia+1,z[ia], pos[ia][0], pos[ia][1],pos[ia][2])+"\n"
		f.write(xyz)
f.close()
print("See ",fname, " file")
