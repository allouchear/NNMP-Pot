import ase
from ase.io.trajectory import Trajectory
from ase.io import read, write


traj =  Trajectory("a206.traj")
fname="a206.xyz"
f = open(fname, "w")
for atoms in traj:
	pos = atoms.get_positions()
	s = atoms.get_chemical_symbols()
	cell = atoms.get_cell()
	
	lat = str(len(pos)+len(cell))+"\n\n"
	for c in cell:
		lat +='{:5s} {:0.10f} {:0.10f} {:0.10f}'.format("Tv", c[0], c[1],c[2])+"\n"
	f.write(lat)
	for ia in range(len(pos)):
		xyz='{:5s} {:0.10f} {:0.10f} {:0.10f}'.format(s[ia], pos[ia][0], pos[ia][1],pos[ia][2])+"\n"
		f.write(xyz)
f.close()
print("See ",fname, " file")
