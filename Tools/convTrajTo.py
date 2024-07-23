import ase
from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.io.vasp import write_vasp_xdatcar
from ase.io.netcdftrajectory import write_netcdftrajectory
from ase.io.cif import write_cif
from ase.io.gromacs import write_gromacs
from ase.io.lammpsdata import write_lammps_data
from ase.io.vasp import write_vasp_xdatcar
from ase.io.extxyz import write_extxyz


bec =  Trajectory("mdNVE.traj")
bec_new_nc = write_netcdftrajectory("mdNVE.nc", bec)
bec_new_cif = write_cif("mdNVE.cif", bec)
bec_new_gro = write_cif("mdNVE.gro", bec)
bec_new_xdat = write_vasp_xdatcar("mdNVE.xdatcar", bec, label=None)
bec_new_xyz = write_extxyz("mdNVE.xyz",bec)
#bec_new_lam = write_lammps_data("mdNVE.lammpstrj", bec)

