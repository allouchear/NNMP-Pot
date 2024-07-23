
**Edit env.sh to set paths to your configuration (openmi,...)**

## compile lammps with nnmp
cp env.sh <path-to-LAMMPS>/
cp xcompWithNNMP <path-to-LAMMPS>/
cp -r Interfaces/LAMMPS/src/USER-NNMP <path-to-LAMMPS>/src
cd <path-to-LAMMPS>/src
./xcompWithNNMP

## to run lammps with NNMP
set python env (conda ?)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:yourpathtopythonlib ( ? ~/anaconda3/envs/tf/lib)
<path-to-LAMMPS>/lmp_serial < inputLAMMPS