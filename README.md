# NNMP-Pot  A neural network message passing potential
======================================================

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Requirement
 - tensorflow, 
 - tensorflow_probability, 
 - tensorflow_addons, 
 - nvidia-tensorrt

After installation of conda, and activation of your envoronnement,  Type : 
```console
pip install tensorflow==2.12.0
pip install tensorflow_probability==0.19.0
pip install tensorflow_addons==0.20.0
pip3 install nvidia-tensorrt
```

## Installation

Using git,  Type : 
```console
git clone https://github.com/allouchear/NNMP-Pot.git
```
You can also download the .zip file of NNMP-Pot : Click on Code and Download ZIP
The code is interfaced with LAMMPS. To compile LAMMPS for NNMP-Pot, see README file in NNMP-Pot/Interfaces /LAMMPS/

## How to use it 
See examples/H2O folder. 
You can find a input file for train (train.inp) and a script to run it (xtrain).
NNMP-Pot is intefraced with [ase](https://wiki.fysik.dtu.dk/ase/) and [LAMMPS](https://www.lammps.org/#gsc.tab=0)
After training, you can use ase (see xaseMD) or lammps (see xlammpsNVE).
NNMP-Pot is tested with ase and LAMMPS for NVE and NVT molecular dynamics **but not fully tested with NPT one**.

## Implemented Methods
 - MPNN : similar to [PhyNet](https://github.com/MMunibas/PhysNet) model. However, we can use other type of basis functions, repulsive potential, several methods to add electrostatic potential. 
You can find a input file for train (train.inp) and a script to run it (xtrain).
 - EAMM : Embeded Atom Neural network
 - EANNP : MPNN using Embeded atom overlap as basis functions
 - EMMPNN : MPNN with Element modes instead a embaded layer as input

**See getArguments function in Utils/UtilsFunctions.py file for more details about all available parameters.**

## Contributors
The code is written by Abdul-Rahman Allouche.\
A part of the code (written for tensorflow2) is based on [PhysNet](https://github.com/MMunibas/PhysNet) code (written for tensorflow1)
