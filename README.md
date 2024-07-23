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

```console
git clone https://github.com/allouchear/NNMP-Pot.git

## How to use it 
See examples/H2O folder. 
You can find a input file for train (train.inp) and a script to run it (xtrain).
NNMP-Pot is intefraced with [ase](https://wiki.fysik.dtu.dk/ase/) and [LAMMPS](https://www.lammps.org/#gsc.tab=0)
After training, you can use ase (see xaseMD) or lammps (see xlammpsNVE).
NNMP-Pot is tested with ase and LAMMPS for NVE and NVT molecular dynamics **but not fully tested with NPT one**.

