#!/usr/bin/env python

# n2p2 - A neural network potential package
# Copyright (C) 2018 Andreas Singraber (University of Vienna)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

###############################################################################
# File converter from VASP OUTCAR to input.data format.
# Works also if OUTCAR contains trajectories.
# Tested with VASP 5.2.12
###############################################################################

EV_TO_HARTREE=1.0/27.21138469
ANG_TO_BOHR=1.0/0.52917721

import numpy as np
import sys

def print_usage():
    sys.stderr.write("USAGE: {0:s} <in_file> <<out_file>>\n".format(sys.argv[0]))
    sys.stderr.write("       <in_file> .... OUTCAR file name.\n")
    sys.stderr.write("       <out_file> ... Output file name (optional).\n")
    return

if len(sys.argv) < 2 or sys.argv[1] in ["-?", "-h", "--help"]:
    print_usage()
    sys.exit(1)

file_name = sys.argv[1]
if len(sys.argv) > 2:
    outfile_name = sys.argv[2]
else:
    outfile_name = None

# Read in the whole file first.
f = open(file_name, "r")
lines = [line for line in f]
f.close()

# If OUTCAR contains ionic movement run (e.g. from an MD simulation) multiple
# configurations may be present. Thus, need to prepare empty lists.
lattices   = []
energies   = []
atom_lists = []

# Loop over all lines.
elements = []
for i in range(len(lines)):
    line = lines[i]
    # Collect element type information, expecting VRHFIN lines like this:
    #
    # VRHFIN =Cu: d10 p1
    #
    if "VRHFIN" in line:
        elements.append(line.split()[1].replace("=", "").replace(":", ""))
    # VASP specifies how many atoms of each element are present, e.g.
    #
    # ions per type =              48  96
    #
    if "ions per type" in line:
        atoms_per_element = [int(it) for it in line.split()[4:]]
    # Simulation box may be specified multiple times, I guess this line
    # introduces the final lattice vectors.
    if "VOLUME and BASIS-vectors are now" in line:
        lattices.append([lines[i+j].split()[0:3] for j in range(5, 8)])
    # Total energy is found in the line with "energy  without" (2 spaces) in
    # the column with sigma->0:
    #
    # energy  without entropy=     -526.738461  energy(sigma->0) =     -526.738365
    #
    if "energy  without entropy" in line:
        energies.append(line.split()[6])
    # Atomic coordinates and forces are found in the lines following
    # "POSITION" and "TOTAL-FORCE".
    if "POSITION" in line and "TOTAL-FORCE" in line:
        atom_lists.append([])
        count = 0
        for ei in range(len(atoms_per_element)):
            for j in range(atoms_per_element[ei]):
                atom_line = lines[i+2+count]
                atom_lists[-1].append(atom_line.split()[0:6])
                atom_lists[-1][-1].extend([elements[ei]])
                atom_lists[-1][-1].extend([charges[count]])
                count += 1
    if "Hirshfeld " in line and "charges:" in line:
        count = 0
        charges=[]
        for ei in range(len(atoms_per_element)):
            for j in range(atoms_per_element[ei]):
                charge = lines[i+3+count].split()[2]
                charges.append(charge)
                count += 1

# Sanity check: do all lists have the same length. 
if not (len(lattices) == len(energies) and len(energies) == len(atom_lists)):
    raise RuntimeError("ERROR: Inconsistent OUTCAR file.")

# Open output file or write to stdout.
if outfile_name is not None:
    f = open(outfile_name, "w")
else:
    f = sys.stdout

# Write configurations in "input.data" format.
for i, (lattice, energy, atoms) in enumerate(zip(lattices, energies, atom_lists)):
    f.write("begin\n")
    f.write("comment source_file_name={0:s} structure_number={1:d}\n".format(file_name, i + 1))
    for i in range(3):
       for j in range(3):
                lattice[i][j]=float(lattice[i][j])*ANG_TO_BOHR
    f.write("lattice {0:22.14f} {1:22.14f} {2:22.14f}\n".format(lattice[0][0], lattice[0][1], lattice[0][2]))
    f.write("lattice {0:22.14f} {1:22.14f} {2:22.14f}\n".format(lattice[1][0], lattice[1][1], lattice[1][2]))
    f.write("lattice {0:22.14f} {1:22.14f} {2:22.14f}\n".format(lattice[2][0], lattice[2][1], lattice[2][2]))
    for a in atoms:
        x=float(a[0])*ANG_TO_BOHR
        y=float(a[1])*ANG_TO_BOHR
        z=float(a[2])*ANG_TO_BOHR
        fx=float(a[3])*EV_TO_HARTREE/ANG_TO_BOHR
        fy=float(a[4])*EV_TO_HARTREE/ANG_TO_BOHR
        fz=float(a[5])*EV_TO_HARTREE/ANG_TO_BOHR
        f.write("atom {0:22.14f} {1:22.14f} {2:22.14f} {3:2s} {4:>14.8f} {5:s} {6:>22.14f} {7:>22.14f} {8:>22.14f}\n".format(x, y, z, a[6], float(a[7]), "0.0", fx, fy, fz))
    energy=float(energy)*EV_TO_HARTREE
    f.write("energy {0:22.14f}\n".format(energy))
    f.write("charge {0:22.14f}\n".format(0.0))
    f.write("end\n")
