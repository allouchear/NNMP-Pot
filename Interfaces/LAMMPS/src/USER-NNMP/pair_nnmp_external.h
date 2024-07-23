#ifdef PAIR_CLASS

PairStyle(nnmp/external,PairNNMPExternal)

#else

#ifndef LMP_PAIR_NNMP_EXTERNAL_H
#define LMP_PAIR_NNMP_EXTERNAL_H

#include <string>
#include <vector>
#include "pair.h"

namespace LAMMPS_NS {

class PairNNMPExternal : public Pair {

	public:
		PairNNMPExternal(class LAMMPS *);
		virtual ~PairNNMPExternal();
		virtual void compute(int, int);
		virtual void settings(int, char **);
		virtual void coeff(int, char **);
		virtual void init_style();
		virtual double init_one(int, int);
		virtual void write_restart(FILE *);
		virtual void read_restart(FILE *);
		virtual void write_restart_settings(FILE *);
		virtual void read_restart_settings(FILE *);
		virtual void writeNeighborList(const std::string& WDir);
		virtual void writeAtoms(const std::string& WDir);
		virtual void writeBox(const std::string& WDir);

	protected:
		virtual void computeInverseTagPosType();
		virtual void allocate();
		double cflength;
		double cfenergy;
		double maxCutoffRadius;
		char* directory;
		std::vector<std::string> elements;
		char* command;
		std::vector< std::vector<double> > pos; //  real atoms positions
		std::vector< int > tag2i; //  Inverse mapping from tag to x/f atom index
		std::vector< int > tag2type; //  Inverse mapping type
};

}

#endif
#endif
