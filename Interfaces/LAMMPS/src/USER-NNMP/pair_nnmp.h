#ifdef PAIR_CLASS

PairStyle(nnmp,PairNNMP)

#else

#ifndef LMP_PAIR_NNMP_H
#define LMP_PAIR_NNMP_H

#include <string>
#include <vector>
#include "pair.h"
#include <Python.h>

namespace LAMMPS_NS {

class PairNNMP : public Pair {

	public:
		PairNNMP(class LAMMPS *);
		virtual ~PairNNMP();
		virtual void compute(int, int);
		virtual void settings(int, char **);
		virtual void coeff(int, char **);
		virtual void init_style();
		virtual double init_one(int, int);
		virtual void write_restart(FILE *);
		virtual void read_restart(FILE *);
		virtual void write_restart_settings(FILE *);
		virtual void read_restart_settings(FILE *);
		virtual void InitializeModule();
		virtual void computeVirial();

	protected:
		virtual void computeInverseTagPosType();
		virtual void computeIndex();
		virtual void computeBox();
		virtual void allocate();
		virtual void computeEnergyAndForces();
		double box[3][3];
		int pbc[3];
		double cflength;
		double cfenergy;
		double maxCutoffRadius;
		std::vector<std::string> elements;
		std::vector< std::vector<double> > pos; //  real atoms positions
		std::vector< int > tag2i; //  Inverse mapping from tag to x/f atom index
		std::vector< int > tag2type; //  Inverse mapping type
		std::vector< int > idx_i;
		std::vector< int > idx_j;
		std::vector< std::vector<double> > offsets;
		std::string moduleName;
		std::string listmodels;
		PyObject *pModule;
		PyObject *pComputeEnergyAndForces;
		PyObject* vectorToList_Str(char** data, int size);
		PyObject* vectorToList_Float(double* data, int size);
		PyObject* vectorToList_Int(int* data, int size);
		double* listToVector_Float(PyObject* incoming);
};

}

#endif
#endif
