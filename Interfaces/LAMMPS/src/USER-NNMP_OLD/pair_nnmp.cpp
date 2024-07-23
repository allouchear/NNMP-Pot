#include <mpi.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "pair_nnmp.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "utils.h"
#include <iostream>
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include <iomanip>

using namespace LAMMPS_NS;
using namespace std;

PyObject* PairNNMP::vectorToList_Str(char** data, int size)
{
	int i;
	PyObject* listObj = PyList_New(size);
	if (!listObj) 
		error->all(FLERR," Unable to allocate memory for Python list");
	for (i = 0; i < size; i++)
	{
		PyObject *num = PyUnicode_FromString(data[i]);
		if (!num) {
			Py_DECREF(listObj);
			error->all(FLERR," Unable to allocate memory for Python list");
		}
		PyList_SET_ITEM(listObj, i, num);
	}
	return listObj;
}
PyObject* PairNNMP::vectorToList_Float(double* data, int size)
{
	int i;
	PyObject* listObj = PyList_New(size);
	if (!listObj) 
		error->all(FLERR,"  Unable to allocate memory for Python list");
	for (i = 0; i < size; i++)
	{
		PyObject *num = PyFloat_FromDouble(data[i]);
		if (!num) {
			Py_DECREF(listObj);
			error->all(FLERR,"  Unable to allocate memory for Python list");
		}
		PyList_SET_ITEM(listObj, i, num);
	}
	return listObj;
}
PyObject* PairNNMP::vectorToList_Int(int* data, int size)
{
	int i;
	PyObject* listObj = PyList_New(size);
	if (!listObj) 
		error->all(FLERR,"  Unable to allocate memory for Python list");
	for (i = 0; i < size; i++)
	{
		PyObject *num = PyFloat_FromDouble(data[i]);
		if (!num) {
			Py_DECREF(listObj);
			error->all(FLERR,"  Unable to allocate memory for Python list");
		}
		PyList_SET_ITEM(listObj, i, num);
	}
	return listObj;
}
// PyObject -> Vector
double* PairNNMP::listToVector_Float(PyObject* incoming)
{
	double* data = NULL;
	if (PyList_Check(incoming))
	{
		int i;
		Py_ssize_t j = 0;
		int size = PyList_Size(incoming);
		data = (double*) malloc(size*sizeof(double));
		for(j = 0, i=0; j < PyList_Size(incoming); j++,i++)
				data[i] = PyFloat_AsDouble(PyList_GetItem(incoming, j));
	}
	else error->all(FLERR,"  Passed PyObject pointer was not a list or tuple!");

	return data;
}
//////////////////////////////////////////////////////////////////////////////////////
PairNNMP::PairNNMP(LAMMPS *lmp) : Pair(lmp)
{

	idx_i=vector<int>();
	idx_j=vector<int>();
	offsets=vector< vector<double> >();
	pModule = NULL;
	pGetEnergy = NULL;
	pGetEnergyAndForces = NULL;
	pComputeEnergyAndForces = NULL;
	pSetData = NULL;
	pData = NULL;
	initialized = 0;
	moduleName="NNMPLAMMPSModule.py";
	listmodels="Checkpoint";
}
//////////////////////////////////////////////////////////////////////////////////////
PairNNMP::~PairNNMP()
{
	idx_i=vector<int>();
	idx_j=vector<int>();
	offsets=vector< vector<double> >();
	pModule = NULL;
	pGetEnergy = NULL;
	pGetEnergyAndForces = NULL;
	pComputeEnergyAndForces = NULL;
	pSetData = NULL;
	pData = NULL;
	initialized = 0;
	moduleName="NNMPLAMMPSModule.py";
	listmodels="Checkpoint";
}
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::getEnergy()
{
	double energy;
	if(!pData) setPyData();
	if(pData && pGetEnergy && PyCallable_Check(pGetEnergy)) 
	{
		PyObject* listRes = NULL;
		/*
		PyObject *pArgs = PyTuple_New(2);
               	PyTuple_SetItem(pArgs, 0, pData );
		PyObject* pEner = PyObject_CallObject(pGetEnergy, pArgs);
		Py_DECREF(pArgs);
		*/
		//setCoordinates();
		listRes = PyObject_CallFunctionObjArgs(pGetEnergy, pData,  NULL);
            	if (listRes != NULL)
		{
			energy = PyFloat_AsDouble(PyList_GetItem(listRes, 0));
			energy /= cfenergy;
        	}
		else error->all(FLERR,"pEnergy = NULL in getEnergy");
	}
	if (eflag_global)
		ev_tally(0,0,atom->nlocal,1,energy,0.0,0.0,0.0,0.0,0.0);

}
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::getEnergyAndForces()
{
	double energy;
	if(!pData) setPyData();
	if(pData && pGetEnergyAndForces && PyCallable_Check(pGetEnergyAndForces))
	{
		PyObject* listRes = NULL;
               	/* fprintf(stderr,"ok pGetEnergyAndForces in getEnergyAndForces\n");*/
		/*
		PyObject *pArgs = PyTuple_New(2);
               	PyTuple_SetItem(pArgs, 0, pData );
		PyObject* pEner = PyObject_CallObject(pGetEnergy, pArgs);
		Py_DECREF(pArgs);
		*/
		//setCoordinates();
               	/* fprintf(stderr,"End setCoordinates in getEnergyAndForces\n");*/
		listRes = PyObject_CallFunctionObjArgs(pGetEnergyAndForces, pData,  NULL);
            	if (listRes != NULL)
		{
			energy = PyFloat_AsDouble(PyList_GetItem(listRes, 0));
			energy /= cfenergy;
			int nAtoms =pos.size();
			double const cfforce = cfenergy / cflength;
			int cc=0;
			int j=1;
			for(int i=0;i<nAtoms;i++)
			{
				cc=tag2i[i];
				for(int k=0;k<3;k++) 
				{
					double f = PyFloat_AsDouble(PyList_GetItem(listRes, j));
					atom->f[cc][k] = f/ cfforce;
					j++;
				}
			}
		}
		else error->all(FLERR,"listRed of pGetEnergyAndForces = NULL in getEnergyAndForces");
	}
	if (eflag_global)
		ev_tally(0,0,atom->nlocal,1,energy,0.0,0.0,0.0,0.0,0.0);
}
//////////////////////////////////////////////////////////////////////////////////////
std::string getfilename(const std::string path)
{
	string sep = "/";
#ifdef _WIN32
	sep = "\\";
#endif
	string p = path.substr(path.find_last_of(sep) + 1);
	size_t dot_i = p.find_last_of('.');
	return p.substr(0, dot_i);
}
string getpathname(const std::string& path) {

	char sep = '/';
#ifdef _WIN32
	sep = '\\';
#endif
	size_t i = path.rfind(sep, path.length());
	if (i != string::npos)
		return(path.substr(0, i));

	return(".");
}
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::InitializeModule()
{
	if(!Py_IsInitialized()) Py_Initialize();
	string pathdir = getpathname(moduleName);
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("import ase");
	string path="sys.path.append(\""+pathdir+"\")";
	string basename=getfilename(moduleName);
	PyRun_SimpleString(path.c_str());
	PyObject *pName = PyUnicode_DecodeFSDefault(basename.c_str());
	if(pName) 
	{
		pModule = PyImport_Import(pName);
		Py_DECREF(pName);
		pName = NULL;
		if(!pModule) 
			error->all(FLERR,"problem with PyImport_Import in InitializeModule");
	}
	if (pModule)
	{
        	pGetEnergy = PyObject_GetAttrString(pModule, "getEnergy");
		if(!pGetEnergy) 
			error->all(FLERR,"problem with getEnergy");
	}
	if (pGetEnergy)
	{
        	pGetEnergyAndForces = PyObject_GetAttrString(pModule, "getEnergyAndForces");
		if(!pGetEnergyAndForces) 
			error->all(FLERR,"problem with getEnergyAndForces");
	}
	if (pGetEnergyAndForces)
	{
        	pSetData = PyObject_GetAttrString(pModule, "setData");
		if(!pSetData) 
			error->all(FLERR,"problem with setData");
	}
	if (pGetEnergyAndForces)
	{
        	pComputeEnergyAndForces = PyObject_GetAttrString(pModule, "computeEnergyAndForces");
		if(!pComputeEnergyAndForces) 
			error->all(FLERR,"problem with computeEnergyAndForces");
	}
	initialized = 1;
}
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::compute(int eflag, int vflag)
{
	if(eflag || vflag) ev_setup(eflag,vflag);
	else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;
	computeInverseTagPosType();
	computeIndex();
	computeBox();
	/*
	setPyData();
	getEnergyAndForces();
	*/
	computeEnergyAndForces();

	// If virial needed calculate via F dot r.
	if (vflag_fdotr) 
		virial_fdotr_compute();
}
//////////////////////////////////////////////////////////////////////////////////////
//   global settings
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::settings(int narg, char **arg)
{
	int iarg = 0;

	// elements list is mandatory
	if (narg < 1) error->all(FLERR,"Illegal pair_style command : narg<1");

	// element list, mandatory
	int len = strlen(arg[iarg]) + 1;
	string selements=string(arg[iarg]);
	stringstream sse(selements);
	elements = vector<string>();
	while(sse.good())
	{
		string e;
		sse>>e;
		elements.push_back(e);
	}
	iarg++;
	vector<FILE*> fp;
	if (screen) fp.push_back(screen);
	if (logfile) fp.push_back(logfile);

	// default settings
	cflength = 1.0;
	cfenergy = 1.0;
	moduleName="NNMPLAMMPSModule.py";
	listmodels="Checkpoint";

	while(iarg < narg) {
		// set NNMP module
		if (strcmp(arg[iarg],"module") == 0) {
			if (iarg+2 > narg)
				error->all(FLERR,"Illegal pair_style command: module");
		moduleName=string(arg[iarg+1]);
		iarg += 2;
		// set listmodels
		} else if (strcmp(arg[iarg],"listmodels") == 0) {
			if (iarg+2 > narg)
				error->all(FLERR,"Illegal pair_style command: listmodels");
		listmodels=string(arg[iarg+1]);
		iarg += 2;
		}
		// length unit conversion factor
		else if (strcmp(arg[iarg],"cflength") == 0) {
			if (iarg+2 > narg)
				error->all(FLERR,"Illegal pair_style command : cflength");
			cflength = utils::numeric(FLERR,arg[iarg+1],false,lmp);
			iarg += 2;
			// energy unit conversion factor
		} else if (strcmp(arg[iarg],"cfenergy") == 0) {
			if (iarg+2 > narg)
				error->all(FLERR,"Illegal pair_style command : cfenergy");
			cfenergy = utils::numeric(FLERR,arg[iarg+1],false,lmp);
			iarg += 2;
		} else 
		{
			string s = "Illegal pair_style command : other";
			s += " arg =";
			s +=string(arg[iarg]);
			//error->all(FLERR,"Illegal pair_style command : other");
			error->all(FLERR,s) ;
		}
	}
	for (auto f : fp)
	{
		fprintf(f, "*****************************************"
			   "**************************************\n");
		fprintf(f, "pair_style nnmp settings:\n");
		fprintf(f, "---------------------------------\n");
		fprintf(f, "elements        = %s\n", selements.c_str());
		fprintf(f, "pyModule        = %s\n", moduleName.c_str());
		fprintf(f, "list of models  = %s\n", listmodels.c_str());
		fprintf(f, "cflength        = %16.8E\n", cflength);
		fprintf(f, "cfenergy        = %16.8E\n", cfenergy);
		fprintf(f, "*****************************************"
			   "**************************************\n");
		fprintf(f, "CAUTION: Please carefully check whether this map between LAMMPS\n");
		fprintf(f, "         atom types and element strings is correct:\n");
		fprintf(f, "---------------------------\n");
		fprintf(f, "LAMMPS type  |  NNP element\n");
		fprintf(f, "---------------------------\n");
		int lammpsNtypes = elements.size();
		for (int i = 0; i < lammpsNtypes; ++i)
		{
			fprintf(f, "%11d  |  %2s \n", i+1, elements[i].c_str());
		}
		fprintf(f, "*****************************************"
			   "**************************************\n");
	}
	InitializeModule();
}

//////////////////////////////////////////////////////////////////////////////////////
//   set coeffs for one or more type pairs
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::coeff(int narg, char **arg)
{
	if (!allocated) allocate();

	if (narg != 3) error->all(FLERR,"Incorrect args for pair coefficients");

	int ilo,ihi,jlo,jhi;
	utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
	utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

	maxCutoffRadius = utils::numeric(FLERR,arg[2],false,lmp);

	int count = 0;
	for(int i=ilo; i<=ihi; i++) {
		for(int j=MAX(jlo,i); j<=jhi; j++) {
			setflag[i][j] = 1;
			count++;
		}
	}

	if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

//////////////////////////////////////////////////////////////////////////////////////
//   init specific to this pair style
//////////////////////////////////////////////////////////////////////////////////////

void PairNNMP::init_style()
{
	if (comm->nprocs > 1)
	{
		error->all(FLERR,"MPI is not supported for this pair_style");
	}
	int irequest = neighbor->request((void *) this);
	neighbor->requests[irequest]->pair = 1;
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->full = 1;
}

//////////////////////////////////////////////////////////////////////////////////////
//   init for one type pair i,j and corresponding j,i
//////////////////////////////////////////////////////////////////////////////////////
double PairNNMP::init_one(int i, int j)
{
	return maxCutoffRadius;
}

//////////////////////////////////////////////////////////////////////////////////////
//   proc 0 writes to restart file
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::write_restart(FILE *fp)
{
	return;
}

//////////////////////////////////////////////////////////////////////////////////////
//   proc 0 reads from restart file, bcasts
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::read_restart(FILE *fp)
{
	return;
}

//////////////////////////////////////////////////////////////////////////////////////
//   proc 0 writes to restart file
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::write_restart_settings(FILE *fp)
{
	return;
}

//////////////////////////////////////////////////////////////////////////////////////
//   proc 0 reads from restart file, bcasts
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::read_restart_settings(FILE *fp)
{
	return;
}

//////////////////////////////////////////////////////////////////////////////////////
//   allocate all arrays
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::allocate()
{
	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag,n+1,n+1,"pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;
	memory->create(cutsq,n+1,n+1,"pair:cutsq");
}
//////////////////////////////////////////////////////////////////////////////////////
// compute box
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::computeBox()
{
	box[0][0] = domain->h[0] * cflength;
	box[0][1] = 0.0;
	box[0][2] = 0.0;
	box[1][0] = domain->h[5] * cflength;
	box[1][1] = domain->h[1] * cflength;
	box[1][2] = 0.0;
	box[2][0] = domain->h[4] * cflength;
	box[2][1] = domain->h[3] * cflength;
	box[2][2] = domain->h[2] * cflength;
	pbc[0]=domain->xperiodic;
	pbc[1]=domain->yperiodic;
	pbc[2]=domain->zperiodic;
}
//////////////////////////////////////////////////////////////////////////////////////
// compute real atoms to store tags, types and positions
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::computeInverseTagPosType()
{
	vector<double> v(3,0.0);
	pos = vector< vector<double> >(list->inum,v);
	tag2i = vector< int >(list->inum);
	tag2type = vector< int >(list->inum);
	for (int ii = 0; ii < list->inum; ++ii) {
		int i = list->ilist[ii];
		int ti=atom->tag[i]-1;
		tag2i[ti] = i;
		tag2type[ti] = atom->type[i];
		for(int c=0;c<3;c++)
			pos[ti][c]=atom->x[i][c];
	}
}
//////////////////////////////////////////////////////////////////////////////////////
// compute idx_i, idx_j, offsets
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::computeIndex()
{
	double rc2 = maxCutoffRadius * maxCutoffRadius;// maxCutoffRadius& rc2 in lammps unit
	if(list==NULL)
		return;
	//cout<<"maxCutoffRadius="<<maxCutoffRadius<<endl;
	if(pos.size()!= atom->nlocal) computeInverseTagPosType();
	idx_i.clear();
	idx_j.clear();
	offsets.clear();
	for (int ii = 0; ii < list->inum; ++ii) {
		int i = list->ilist[ii];
		int ti=atom->tag[i]-1;
		//cout<<"i="<<i<<" tag i = "<<ti<<endl;
		for (int jj = 0; jj < list->numneigh[i]; ++jj) {
			int j = list->firstneigh[i][jj];
			j &= NEIGHMASK;
			double dx = atom->x[i][0] - atom->x[j][0];
			double dy = atom->x[i][1] - atom->x[j][1];
			double dz = atom->x[i][2] - atom->x[j][2];
			double d2 = dx * dx + dy * dy + dz * dz;
			if (d2 <= rc2)
			{
				// atom->x : Atom positions, including ghost atoms
				// offsets
				int tj=atom->tag[j]-1;
				dx = atom->x[j][0]-pos[tj][0];
				dy = atom->x[j][1]-pos[tj][1];
				dz = atom->x[j][2]-pos[tj][2];
				dx *= cflength;
				dy *= cflength;
				dz *= cflength;
				
				idx_i.push_back(ti);
				idx_j.push_back(tj);
				vector<double> v={dx,dy,dz};
				offsets.push_back(v);
			}
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::setPyData()
{
	int i;
	if (pSetData && PyCallable_Check(pSetData))
	{
		int nAtoms= pos.size();
		int idxSize=idx_i.size();
		PyObject *plmodels = PyUnicode_FromString(listmodels.c_str());
		PyObject *pnatoms = PyLong_FromLongLong((long long) nAtoms);
		double *R[]={(double*) malloc(nAtoms*sizeof(double)), (double*) malloc(nAtoms*sizeof(double)),(double*) malloc(nAtoms*sizeof(double))};
		char** symbols = (char**) malloc(nAtoms*sizeof(char*));
		PyObject* pR[] = {NULL, NULL, NULL};
		PyObject* pSymbols = NULL;
		int *idxi = (int*) malloc(idxSize*sizeof(int));
		int *idxj = (int*) malloc(idxSize*sizeof(int));
		double *offsetsV[] = {(double*) malloc(idxSize*sizeof(double)), (double*) malloc(idxSize*sizeof(double)), (double*) malloc(idxSize*sizeof(double))};
		PyObject* poffsetsV[] = {NULL, NULL, NULL};
		PyObject* pidxi = NULL;
		PyObject* pidxj = NULL;

		for(i=0;i<nAtoms;i++) 
		{
			int type = tag2type[i]-1;
			if(type<0 || type>elements.size()-1)
			{
				error->all(FLERR,
					"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
					"Problem with elements.\n"
					"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
					);
			}
			symbols[i] = strdup(elements[type].c_str());
		}
		pSymbols = vectorToList_Str(symbols, nAtoms);

		for(int c=0;c<3;c++)
		{
			for(i=0;i<nAtoms;i++) R[c][i] = pos[i][c] * cflength;
			pR[c] = vectorToList_Float(R[c], nAtoms);
		}
		for(i=0;i<idxSize;i++) idxi[i] = idx_i[i];
		pidxi = vectorToList_Int(idxi, idxSize);
		for(i=0;i<idxSize;i++) idxj[i] = idx_j[i];
		pidxj = vectorToList_Int(idxj, idxSize);

		for(int c=0;c<3;c++)
		{
			for(i=0;i<idxSize;i++) offsetsV[c][i] = offsets[i][c]*cflength;
			poffsetsV[c] = vectorToList_Float(offsetsV[c], idxSize);
		}

		PyObject* pBox[3];
		for(int c=0;c<3;c++)
			pBox[c] =vectorToList_Float(box[c], 3);
		double cutoff = maxCutoffRadius * cflength;
		PyObject* pCutOff =vectorToList_Float(&cutoff, 1);
		PyObject* pPBC =vectorToList_Int(pbc, 3);

		PyObject *pArgs = PyTuple_New(16);
                PyTuple_SetItem(pArgs, 0, plmodels);
                PyTuple_SetItem(pArgs, 1, pnatoms);
                PyTuple_SetItem(pArgs, 2, pSymbols);
                PyTuple_SetItem(pArgs, 3, pR[0]);
                PyTuple_SetItem(pArgs, 4, pR[1]);
                PyTuple_SetItem(pArgs, 5, pR[2]);
                PyTuple_SetItem(pArgs, 6, pBox[0]);
                PyTuple_SetItem(pArgs, 7, pBox[1]);
                PyTuple_SetItem(pArgs, 8, pBox[2]);
                PyTuple_SetItem(pArgs, 9, pPBC);
                PyTuple_SetItem(pArgs, 10, pCutOff);
                PyTuple_SetItem(pArgs, 11, pidxi);
                PyTuple_SetItem(pArgs, 12, pidxj);
                PyTuple_SetItem(pArgs, 13, poffsetsV[0]);
                PyTuple_SetItem(pArgs, 14, poffsetsV[1]);
                PyTuple_SetItem(pArgs, 15, poffsetsV[2]);
		pData = PyObject_CallObject(pSetData, pArgs);

		/*
		if(pSymbols) Py_DECREF(pSymbols);
		for(int c=0;c<3;c++)
			if(pR[c]) Py_DECREF(pR[c]);
		for(int c=0;c<3;c++)
			if(poffsetsV[c]) Py_DECREF(poffsetsV[c]);
		for(int c=0;c<3;c++)
			if(pBox[c]) Py_DECREF(pBox[c]);
		if(pPBC) Py_DECREF(pPBC);
		if(pidxi) Py_DECREF(pidxi);
		if(pidxj) Py_DECREF(pidxj);
		*/

		for(i=0;i<nAtoms;i++) if(symbols[i]) free(symbols[i]);
		free(symbols);
		for(int c=0;c<3;c++)
			free(R[c]);
		for(int c=0;c<3;c++)
			free(offsetsV[c]);
		free(idxi);
		free(idxj);
        }

	initialized = 1;
}
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMP::computeEnergyAndForces()
{
	double energy;
	int i;
	if (pSetData && PyCallable_Check(pSetData))
	{
		int nAtoms= pos.size();
		int idxSize=idx_i.size();
		PyObject *plmodels = PyUnicode_FromString(listmodels.c_str());
		PyObject *pnatoms = PyLong_FromLongLong((long long) nAtoms);
		double *R[]={(double*) malloc(nAtoms*sizeof(double)), (double*) malloc(nAtoms*sizeof(double)),(double*) malloc(nAtoms*sizeof(double))};
		char** symbols = (char**) malloc(nAtoms*sizeof(char*));
		PyObject* pR[] = {NULL, NULL, NULL};
		PyObject* pSymbols = NULL;
		int *idxi = (int*) malloc(idxSize*sizeof(int));
		int *idxj = (int*) malloc(idxSize*sizeof(int));
		double *offsetsV[] = {(double*) malloc(idxSize*sizeof(double)), (double*) malloc(idxSize*sizeof(double)), (double*) malloc(idxSize*sizeof(double))};
		PyObject* poffsetsV[] = {NULL, NULL, NULL};
		PyObject* pidxi = NULL;
		PyObject* pidxj = NULL;

		for(i=0;i<nAtoms;i++) 
		{
			int type = tag2type[i]-1;
			if(type<0 || type>elements.size()-1)
			{
				error->all(FLERR,
					"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
					"Problem with elements.\n"
					"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
					);
			}
			symbols[i] = strdup(elements[type].c_str());
		}
		pSymbols = vectorToList_Str(symbols, nAtoms);

		for(int c=0;c<3;c++)
		{
			for(i=0;i<nAtoms;i++) R[c][i] = pos[i][c] * cflength;
			pR[c] = vectorToList_Float(R[c], nAtoms);
		}
		for(i=0;i<idxSize;i++) idxi[i] = idx_i[i];
		pidxi = vectorToList_Int(idxi, idxSize);
		for(i=0;i<idxSize;i++) idxj[i] = idx_j[i];
		pidxj = vectorToList_Int(idxj, idxSize);

		for(int c=0;c<3;c++)
		{
			for(i=0;i<idxSize;i++) offsetsV[c][i] = offsets[i][c]*cflength;
			poffsetsV[c] = vectorToList_Float(offsetsV[c], idxSize);
		}

		PyObject* pBox[3];
		for(int c=0;c<3;c++)
			pBox[c] =vectorToList_Float(box[c], 3);
		double cutoff = maxCutoffRadius * cflength;
		PyObject* pCutOff =vectorToList_Float(&cutoff, 1);
		PyObject* pPBC =vectorToList_Int(pbc, 3);

		PyObject *pArgs = PyTuple_New(16);
            PyTuple_SetItem(pArgs, 0, plmodels);
            PyTuple_SetItem(pArgs, 1, pnatoms);
            PyTuple_SetItem(pArgs, 2, pSymbols);
            PyTuple_SetItem(pArgs, 3, pR[0]);
            PyTuple_SetItem(pArgs, 4, pR[1]);
            PyTuple_SetItem(pArgs, 5, pR[2]);
            PyTuple_SetItem(pArgs, 6, pBox[0]);
            PyTuple_SetItem(pArgs, 7, pBox[1]);
            PyTuple_SetItem(pArgs, 8, pBox[2]);
            PyTuple_SetItem(pArgs, 9, pPBC);
            PyTuple_SetItem(pArgs, 10, pCutOff);
            PyTuple_SetItem(pArgs, 11, pidxi);
            PyTuple_SetItem(pArgs, 12, pidxj);
            PyTuple_SetItem(pArgs, 13, poffsetsV[0]);
            PyTuple_SetItem(pArgs, 14, poffsetsV[1]);
            PyTuple_SetItem(pArgs, 15, poffsetsV[2]);

		PyObject* listRes = PyObject_CallObject(pComputeEnergyAndForces, pArgs);

            	if (listRes != NULL)
		{
			energy = PyFloat_AsDouble(PyList_GetItem(listRes, 0));
			energy /= cfenergy;
			int nAtoms =pos.size();
			double const cfforce = cfenergy / cflength;
			int cc=0;
			int j=1;
			for(int i=0;i<nAtoms;i++)
			{
				cc=tag2i[i];
				for(int k=0;k<3;k++) 
				{
					double f = PyFloat_AsDouble(PyList_GetItem(listRes, j));
					atom->f[cc][k] = f/ cfforce;
					j++;
				}
			}
		}
		else error->all(FLERR,"listRed of pComputeEnergyAndForces = NULL in pComputeEnergyAndForces");
		if (eflag_global)
			ev_tally(0,0,atom->nlocal,1,energy,0.0,0.0,0.0,0.0,0.0);

		if(listRes) Py_DECREF(listRes);
		if(pArgs) Py_DECREF(pArgs);
		/*
		if(pSymbols) Py_DECREF(pSymbols);
	
		for(int c=0;c<3;c++)
			if(pR[c]) Py_DECREF(pR[c]);
		for(int c=0;c<3;c++)
			if(poffsetsV[c]) Py_DECREF(poffsetsV[c]);
		for(int c=0;c<3;c++)
			if(pBox[c]) Py_DECREF(pBox[c]);
		if(pPBC) Py_DECREF(pPBC);
		if(pidxi) Py_DECREF(pidxi);
		if(pidxj) Py_DECREF(pidxj);
		*/
		for(i=0;i<nAtoms;i++) if(symbols[i]) free(symbols[i]);
		free(symbols);
		for(int c=0;c<3;c++)
			free(R[c]);
		for(int c=0;c<3;c++)
			free(offsetsV[c]);
		free(idxi);
		free(idxj);
        }
}
