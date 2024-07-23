#include <mpi.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "pair_nnmp_external.h"
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

//////////////////////////////////////////////////////////////////////////////////////
PairNNMPExternal::PairNNMPExternal(LAMMPS *lmp) : Pair(lmp)
{

}
//////////////////////////////////////////////////////////////////////////////////////
PairNNMPExternal::~PairNNMPExternal()
{

}
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMPExternal::compute(int eflag, int vflag)
{
	if(eflag || vflag) ev_setup(eflag,vflag);
	else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;
	string WDir=string(directory);
	if (comm->nprocs > 1)
		WDir=string(directory)+ "_"+ std::to_string(comm->me);
	string com = "mkdir -p "+ WDir;
	system(com.c_str());
	computeInverseTagPosType();
	writeBox(WDir);
	writeAtoms(WDir);
	writeNeighborList(WDir);

	com = "cd " + WDir + "; rm energy.out";
	system(com.c_str());
	com = "cd " + WDir + "; rm forces.out";
	system(com.c_str());
	// Run external command and throw away stdout output.
	//com = "cd " + WDir + "; " + string(command) + " > external.out";
	com = "cd " + WDir + "; " + string(command);
	system(com.c_str());

	// Read back in total potential energy.
	ifstream f;
	f.open(WDir + "energy.out");
	if (!f.is_open()) error->all(FLERR,"Could not open energy output file");
	string line;
	double energy = 0.0;
	f>>energy;
	f.close();
	energy /= cfenergy;
	//cout<<"energy="<<energy<<endl;

	// Add energy contribution to total energy.
	if (eflag_global)
		ev_tally(0,0,atom->nlocal,1,energy,0.0,0.0,0.0,0.0,0.0);

	// Read forces.
	f.open(WDir + "forces.out");
	if (!f.is_open()) error->all(FLERR,"Could not open forces output file");
	int c = 0;
	double const cfforce = cfenergy / cflength;
	while (getline(f, line))
	{
		if ((line.size() > 0) && (line.at(0) != '#'))
		{
			if (c > atom->nlocal - 1) error->all(FLERR,"Too many atoms in force file.");
			double fx;
			double fy;
			double fz;
			sscanf(line.c_str(), "%le %le %le", &fx, &fy, &fz);
			int cc=tag2i[c];
			atom->f[cc][0] = fx / cfforce;
			atom->f[cc][1] = fy / cfforce;
			atom->f[cc][2] = fz / cfforce;
			//cout<<"force="<<fx<<" "<<fy<<" "<<fz<<endl;
			c++;
		}
	}
	f.close();
	if(atom->nlocal!=c)
	{
		error->all(FLERR,
				"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				"# of atoms <> number of forces\n"
				"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				);
	}

	// If virial needed calculate via F dot r.
	if (vflag_fdotr) 
		virial_fdotr_compute();
}
//////////////////////////////////////////////////////////////////////////////////////
//   global settings
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMPExternal::settings(int narg, char **arg)
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
	len = strlen("nnmp/") + 1;
	directory = new char[len];
	strcpy(directory,"nnmp/");
	string s("xeforces");
	len = s.size() + 1;
	command = new char[len];
	strcpy(command,s.c_str());
	cflength = 1.0;
	cfenergy = 1.0;

	while(iarg < narg) {
		// set NNMP directory
		if (strcmp(arg[iarg],"dir") == 0) {
			if (iarg+2 > narg)
				error->all(FLERR,"Illegal pair_style command : dir");
		delete[] directory;
		len = strlen(arg[iarg+1]) + 2;
		directory = new char[len];
		sprintf(directory, "%s/", arg[iarg+1]);
		iarg += 2;
		// set external prediction command
		} else if (strcmp(arg[iarg],"command") == 0) {
			if (iarg+2 > narg)
				error->all(FLERR,"Illegal pair_style command: command");
			delete[] command;
		len = strlen(arg[iarg+1]) + 1;
		command = new char[len];
		sprintf(command, "%s", arg[iarg+1]);
		iarg += 2;
		// length unit conversion factor
		} else if (strcmp(arg[iarg],"cflength") == 0) {
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
		fprintf(f, "elements = %s\n", selements.c_str());
		fprintf(f, "dir      = %s\n", directory);
		fprintf(f, "command  = %s\n", command);
		fprintf(f, "cflength = %16.8E\n", cflength);
		fprintf(f, "cfenergy = %16.8E\n", cfenergy);
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
}

//////////////////////////////////////////////////////////////////////////////////////
//   set coeffs for one or more type pairs
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMPExternal::coeff(int narg, char **arg)
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

void PairNNMPExternal::init_style()
{
	if (comm->nprocs > 1)
	{
		error->all(FLERR,"MPI is not supported for this pair_style");
	}
	neighbor->add_request(this, NeighConst::REQ_FULL);
	/*
	int irequest = neighbor->request((void *) this);
	neighbor->requests[irequest]->pair = 1;
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->full = 1;
	*/
}

//////////////////////////////////////////////////////////////////////////////////////
//   init for one type pair i,j and corresponding j,i
//////////////////////////////////////////////////////////////////////////////////////
double PairNNMPExternal::init_one(int i, int j)
{
	return maxCutoffRadius;
}

//////////////////////////////////////////////////////////////////////////////////////
//   proc 0 writes to restart file
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMPExternal::write_restart(FILE *fp)
{
	return;
}

//////////////////////////////////////////////////////////////////////////////////////
//   proc 0 reads from restart file, bcasts
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMPExternal::read_restart(FILE *fp)
{
	return;
}

//////////////////////////////////////////////////////////////////////////////////////
//   proc 0 writes to restart file
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMPExternal::write_restart_settings(FILE *fp)
{
	return;
}

//////////////////////////////////////////////////////////////////////////////////////
//   proc 0 reads from restart file, bcasts
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMPExternal::read_restart_settings(FILE *fp)
{
	return;
}

//////////////////////////////////////////////////////////////////////////////////////
//   allocate all arrays
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMPExternal::allocate()
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
// Save box
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMPExternal::writeBox(const string& WDir)
{
	string boxfile=WDir + "/box.data";
	ofstream f(boxfile);
	double box[3][3];
	box[0][0] = domain->h[0] * cflength;
	box[0][1] = 0.0;
	box[0][2] = 0.0;
	box[1][0] = domain->h[5] * cflength;
	box[1][1] = domain->h[1] * cflength;
	box[1][2] = 0.0;
	box[2][0] = domain->h[4] * cflength;
	box[2][1] = domain->h[3] * cflength;
	box[2][2] = domain->h[2] * cflength;
	f<<domain->xperiodic;
	f<<" "<<domain->yperiodic;
	f<<" "<<domain->zperiodic;
	f<<endl;
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
			f<<" "<<scientific<<setprecision(12)<<setw(20)<<box[i][j];
		f<<endl;
	}
	f.close();
}

//////////////////////////////////////////////////////////////////////////////////////
// Save structure
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMPExternal::writeAtoms(const string& WDir)
{
	string datafile=WDir + "/atoms.xyz";
	ofstream f(datafile);
	double r[3];
	int type;
	if(pos.size()!= atom->nlocal) computeInverseTagPosType();
	f<<pos.size()<<endl;
	f<<endl;
	for (size_t i = 0; i < pos.size(); ++i)
	{
    		r[0] = pos[i][0] * cflength;
    		r[1] = pos[i][1] * cflength;
    		r[2] = pos[i][2] * cflength;
		type = tag2type[i]-1;
		if(type<0 || type>elements.size()-1)
		{
			error->all(FLERR,
				"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				"Problem with elements.\n"
				"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				);
		}
		f<<setw(5)<<elements[type]<<" "
		<<scientific<<setprecision(12)<<setw(20)<<r[0]<<" "
		<<scientific<<setprecision(12)<<setw(20)<<r[1]<<" "
		<<scientific<<setprecision(12)<<setw(20)<<r[2]<<endl;
	}
	f.close();
}
//////////////////////////////////////////////////////////////////////////////////////
// Save NeighborList
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMPExternal::writeNeighborList(const string& WDir)
{
	double rc2 = maxCutoffRadius * maxCutoffRadius;// maxCutoffRadius& rc2 in lammps unit
	string neighfile=WDir + "/neigh.data";
	ofstream f(neighfile);
	f<<maxCutoffRadius * cflength<<endl;
	if(list==NULL)
	{
		f.close();
		return;
	}
	//cout<<"maxCutoffRadius="<<maxCutoffRadius<<endl;
	if(pos.size()!= atom->nlocal) computeInverseTagPosType();
#ifdef _OPENMP
  #pragma omp parallel for
#endif
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
				
				f<<setw(10)<<ti<<" "<<setw(10)<<tj<<" "
				<<scientific<<setprecision(12)<<setw(20)<<dx<<" "
				<<scientific<<setprecision(12)<<setw(20)<<dy<<" "
				<<scientific<<setprecision(12)<<setw(20)<<dz
				<<endl;
			}
		}
	}
	f.close();
}
//////////////////////////////////////////////////////////////////////////////////////
// compute real atoms to store tags, types and positions
//////////////////////////////////////////////////////////////////////////////////////
void PairNNMPExternal::computeInverseTagPosType()
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
