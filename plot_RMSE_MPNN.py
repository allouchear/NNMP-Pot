#!/home/csanz/bin/miniconda3/bin/python


import numpy as np
import pandas as pd
import os
import glob
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as tck


def getArguments():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--save', '-s',
                        help    = 'Save the figure with this file name and extension (jpg, png, pdf or svg).\nExample: `xplotRMS -x 10 -X 100 -s plot.pdf`',
                        default = plot_name,
                        type    = str)
    parser.add_argument('--train_unit', '-ta',
                        help    = 'plot in u.a instead of meV/A',
                        #default = False,
                        action  = "store_true")
    parser.add_argument('--dirname', '-d',
                        help    = 'name of checkpoint directory',
                        default = "CheckpointDir",
                        type    = str)
    args = parser.parse_args()
    return args


now = datetime.now()
plot_name = now.strftime("plot_RMSE_%Y-%m-%d.jpg")
args = getArguments()
Dir=args.dirname


listfE1=glob.glob(f'{Dir}/metrics/validation_*energies.txt')
listfE2=glob.glob(f"{Dir}/metrics/validation_*forces.txt")
RMSE_E=[]
RMSE_F=[]
for fE1, fE2 in zip(listfE1,listfE2):
    colnames=["Eref","Epred"]
    data_E = pd.read_table(fE1, comment='#', sep=" ",
                          engine="python", names=colnames, header=None)
    colnames=["Fref","Fpred"]
    data_F = pd.read_table(fE2, comment='#', sep=" ",
                          engine="python", names=colnames, header=None)

    if args.train_unit:
        RMSE_E.append(np.sqrt(np.sum((data_E.Eref - data_E.Epred)**2)/len(data_E.Eref)))
        RMSE_F.append(np.sqrt(np.sum((data_F.Fref - data_F.Fpred)**2)/len(data_F.Fref)))
    else :
        data_E.Eref *= 1000.0*27.21138469 #: a.u to  meV/atom & 1000.0*0.0434 : kcal/mol to meV/atom
        data_E.Epred  *= 1000.0*27.21138469 # meV/Å
        data_F.Fref *= 1000.0*27.21138469/0.529177 # meV/Å
        data_F.Fpred  *= 1000.0*27.21138469/0.529177 # meV/Å
        RMSE_E.append(np.sqrt(np.sum((data_E.Eref - data_E.Epred)**2)/len(data_E.Eref)))
        RMSE_F.append(np.sqrt(np.sum((data_F.Fref - data_F.Fpred)**2)/len(data_F.Fref)))

if len(listfE1)>0:
    latest_file = max(listfE1, key=os.path.getctime)
    latestE = latest_file.split("n_")[1]
if len(listfE2)>0:
    latest_file = max(listfE2, key=os.path.getctime)
    latestF = latest_file.split("n_")[1]

epoch=np.arange(1000,(len(listfE1)+1)*1000,1000)

training = os.path.basename(os.getcwd())
nnp = os.path.basename(os.path.dirname(os.getcwd()))


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), tight_layout=True)
fig.suptitle(f"{nnp} / {training}", fontweight="bold")
ax[0][0].set_title('Forces RMSE', fontweight="bold")
ax[0][0].set_xlabel('Epoch')
if args.train_unit:
    ax[0][0].set_ylabel('Force RMSE [u.a]')
else:
    ax[0][0].set_ylabel('Force RMSE [meV/Å]')
ax[0][0].grid()
l1 = ax[0][0].plot(epoch, RMSE_F,
              label='validation set', alpha=0.9)
ax[0][0].legend(framealpha=1)
ax[0][0].set_xlim(left=0)
ax[0][0].set_ylim(bottom=0)
#ax[0][0].set_ylim(bottom=0, top=500) chloe : si besoin ajustement

ax[1][0].set_title('Energies RMSE', fontweight="bold")
ax[1][0].set_xlabel('Epoch')
if args.train_unit:
    ax[1][0].set_ylabel('Energies RMSE [u.a]')
else :
    ax[1][0].set_ylabel('Energies RMSE [meV/atom]')
ax[1][0].yaxis.set_minor_locator(tck.AutoMinorLocator(5))  #chloe : set minor ticks
ax[1][0].grid()
l1 = ax[1][0].plot(epoch, RMSE_E,
              label='validation set', alpha=0.9)
ax[1][0].legend(framealpha=1)
ax[1][0].set_xlim(left=0)
ax[1][0].set_ylim(bottom=0) #original
#ax[1][0].set_ylim(bottom=0,top=50)  #chloe : modif pour meilleur echelle

ax[0][1].set_title(f'Forces (step {latestF})', fontweight="bold")
if args.train_unit:
    ax[0][1].set_xlabel('F$_{ref}$ [u.a]')
    ax[0][1].set_ylabel('F$_{nnp}$ [u.a]')
else:
    ax[0][1].set_xlabel(r'F$_{ref}$ [meV/Å]')
    ax[0][1].set_ylabel(r'F$_{nnp}$ [meV/Å]')

ax[0][1].grid()
ax[0][1].axline((0, 0), slope=1, color='black', linewidth=.7)
"""
if args.train_unit :
    ax[0][1].scatter(data_F.Fref, data_F.Fpred, alpha=0.5, edgecolors='none', label="validation set")
else :
    ax[0][1].scatter(data_F.Fref*1000.0*27.21138469/0.529177, data_F.Fpred*1000.0*27.21138469/0.529177, alpha=0.5, edgecolors='none', label="validation set")
"""
ax[0][1].scatter(data_F.Fref, data_F.Fpred, alpha=0.5, edgecolors='none', label="validation set")
ax[0][1].legend(framealpha=1)

ax[1][1].set_title(f'Energies (step {latestE})', fontweight="bold")

if args.train_unit:
    ax[1][1].set_xlabel('E$_{ref}$ [u.a]')
    ax[1][1].set_ylabel('E$_{nnp}$ [u.a]')
else :
    ax[1][1].set_xlabel('E$_{ref}$ [meV/atom]')
    ax[1][1].set_ylabel('E$_{nnp}$ [meV/atom]')

ax[1][1].grid()
ax[1][1].axline((0, 0), slope=1, color='black', linewidth=.7)
"""
if args.train_unit:
    ax[1][1].scatter(data_E.Eref, data_E.Epred, alpha=0.5, edgecolors='none', label="validation set")
else :
    ax[1][1].scatter(data_E.Eref*1000.0*27.21138469, data_E.Epred*1000.0*27.21138469, alpha=0.5, edgecolors='none', label="validation set")
"""
ax[1][1].scatter(data_E.Eref, data_E.Epred, alpha=0.5, edgecolors='none', label="validation set")
ax[1][1].legend(framealpha=1)

plt.savefig(args.save, transparent=True, dpi=300)
plt.show()

