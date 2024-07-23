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
    parser.add_argument('--dirname', '-d',
                        help    = 'name of checkpoint directory',
                        default = "CheckpointDir",
                        type    = str)
    args = parser.parse_args()
    return args


now = datetime.now()
plot_name = now.strftime("plot_mae_%Y-%m-%d.jpg")
args = getArguments()
Dir = args.dirname


fE1=f'{Dir}/metrics/mae.txt'

colnames=["MAE_E","MAE_F"]
data = pd.read_table(fE1, comment='#', sep="\s",
                      engine="python", names=colnames, header=None)

epoch=np.arange(1000,(len(data)+1)*1000,1000)

training = os.path.basename(os.getcwd())
nnp = os.path.basename(os.path.dirname(os.getcwd()))


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), tight_layout=True)
fig.suptitle(f"{nnp} / {training}", fontweight="bold")
ax[0].set_title('Forces MAE', fontweight="bold")
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Force MAE [training units]')
ax[0].grid()
l1 = ax[0].plot(epoch, data.MAE_F,
              label='valid set',color='dodgerblue', alpha=0.9)
l2=ax[0].scatter(epoch, data.MAE_F, s=10, c='tab:blue',alpha=0.9)
ax[0].legend(framealpha=1)
ax[0].set_xlim(left=0)
ax[0].set_ylim(bottom=0)
#ax[0][0].set_ylim(bottom=0, top=500) chloe : si besoin ajustement

ax[1].set_title('Energies MAE', fontweight="bold")
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Energies MAE [training units]')
ax[1].yaxis.set_minor_locator(tck.AutoMinorLocator(5))  #chloe : set minor ticks
ax[1].grid()
l1 = ax[1].plot(epoch, data.MAE_E, color='dodgerblue',
              label='valid set', alpha=0.9)
l2=ax[1].scatter(epoch, data.MAE_E,c='tab:blue',s=10, alpha=0.9)
ax[1].legend(framealpha=1)
ax[1].set_xlim(left=0)
ax[1].set_ylim(bottom=0) #original
#ax[1][0].set_ylim(bottom=0,top=50)  #chloe : modif pour meilleur echelle


plt.savefig(args.save, transparent=True, dpi=300)
plt.show()

