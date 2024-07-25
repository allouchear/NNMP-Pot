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
plot_name = now.strftime("plot_loss_%Y-%m-%d.jpg")

training = os.path.basename(os.getcwd())
nnp = os.path.basename(os.path.dirname(os.getcwd()))

args = getArguments()
Dir=args.dirname
fE1=f'{Dir}/metrics/Loss.txt'


colnames=["loss"]
data = pd.read_table(fE1, comment='#',
                      engine="python", names=colnames, header=None)

epoch=np.arange(1000,(len(data.loss)+1)*1000,1000)



fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
fig.suptitle(f"{nnp} / {training}", fontweight="bold")
plt.title('Loss', fontweight="bold")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
l1 = plt.plot(epoch, data.loss, 
              label='loss', alpha=0.9)
l2 = plt.scatter(epoch, data.loss, s=10,
                c='tab:blue', alpha=0.9)
plt.legend(framealpha=1)
#plt.xlim(left=0)
#plt.ylim(bottom=0)
#ax[0][0].set_ylim(bottom=0, top=500) chloe : si besoin ajustement

plt.savefig(args.save, transparent=True, dpi=300)
plt.show()

