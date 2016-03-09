#!/usr/bin/env python2
"""Plot effective area for a dataset from HDF5 files. e.g.

TEXINPUTS=$(pwd): python plot_numisland_current_mean.py /home/jbuss/plots/numIslandsCureents.pdf /fhgfs/users/jbuss/20140615_27_cStd.hdf /fhgfs/users/jbuss/20140615_27_c6_4.hdf /fhgfs/users/jbuss/20140615_27_c7_5.hdf --password r3adfac! --pattern "_c,.hdf" --unit="p.e." --feature="Level:"

Usage:
    plot_ped_var_nsb_rate.py <outputfile> <datafiles>... [options]

Options:
    --tablename=<name>      [default: table]
    --password=<name>       password for the factdb
    --cuts <cuts>           cuts for the pandas data frame as comma separated list
    --default_cuts <cuts>   choose predefined default cuts as comma separted list e.g. qualitycuts, precuts
    --feature=<name>        feature name of these comparisons [default: ped_std_mean]
    --unit=<name>           unit of feature these comparisons [default: $\mathrm{p.e.}$]
    --pattern <name>        pattern of the feature value string e.g "_xT,_c" [default: "_nsb,_c"]
"""
from sqlalchemy import create_engine
import pandas as pd
from docopt import docopt
import numpy as np
import matplotlib
import datetime
import matplotlib.pyplot as plt
import logging
from scipy.stats import moment
from scipy.optimize import curve_fit
from IPython import embed
import os

logger  = logging.getLogger(__name__)
args = docopt(__doc__)

def gauss(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*(x-mu)**2/sigma**2)

def f_sqrt(x, a, b):
    return a * np.sqrt(x) + b

def f_lin(x, a, b):
    return a * x + b

def buildLabel(path, pattern):
    if not pattern:
        return pattern
    pattern = pattern.split(",")
    label_val = os.path.basename(datafile).split(pattern[0])[-1].split(pattern[1])[0]
    return label_val

logging.captureWarnings(True)
logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +  '%(message)s'), level=logging.INFO)


datafiles   = args["<datafiles>"]
outputfile  = args["<outputfile>"]

tablename   = args["--tablename"]
password    = args["--password"]

cuts            = args["--cuts"]
default_cuts    = args["--default_cuts"]
feature_name    = args["--feature"]
unit            = args["--unit"]

pattern         = args["--pattern"]


logger.info("loading Files")
df_list = []
labels = []

ped_var_means  = []
ped_vars_vars  = []
ped_vars_stds  = []
ped_vars_sizes = []
ped_vars_sems  = []

for datafile in datafiles:
    logger.info("loading: {}".format(datafile))
    df = pd.read_hdf(datafile, tablename)
    logger.debug("{} Events in file".format(len(df)))
    # df = df.query("Size > 60")
    df_list.append(df)
    ped_var_means.append(df[feature_name].mean())
    ped_vars_vars.append(df[feature_name].var())
    ped_vars_stds.append(df[feature_name].std())
    # ped_vars_sizes.append(df[feature_name].size())
    ped_vars_sems.append(df[feature_name].sem())
    label = buildLabel (datafile, pattern)
    nsb_rate = int(label)
    labels.append(nsb_rate)

ped_var_means = np.array(ped_var_means)
ped_vars_vars = np.array(ped_vars_vars)
ped_vars_stds = np.array(ped_vars_stds)
ped_vars_sizes = np.array(ped_vars_sizes)
ped_vars_sems = np.array(ped_vars_sems)
nsb_rate = np.array(labels)

fig = plt.figure()
ax = plt.subplot(1,1,1)

gain = 257.

params, cov = curve_fit(f_lin, nsb_rate, ped_var_means/gain**2)
x_plot = np.linspace(0, 300, 1000)
print(params)

ax.errorbar(nsb_rate,
            ped_var_means/gain**2,
            xerr=0,
            # yerr=binned[feature_name+"_std"].values/binned[feature_name+"_size"].values,
            yerr=ped_vars_stds/gain**2,
            fmt=".",
            capsize=1,
            label="simulated pedestal"
            )
ax.plot(x_plot, f_lin(x_plot, *params), 'b-', linewidth=0.8, label="linear fit")
ax.text(210, 4.1, "$f(x)={:.2f} \cdot x + {:.2f}".format(*params),
            fontsize=12, color='b')

ax.set_xlabel("Simulated NSB rate / $\si{\mega \hertz}$")
ax.set_ylabel("Mean pedestal variance / $\mathrm{p.e.^2}")
ax.legend(loc="upper left")
logger.info("saving image data: {}".format(outputfile))
fig.savefig(outputfile)



#pdftoppm -png -r 600 test.pdf > test.png
