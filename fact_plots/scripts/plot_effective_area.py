#!/usr/bin/env python2
"""Plot effective area for a dataset from HDF5 files

Usage:
    plot_effective_area.py <outputfile> <datafiles>... [options]

Options:
    --tablename=<name>      [default: table]
    --cuts <cuts>           cuts for the pandas data frame as comma separated list
    --default_cuts <cuts>   choose predefined default cuts as comma separted list e.g. qualitycuts, precuts
    --feature=<name>        feature name of these comparisons [default: crosstalk]
    --unit=<name>           unit of feature these comparisons [default: %]
    --pattern <name>        pattern of the feature value string e.g "_xT,_c" [default: "_nsb,_c"]
"""
from __future__ import print_function
import numpy as np
import matplotlib
import datetime
# matplotlib.rc_file("./matplotlibrc")
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from docopt import docopt
import logging
import pandas as pd
import default_plots as dp
import default_cuts as dc
import gc
import os
from cycler import cycler
from effectiveArea import *

logger  = logging.getLogger(__name__)

args = docopt(__doc__)
logging.captureWarnings(True)
logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +  '%(message)s'), level=logging.DEBUG)




datafiles    = args["<datafiles>"]
outputfile  = args["<outputfile>"]

tablename   = args["--tablename"]

cuts            = args["--cuts"]
default_cuts    = args["--default_cuts"]
feature         = args["--feature"]
unit            = args["--unit"]
pattern            = args["--pattern"].split(",")

data_df_list = []

cuts = dc.cuts["ICRC2015_pre_Xtalk"]
cuts = " & ".join(cuts)

# c_cycle = cycler('color', ['red', 'b', 'black', 'g', 'yellow', 'orange','darkgrey', 'c', 'm'])
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
for datafile in datafiles:
    print("loading data file")
    data_df = pd.read_hdf(datafile, tablename)
    data_df = data_df.query(cuts)
    data_df_list.append(data_df)
    # ax.set_prop_cycle(c_cycle)

    logger.info("{} events in File".format(len(data_df)))
    val = os.path.basename(datafile).split(pattern[0])[-1].split(pattern[1])[0]
    label= "{} {} {}".format(feature, val, unit)

    if cuts:
        query = "numIslands < 6"
        print(query)
        data_df = data_df.query(query)

    e_mc = data_df["MCorsikaEvtHeader.fTotalEnergy"]
    effA_result = plot_eff_area(ax[0], e_mc, label=label )
    ax[0].set_xlabel("$\log_{10} (E_\mathrm{True} / \si{\GeV})$")
    ax[0].set_ylabel("$\log_{10} (A_\mathrm{eff} / \si{\meter}^2)$")
    # ax.set_ylim([2e2,2e5])

    plot_nTriggers(ax[1], effA_result, label=label )

fig.subplots_adjust(hspace=0.06)
ax[0].legend(loc = 'lower right')
ax[1].legend()

# plt.show()
fig.savefig(outputfile)


fig = plt.figure()
for data_df in data_df_list:
    val = os.path.basename(datafile).split(pattern[0])[-1].split(pattern[1])[0]
    label= "{} {} {}".format(feature, val, unit)
    hist, edges, patches = plt.hist(np.log10(data_df["MCorsikaEvtHeader.fTotalEnergy"]), log=True, bins=20, label=label, alpha=0.5, histtype="step")
    xy  = patches[0].get_xy()
    xy[:, 1][xy[:, 1] == 0] = 0.1
    patches[0].set_xy(xy)

fig.get_axes()[0].legend()
fig.savefig("E_True"+outputfile)
