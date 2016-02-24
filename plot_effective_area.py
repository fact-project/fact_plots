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

# c_cycle = cycler('color', ['red', 'b', 'black', 'g', 'yellow', 'orange','darkgrey', 'c', 'm'])
fig = plt.figure()
for datafile in datafiles:
    print("loading data file")
    data_df = pd.read_hdf(datafile, tablename)

    ax = plt.subplot(1,1,1)
    # ax.set_prop_cycle(c_cycle)

    logger.info("{} events in File".format(len(data_df)))
    val = os.path.basename(datafile).split(pattern[0])[-1].split(pattern[1])[0]
    label= "{} {} {}".format(feature, val, unit)

    if cuts:
        query = "numIslands < 6"
        print(query)
        data_df = data_df.query(query)

    e_mc = data_df["MCorsikaEvtHeader.fTotalEnergy"]
    plot_eff_area(ax, e_mc, label=label )
    ax.set_xlabel("$log_{10}$($E_{True}$ / $\si{\GeV}$)")
    ax.set_ylabel("$log_{10}$($A_{eff}$ / $\si{\meter}^2$)")
    ax.set_ylim([2e2,2e5])

plt.legend(loc = 'lower right')
# plt.show()
fig.savefig(outputfile)
