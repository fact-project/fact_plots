#!/usr/bin/env python2
"""Plot effective area for a dataset from HDF5 files

Usage:
    plot_effective_area.py <datafile> <outputfile> [options]

Options:
    --tablename=<name>      [default: table]
    --cuts <cuts>           cuts for the pandas data frame as comma separated list
    --default_cuts <cuts>   choose predefined default cuts as comma separted list e.g. qualitycuts, precuts
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
import pandas as pd
import default_plots as dp
import default_cuts as dc
import gc

def expE(E,x):
    return E**x

def integrateEnergySpectrum(e_min, e_max, gamma):
    upper = expE( e_max, gamma+1) / gamma+1
    lower = expE( e_min, gamma+1) / gamma+1
    return upper-lower

def calculate_a_eff_bin(n_trigger, max_impact, E_bin_low, e_bin_high, E_min, E_max, gamma):
    N_0 =   12e6
    N_0 *=  integrateEnergySpectrum(E_bin_low, e_bin_high, gamma)
    N_0 /=  integrateEnergySpectrum(E_min, E_max, gamma)

    return n_trigger/N_0 * np.pi() * max_impact**2



args = docopt(__doc__)

datafile    = args["<datafile>"]
outputfile  = args["<outputfile>"]

tablename   = args["--tablename"]

cuts            = args["--cuts"]
default_cuts    = args["--default_cuts"]

print("loading data file")
data_df = pd.read_hdf(datafile, tablename)
