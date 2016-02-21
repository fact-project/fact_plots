#!/usr/bin/env python2
"""Plot effective area for a dataset from HDF5 files

Usage:
    plot_effective_area.py <datafile> [options]

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
import logging
import pandas as pd
import default_plots as dp
import default_cuts as dc
import gc

logger  = logging.getLogger(__name__)

def expE(E,x):
    return E**x

def integrateEnergySpectrum(e_min, e_max, gamma):
    upper = expE( e_max, gamma+1) / gamma+1
    lower = expE( e_min, gamma+1) / gamma+1
    return upper-lower

def calculate_a_eff_bin(n_trigger, max_impact, E_bin_low, e_bin_high, E_min, E_max, gamma):
    N_0 =   12e6
    N_0 =   1
    N_0 *=  integrateEnergySpectrum(E_bin_low, e_bin_high, gamma)
    N_0 /=  integrateEnergySpectrum(E_min, E_max, gamma)

    return n_trigger / N_0 * np.pi * max_impact**2

def calc_a_eff(true_energy, bins, min_e=None, max_e=None, max_impact=270, gamma=-2.7):
    e_mc = np.log10(true_energy)
    if min_e is None:
        min_e = np.min(e_mc)
    if max_e is None:
        max_e = np.max(e_mc)

    bin_width = (max_e - min_e)/bins
    logger.info("Number of bins: {}".format(bins))

    bin_middles = np.empty(bins)
    bin_middles[:] = np.nan

    A_eff = np.empty(bins)
    A_eff[:] = np.nan

    A_eff_err = np.empty(bins)
    A_eff_err[:] = np.nan

    n_trigger = np.empty(bins)
    n_trigger[:] = np.nan

    for i in range(bins):
        bin_low_edge  = bin_width*i + min_e
        bin_high_edge = bin_width*(i+1) + min_e

        mask1 = e_mc < bin_high_edge
        mask2 = e_mc >= bin_low_edge

        mask = np.logical_and(mask1, mask2)

        if np.sum(mask) <= 1:
            logger.info("no entries for given mask {} <= emc < {}".format( '%.2E' % bin_low_edge, '%.2E' % bin_high_edge))
            continue

        bin_middles[i] = bin_width * (i+0.5) + min_e

        energy_bin = e_mc[mask]
        #print(energy_bin)
        n_trigger[i] = np.sum(mask)

        A_eff[i] = calculate_a_eff_bin(n_trigger[i], max_impact,
                                        bin_low_edge, bin_high_edge,
                                        min_e, max_e, gamma)

        A_eff_err[i] = np.sqrt(n_trigger[i])

        logger.debug("Aeff={}m² with {} entries for given range {} <= emc < {}".format(
                                        '%.2E' % A_eff[i],
                                        '%.2E' % n_trigger[i],
                                        '%.2E' % bin_low_edge,
                                        '%.2E' % bin_high_edge))

    return_dict = dict(
        range       = np.array([min_e,max_e]),
        Aeff        = np.array(A_eff),
        Aeff_res    = np.array(A_eff_err),
        nTrigger    = np.array(n_trigger),
        bin_middles = np.array(bin_middles),
        bin_width   = np.array(bin_width)
    )

    return return_dict



args = docopt(__doc__)
logging.captureWarnings(True)
logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +  '%(message)s'), level=logging.DEBUG)

datafile    = args["<datafile>"]
#outputfile  = args["<outputfile>"]

tablename   = args["--tablename"]

cuts            = args["--cuts"]
default_cuts    = args["--default_cuts"]

print("loading data file")
data_df = pd.read_hdf(datafile, tablename)

e_mc = data_df["MCorsikaEvtHeader.fTotalEnergy"]

effectiveArea = calc_a_eff(e_mc, 20)

ax = plt.subplot(1,1,1)
plt.errorbar(effectiveArea["bin_middles"],
            effectiveArea["Aeff"],
            xerr=effectiveArea["bin_width"]/2,
            yerr=effectiveArea["Aeff_res"],
            fmt="o")

ax.set_yscale('log')
#ax.set_xscale('log')
plt.show()
