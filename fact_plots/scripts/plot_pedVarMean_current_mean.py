#!/usr/bin/env python2
"""Plot effective area for a dataset from HDF5 files

Usage:
    plot_numisland_cureent_mean.py <outputfile> <datafiles>... [options]

Options:
    --tablename=<name>      [default: table]
    --password=<name>       password for the factdb
    --cuts <cuts>           cuts for the pandas data frame as comma separated list
    --default_cuts <cuts>   choose predefined default cuts as comma separted list e.g. qualitycuts, precuts
    --feature=<name>        feature name of these comparisons [default: crosstalk]
    --unit=<name>           unit of feature these comparisons [default: %]
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

logger  = logging.getLogger(__name__)
args = docopt(__doc__)

def gauss(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*(x-mu)**2/sigma**2)

def combine_data_to_db(db_df, data_df):
    df = data_df.rename(columns={'RUNID': 'fRunID', 'NIGHT': 'fNight'})
    df = pd.merge(df, db_df, on=['fNight', 'fRunID'])
    logger.debug("{} events after merge with db".format(len(df)))
    return df

def mean_data_binned(df_for_bin, df_for_mean, bin_width, min_val=None, max_val=None):
    logger.debug("Feature for binning {}".format(df_for_bin.describe()))
    logger.debug("Feature for mean {}".format(df_for_mean.describe()))

    if min_val is None:
        min_val = np.min(df_for_bin)
    if max_val is None:
        max_val = np.max(df_for_bin)

    nBins = int((max_val - min_val)/bin_width)

    means = np.empty(nBins)
    means[:] = np.nan

    means_err = np.empty(nBins)
    means_err[:] = np.nan

    std = np.empty(nBins)
    std[:] = np.nan

    std_err = np.empty(nBins)
    std_err[:] = np.nan

    bin_middles = np.empty(nBins)
    bin_middles[:] = np.nan


    logger.debug("Bin width={}".format(bin_width))

    for i in range(nBins):
        bin_middles[i] = bin_width * (i+0.5) + min_val
        bin_min = bin_width*(i+1) + min_val
        bin_max = bin_width*i + min_val
        mask1 = df_for_bin < bin_min
        mask2 = df_for_bin >= bin_max
        mask = np.logical_and(mask1, mask2)

        logger.debug("{} entries, range=({},{}) for bin at{}".format(np.sum(mask), bin_min, bin_max, bin_middles[i]))

        if np.sum(mask) <= 1:
            logger.debug("No entries for bin")
            continue

        data_for_mean = df_for_mean[mask]

        # fit_entries, fit_edges = np.histogram(data_for_mean,
        #                                       bins=30,
        #                                       range=[np.min(data_for_mean), np.max(data_for_mean)],
        #                                       density=True,
        #                                       )
        # fit_middles = 0.5*(fit_edges[1:] + fit_edges[:-1])
        #
        # try:
        #     params, cov = curve_fit(gauss, fit_middles, fit_entries)
        # except RuntimeError:
        #     logger.debug("runtime exceeded")
        #     continue
        #
        # sigma, f_sigma  = params[1], np.sqrt(cov[1,1])
        # mean, f_mean    = params[0], np.sqrt(cov[0,0])
        #
        # if f_sigma > 1 or f_mean > 1:
        #     logger.debug("Fit failed")
        #     continue

        means[i] = np.mean(data_for_mean)
        means_err[i] = np.std(data_for_mean)#/len(data_for_mean)
        # std[i]        = sigma
        # std_err[i]    = f_sigma
        # means[i]        = mean
        # means_err[i]    = f_mean

    return_dict = dict(
        std         = np.array(std),
        std_err     = np.array(std_err),
        means       = np.array(means),
        means_err   = np.array(means_err),
        bin_middles = np.array(bin_middles),
        bin_width   = bin_width
    )

    return return_dict


logging.captureWarnings(True)
logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +  '%(message)s'), level=logging.DEBUG)


datafiles   = args["<datafiles>"]
outputfile  = args["<outputfile>"]

tablename   = args["--tablename"]
password    = args["--password"]

cuts            = args["--cuts"]
default_cuts    = args["--default_cuts"]
feature         = args["--feature"]
unit            = args["--unit"]
pattern         = args["--pattern"].split(",")

logger.info("loading Data Base")
factdb = create_engine("mysql+pymysql://factread:{}@129.194.168.95/factdata".format(password))
rundb = pd.read_sql("SELECT * from RunInfo WHERE (fNight > 20140614 AND fNight < 20140629)", factdb)

print(rundb["fCurrentsMedMean"].describe())

logger.info("loading Files")
df_list = []
for datafile in datafiles:
    logger.info("loading: {}".format(datafile))
    df = pd.read_hdf(datafile, tablename)
    logger.debug("{} Events in file".format(len(df)))
    logger.info("merging database")
    df = combine_data_to_db(rundb, df)
    # df = df.query("Size > 60")
    df_list.append(df)

fig = plt.figure()
ax = plt.subplot(1,1,1)

logger.info("binning data")
for df in df_list:
    binned = mean_data_binned(df["fCurrentsMedMean"], df["numIslands"], bin_width=1)
    ax.errorbar(binned["bin_middles"],
                binned["means"],
                xerr=binned["bin_width"],
                yerr=binned["means_err"],
                fmt="o")


ax.set_xlabel("Mean Current in pixels / $\si{\micro A}$")
ax.set_ylabel("Mean number of islands")


fig.savefig(outputfile)
