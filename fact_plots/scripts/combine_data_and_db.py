#!/usr/bin/env python2
"""Plot effective area for a dataset from HDF5 files

Usage:
    combine_data_to_db.py <datafile> [options]

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
    return pd.merge(df, db_df, on=['fNight', 'fRunID'])


def mean_data_binned(df_for_bin, df_for_mean, nBins, min_val=None, max_val=None):

    if min_val is None:
        min_val = np.min(df_for_bin)
    if max_val is None:
        max_val = np.max(df_for_bin)

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

    bin_width = (max_val - min_val)/nBins

    for i in range(nBins):
        bin_middles[i] = bin_width * (i+0.5) + min_val

        mask1 = df_for_bin < bin_width*(i+1) + min_val
        mask2 = df_for_bin >= bin_width*i + min_val

        mask = np.logical_and(mask1, mask2)

        if np.sum(mask) <= 1:
            logger.debug("No entries for bin at{}".format(bin_middles[i]))
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
        means_err[i] = np.std(data_for_mean)/len(data_for_mean)
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

def main():
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +  '%(message)s'), level=logging.DEBUG)


    datafile    = args["<datafile>"]
    # outputfile  = args["<outputfile>"]

    tablename   = args["--tablename"]
    password   = args["--password"]

    cuts            = args["--cuts"]
    default_cuts    = args["--default_cuts"]
    feature         = args["--feature"]
    unit            = args["--unit"]
    pattern            = args["--pattern"].split(",")

    logger.info("loading Data Base")
    factdb = create_engine("mysql+pymysql://factread:{}@129.194.168.95/factdata".format(password))
    rundb = pd.read_sql("SELECT * from RunInfo WHERE (fNight > 20140614 AND fNight < 20140629)", factdb)

    logger.info("loading File")
    data_df = pd.read_hdf(datafile, tablename)

    logger.info("merging dataframes")
    result = combine_data_to_db(rundb, data_df)
    result = result.query("Size > 60")

    print(result["fCurrentsMedMean"].describe())
    logger.info("binning data")
    binned = mean_data_binned(result["fCurrentsMedMean"], result["numIslands"],
                        nBins=50, max_val=30)

    fig = plt.figure()
    ax = plt.subplot(1,1,1)

    plt.errorbar(binned["bin_middles"],
                binned["means"],
                xerr=binned["bin_width"],
                yerr=binned["means_err"],
                fmt="o")
    ax.set_xlabel("mean current in SiPMs / $\si{\micro\ampere}$")
    ax.set_ylabel("mean number of islands")

    fig.savefig("numIslands.pdf")

    binned = mean_data_binned(result["fCurrentsMedMean"], result["ped_var_mean"],
                        nBins=50, max_val=30)

    fig = plt.figure()
    ax = plt.subplot(1,1,1)

    plt.errorbar(binned["bin_middles"],
                binned["means"],
                xerr=binned["bin_width"],
                yerr=binned["means_err"],
                fmt="o")
    ax.set_xlabel("mean current in SiPMs / $\si{\micro\ampere}$")
    ax.set_ylabel("mean of ped_var")


    fig.savefig("pedVar.pdf")

    # fig = plt.figure()
    # ax = plt.subplot(1,1,1)
    # plt.scatter(result["fCurrentsMedMean"], result["ped_var_mean"])
    # fig.savefig("pedVar_scat.pdf")

    binned = mean_data_binned(result["fCurrentsMedMean"], result["ped_sum_mean"],
                        nBins=50, max_val=30)

    fig = plt.figure()
    ax = plt.subplot(1,1,1)

    plt.errorbar(binned["bin_middles"],
                binned["means"],
                xerr=binned["bin_width"],
                yerr=binned["means_err"],
                fmt="o")

    ax.set_xlabel("mean current in SiPMs / $\si{\micro\ampere}$")
    ax.set_ylabel("mean of ped_sum_mean")

    fig.savefig("ped_sum_mean.pdf")

    binned = mean_data_binned(result["fCurrentsMedMean"], result["pedestalSize"],
                        nBins=50, max_val=30)

    fig = plt.figure()
    ax = plt.subplot(1,1,1)

    plt.errorbar(binned["bin_middles"],
                binned["means"],
                xerr=binned["bin_width"],
                yerr=binned["means_err"],
                fmt="o")

    ax.set_xlabel("mean current in SiPMs / $\si{\micro\ampere}$")
    ax.set_ylabel("mean of pedestal Size")

    fig.savefig("pedestalSize.pdf")

if __name__ == '__main__':
    main()
