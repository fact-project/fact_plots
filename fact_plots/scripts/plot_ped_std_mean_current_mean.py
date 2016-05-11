#!/usr/bin/env python3
"""Plot effective area for a dataset from HDF5 files. e.g.

TEXINPUTS=$(pwd): python plot_numisland_current_mean.py /home/jbuss/plots/numIslandsCureents.pdf /fhgfs/users/jbuss/20140615_27_cStd.hdf /fhgfs/users/jbuss/20140615_27_c6_4.hdf /fhgfs/users/jbuss/20140615_27_c7_5.hdf --password r3adfac! --pattern "_c,.hdf" --unit="p.e." --feature="Level:"

Usage:
    plot_ped_std_mean_cureent_mean.py <outputfile> <datafiles>... [options]

Options:
    --tablename=<name>      [default: table]
    --password=<name>       password for the factdb
    --cuts <cuts>           cuts for the pandas data frame as comma separated list
    --default_cuts <cuts>   choose predefined default cuts as comma separted list e.g. qualitycuts, precuts
    --feature=<name>        feature name of these comparisons
    --unit=<name>           unit of feature these comparisons
    --pattern <name>        pattern of the feature value string e.g "_xT,_c"
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

def buildLabel(path, pattern, feature=None, unit=None):
    if not pattern:
        return pattern
    label_val = os.path.basename(datafile).split(pattern[0])[-1].split(pattern[1])[0]
    if "Std" in label_val:
        label_val = "5.5_3"
    label_val = " ".join(label_val.split('_'))
    if feature:
        label_val = feature + " " + label_val
    if unit:
        label_val += " " + unit
    return label_val

def combine_data_to_db(db_df, data_df):
    df = data_df.rename(columns={'RUNID': 'fRunID', 'NIGHT': 'fNight'})
    df = pd.merge(df, db_df, on=['fNight', 'fRunID'])
    logger.debug("{} events after merge with db".format(len(df)))
    return df

def mean_data_binned(df, key_bin, key_mean, bin_width, min_val=None, max_val=None):
    # logger.debug("Feature for binning {}".format(df_for_bin.describe()))
    # logger.debug("Feature for mean {}".format(df_for_mean.describe()))
    df = df.copy()
    if min_val is None:
        min_val = np.min(df[key_bin])
    if max_val is None:
        max_val = np.max(df[key_bin])

    nBins = int((max_val - min_val)/bin_width)

    logger.debug("Bin width={}".format(bin_width))
    edges = np.linspace(min_val, max_val, nBins + 1)
    df['bin'] = np.digitize(df[key_bin], bins=edges)
    df = df.query('0 < bin < {}'.format(nBins + 1))

    def error_low(x):
        return np.mean(x) - np.percentile(x, 15)

    def error_high(x):
        return np.percentile(x, 85) - np.mean(x)

    binned = df.groupby('bin').aggregate({key_mean:
        ['mean', 'std', error_low , error_high, 'size', 'sem']
    })
    binned.columns = ['_'.join(col) for col in binned.columns]
    binned['bin_center'] = pd.Series(0.5 * (edges[1:] + edges[:-1]), index=np.arange(nBins) + 1)
    return binned

def main():
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +  '%(message)s'), level=logging.INFO)


    datafiles   = args["<datafiles>"]
    outputfile  = args["<outputfile>"]

    tablename   = args["--tablename"]
    password    = args["--password"]

    cuts            = args["--cuts"]
    default_cuts    = args["--default_cuts"]
    feature         = args["--feature"]
    unit            = args["--unit"]

    pattern         = args["--pattern"]
    if pattern:
        pattern = pattern.split(",")

    logger.info("loading Data Base")
    factdb = create_engine("mysql+pymysql://factread:{}@129.194.168.95/factdata".format(password))
    rundb = pd.read_sql("SELECT * from RunInfo WHERE (fNight > 20140614 AND fNight < 20140629)", factdb)

    logger.debug(rundb["fCurrentsMedMean"].describe())

    logger.info("loading Files")
    df_list = []
    labels = []

    for datafile in datafiles:
        logger.info("loading: {}".format(datafile))
        df = pd.read_hdf(datafile, tablename)
        logger.debug("{} Events in file".format(len(df)))
        logger.info("merging database")
        df = combine_data_to_db(rundb, df)
        # df = df.query("Size > 60")
        df_list.append(df)
        labels.append( buildLabel (datafile, pattern, feature, unit))

    fig = plt.figure()
    ax = plt.subplot(1,1,1)

    feature_name = "ped_std_mean"
    gain = 257.

    logger.info("binning data")
    for df, label in zip(df_list, labels):
        binned = mean_data_binned(df, "fCurrentsMedMean", feature_name, bin_width=1.01)
        ax.errorbar(binned["bin_center"].values,
                    binned[feature_name+"_mean"].values/gain,
                    xerr=0.5,
                    # yerr=binned[feature_name+"_std"].values/binned[feature_name+"_size"].values,
                    yerr=binned[feature_name+"_sem"].values/gain,
                    fmt=",",
                    label = label,
                    capsize=1,
                    )

    ax.set_xlabel("Mean current in pixels / $\si{\micro A}$")
    ax.set_ylabel("Mean pedestal standard deviation / $\mathrm{p.e.}$")

    fig.savefig(outputfile)

    #pdftoppm -png -r 600 test.pdf > test.png
if __name__ == '__main__':
    main()
