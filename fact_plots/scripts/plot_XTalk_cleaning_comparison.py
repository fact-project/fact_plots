#!/usr/bin/env python3
"""Plot comparison of different cleaning levels for datasets from HDF5 files

Usage:
    plot_XTalk_cleaning_comparison.py <datafolder> <settingString> <outputfile> [options]

Options:
    --tablename=<name>      [default: table]
    --cuts <cuts>           cuts for the pandas data frame as comma separated list
    --default_cuts <cuts>   choose predefined default cuts as comma separted list e.g. qualitycuts, precuts
    --split_tring=<name>    string to split
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
from .. import default_plots as dp
from .. import default_cuts as dp
import logging
import gc
import os
import resolutionHelper as reso
from cycler import cycler

logger  = logging.getLogger(__name__)

def loadFiles(datafolder, identifierString, tablename, split_tring="_c", file_ending="hdf"):
    logger.info("Loading files and storing them to Dataframe")
    data_dict = dict()
    for root, dirs, files in os.walk(datafolder):
        for f in files:
            if identifierString in f:
                file_abs_path = os.path.join(root, f)
                logger.info("reading: {}".format(file_abs_path))
                cleaning_setting_string = f.replace("."+file_ending, "").split(split_tring)[-1]
                logger.info("cleaning: {}".format(cleaning_setting_string))
                temp_df = pd.read_hdf(file_abs_path, tablename)
                temp_df["cleaning"] = cleaning_setting_string
                data_dict[cleaning_setting_string] = temp_df

    return data_dict


def appendDataFrames(df1, df_list):
    for df in df_list:
        df1.append(df)
    return df1


default_plot_option = dict(
    histtype='step',
    normed=True,
    bottom=0,
    align='left',
)

settings = (
    dict(   title= "Median number of estimated photons in shower",
            xLabel= "Number of photons / p.e.",
            estimated = "truePixel_phChargeShower_median",
            truth = "true_phChargeShower_median",
            nbins = 5 ,
            max = 60, min= 0
        ),
    dict(   title= "75 percentile number of estimated photons in shower",
            xLabel= "Number of photons / p.e.",
            estimated = "truePixel_phChargeShower_p75",
            truth = "true_phChargeShower_p75" ,
            nbins = 5 ,
            max = 60, min= 0
        ),
    dict(   title= "Max number of estimated photons in shower",
            xLabel= "Number of photons / p.e.",
            estimated = "truePixel_phChargeShower_max",
            truth = "true_phChargeShower_max" ,
            nbins = 5 ,
            max = 60, min= 0
        ),
    dict(   title= "Min number of estimated photons in shower",
            xLabel= "Number of photons / p.e.",
            estimated = "truePixel_phChargeShower_min",
            truth = "true_phChargeShower_min" ,
            nbins = 5 ,
            max = 60, min= 0
        ),
    dict(   title= "Mean number of estimated photons in shower",
            xLabel= "Number of photons / p.e.",
            estimated = "truePixel_phChargeShower_mean",
            truth = "true_phChargeShower_mean",
            nbins = 5 ,
            max = 60, min= 0
        ),
    dict(   title= "25 percentile number of estimated photons in shower",
            xLabel= "Number of photons / p.e.",
            estimated = "truePixel_phChargeShower_p25",
            truth = "true_phChargeShower_p25" ,
            nbins = 5 ,
            max = 60, min= 0
        ),
    dict(   title= "Number of pixels in shower",
            xLabel= "Number of pixel",
            estimated = "numPixelInShower",
            truth = "true_numPixelInShower",
            nbins = 5 ,
            max = 150, min= 0
        )
    )

args    = docopt(__doc__)

datafolder  = args["<datafolder>"]
xTalk       = args["<settingString>"]
outputfile  = args["<outputfile>"]
tablename   = args["--tablename"]
cuts        = args["--cuts"]
split_tring = args["--split_tring"]

logging.captureWarnings(True)
logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +  '%(message)s'), level=logging.INFO)

data_dict = loadFiles(datafolder, xTalk, tablename)
c_cycle = cycler('color', ['red', 'b', 'black', 'g', 'yellow', 'orange','darkgrey', 'c', 'm'])


with PdfPages(outputfile) as pdf:
    for setting in settings:
        fig_hist, axs_hist = plt.subplots(nrows=1, ncols=2, sharey=True)
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

        axs[0].set_prop_cycle(c_cycle)
        axs[1].set_prop_cycle(c_cycle)
        axs_hist[0].set_prop_cycle(c_cycle)
        axs_hist[1].set_prop_cycle(c_cycle)

        for key in data_dict:
            data = data_dict[key]
            data[setting["estimated"]].describe()

            axs_hist[0].hist(data[setting["truth"]], bins=20, log=True, histtype='step',
                                align='left', label=key)
            axs_hist[1].hist(data[setting["estimated"]], bins=20, log=True, histtype='step',
                                align='left', label=key)

            res = reso.calc_resolution(data[setting["truth"]], data[setting["estimated"]],
                                        max_e=setting["max"], min_e=setting["min"],
                                        bin_width=setting["nbins"], plot_hists=False)

            # plt.figure()
            # axs[0].errorbar(res["bin_middles"][:len(res["means"])], res["means"], xerr=res["bin_width"]/2, yerr=res["err_means"], fmt='o', label=key)
            # axs[1].errorbar(res["bin_middles"][:len(res["res"])], res["res"], xerr=res["bin_width"]/2, yerr=res["err_res"], fmt='o', label=key)
            axs[0].errorbar(res["bin_middles"], res["means"], xerr=res["bin_width"]/2,
                                    yerr=res["err_means"], fmt='o', label=key,
                                    markersize=1.8, capsize=1)
            axs[1].errorbar(res["bin_middles"], res["res"], xerr=res["bin_width"]/2,
                                    yerr=res["err_res"], fmt='o', label=key,
                                    markersize=1.8, capsize=1)

        axs[0].set_title(setting["title"])
        axs[1].set_xlabel(setting["xLabel"])
        axs[0].set_ylabel("Bias")
        axs[1].set_ylabel("Resolution")
        fig.subplots_adjust(hspace=0.06)
        axs[0].legend(fontsize=8)
        axs[1].legend(fontsize=8)
        pdf.savefig(fig)

        axs_hist[0].set_ylabel("Frequency")
        axs_hist[0].set_xlabel(setting["truth"])
        axs_hist[1].set_xlabel(setting["estimated"])
        axs_hist[0].legend(fontsize=8)
        axs_hist[1].legend(fontsize=8)
        #plt.show()
        pdf.savefig(fig_hist)
