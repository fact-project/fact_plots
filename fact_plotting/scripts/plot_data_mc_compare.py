#!/usr/bin/env python2
"""Plot Data MonteCarlo comparison plots from HDF5 files

Usage:
    plot_data_mc_compare.py <outputfile> <datafiles>... [options]

Options:
    --ignore <keys>         keys to ignore as comma separated list
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
import logging
from IPython import embed
import gc
import os

from ..default_plots import default_plots
from ..default_cuts import cuts as default_cuts

def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

# default plotting options for all comparison plots

default_plot_option = dict(
    histtype='step',
    normed=True,
    bottom=0,
    align='left',
)

args = docopt(__doc__)
logger  = logging.getLogger(__name__)

logging.captureWarnings(True)
logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +  '%(message)s'), level=logging.INFO)

datafiles   = args["<datafiles>"]
outputfile  = args["<outputfile>"]

tablename   = args["--tablename"]
ignorekeys  = args["--ignore"]

cuts            = args["--cuts"]
default_cuts    = args["--default_cuts"]

plotting_cuts = list()
if cuts:
    print("will use given cuts")
    plotting_cuts.extend(cuts.split(","))

if default_cuts:
    print("will use given default cuts: ", default_cuts)
    list_of_default_cuts = default_cuts.split(",")
    for cut_set in list_of_default_cuts:
        plotting_cuts.extend(dc.cuts[cut_set])

if plotting_cuts:
    print("using cuts:", plotting_cuts)
    cuts = " & ".join(plotting_cuts)


df_list = []
key_list = []
common_keys = None
for i, datafile in enumerate(datafiles):
    logger.info("loading: {}".format(datafile))
    df = pd.read_hdf(datafile, tablename)
    logger.debug("{} Events in file".format(len(df)))
    if cuts:
        df = df.query(cuts)
    df["filename"] = os.path.basename(datafile)[:-4]
    df_list.append(df)
    if i == 0:
        common_keys = df.keys()
    else:
        common_keys = set(common_keys).intersection(df.keys())


#Sort the list of keys
common_keys = sorted(common_keys)

if ignorekeys != None:
    common_keys = set(common_keys).difference(ignorekeys)
    for key in ignorekeys:
        logger.info("skipping column{}: on ignore list".format(key))

with PdfPages(outputfile) as pdf:
    logger.info("\nList of Keys:")
    for key in common_keys:
        print(key)

        #skip tupples
        if isinstance(df_list[0][key][0], (list, tuple)):
            logger.info("skipping column{}: cannot interprete content".format(key))
            continue

        plt.figure()
        plt.title(key)
        plot_option = None
        if key in dp.default_plots:
            plot_option = dp.default_plots[key]

            if plot_option == False:
                plt.close()
                continue

            gc.collect()
            print(default_plot_option)

            xlabel = key
            func = None
            xUnit=""

            if plot_option == None:
                plot_option = default_plot_option
            else:
                # embed()
                func    = plot_option["func"]
                xUnit   = plot_option["xUnit"]
                xlabel  += " / " + xUnit

                if func and func.__name__ and not "lambda" in func.__name__:
                    # embed()
                    func_name = str(func.__name__)
                    print("Function:", func_name+"({})".format(key))
                    xlabel = func_name+"({})".format(xlabel)

                del plot_option["func"]
                del plot_option["xUnit"]

                plot_option = merge_dicts(default_plot_option, plot_option)

            for df in df_list:
                data = df[key]

                if func:
                    data = func(data)

                try:
                    plt.hist(data.values, label=df["filename"][0], **plot_option)

                except Exception as inst:
                    print(type(inst))     # the exception instance
                    print(inst.args)      # arguments stored in .args
                    print(inst)

                plt.xlabel(xlabel)
                plt.ylabel("Frequency")

            plt.legend(loc='best')
            # plt.show()
            pdf.savefig()

            plt.close()
            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'Data MC Comparison plots'
            d['Author'] = u'Jens Buss'
            d['Subject'] = 'Comparison'
            d['Keywords'] = 'Data:{}\nRest:{}\nCuts:{}'.format(datafiles, str(args), str(cuts))
            d['CreationDate'] = datetime.datetime.today()
            d['ModDate'] = datetime.datetime.today()
