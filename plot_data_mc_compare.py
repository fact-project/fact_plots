#!/usr/bin/env python2
"""Plot Data MonteCarlo comparison plots from HDF5 files

Usage:
    plot_data_mc_compare.py <datafile> <protonfile> <outputfile> [options]

Options:
    --ignore <keys>         keys to ignore as comma separated list
    --tablename=<name>      [default: table]
    --gammafile=<name>      [default: None]
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

print(matplotlib.matplotlib_fname())


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

datafile    = args["<datafile>"]
protonfile  = args["<protonfile>"]
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
    print("will use given default cuts")
    list_of_default_cuts = default_cuts.split(",")
    for cut_set in list_of_default_cuts:
        plotting_cuts.extend(dc.cuts[cut_set])

if plotting_cuts:
    print("using cuts:", plotting_cuts)

print("loading data file")
data_df = pd.read_hdf(datafile, tablename)
print("loading proton file")
proton_df = pd.read_hdf(protonfile, tablename)
# gamma_df = pd.read_hdf(gammafile, tablename)

data_keys = data_df.keys()
proton_keys = proton_df.keys()


common_keys = set(data_keys).intersection(proton_keys)
common_keys = sorted(common_keys)
print("\nList of Keys:")

with PdfPages(outputfile) as pdf:
    for key in common_keys:
        print(key)

        if ignorekeys != None:
            if key in ignorekeys:
                print("skipping column{}: on ignore list".format(key))
                continue

        if isinstance(proton_df[key][0], (list, tuple)) \
        or isinstance(data_df[key][0], (list, tuple)):
            print("skipping column{}: cannot interprete content".format(key))
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

            if plotting_cuts:
                cuts = " & ".join(plotting_cuts)
                data = data_df.query(cuts)[key]
                proton = proton_df.query(cuts)[key]
            else:
                data = data_df[key]
                proton = proton_df[key]

            if plot_option == None:
                plot_option = default_plot_option
            else:
                func = plot_option["func"]
                xUnit = plot_option["xUnit"]

                xlabel += " / " + xUnit

                if func:
                    print("Function:", func+"(data_df.{})".format(key))
                    data = eval(func+"(data_df.{})".format(key))
                    proton = eval(func+"(proton_df.{})".format(key))
                    if "np." in func:
                        func = func.replace("np.", "")
                        xlabel = func+"({})".format(xlabel)

                del plot_option["func"]
                del plot_option["xUnit"]

                plot_option = merge_dicts(default_plot_option, plot_option)

            try:
                plt.hist(data.values, color="black", label="data", **plot_option)
                plt.hist(proton.values, color="red", label="proton", **plot_option)

            except Exception as inst:
                print(type(inst))     # the exception instance
                print(inst.args)      # arguments stored in .args
                print(inst)

            plt.xlabel(xlabel)
            plt.ylabel("Frequency")

        plt.legend()
        # plt.show()
        pdf.savefig()

        plt.close()
        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'Data MC Comparison plots'
        d['Author'] = u'Jens Buss'
        d['Subject'] = 'Comparison'
        d['Keywords'] = 'Data:{}\n Proton:{}\n Rest:{}'.format(datafile, protonfile, str(args))
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()
