#!/usr/bin/env python2
"""Plot Data MonteCarlo comparison plots from HDF5 files

Usage:
    plot_data_mc_compare.py <datafile> <protonfile> <outputfile> [options]

Options:
    --ignore <keys>      keys to ignore as comma separated list
    --tablename=<name>   [default: table]
    --gammafile=<name>   [default: None]
"""
from __future__ import print_function
import numpy as np
# import matplotlib
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from docopt import docopt
import pandas as pd
import default_plots as dp
args = docopt(__doc__)

datafile = args["<datafile>"]
protonfile = args["<protonfile>"]
outputfile = args["<outputfile>"]

tablename = args["--tablename"]

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
        plt.figure()
        plt.title(key)

        if isinstance(proton_df[key][0], (list, tuple)) \
        or isinstance(data_df[key][0], (list, tuple)):
                plt.close()
                print("not a numeber")
                continue

        plot_option = None
        if key in dp.default_plots:
            plot_option = dp.default_plots[key]

            if plot_option == False:
                plt.close()
                continue

            if plot_option == None:
                try:
                    plt.hist(data_df[key], histtype='step', normed=True, label="data", color="black")
                    plt.hist(proton_df[key], histtype='step', normed=True, label="proton", color="red")
                    plt.legend()

                except Exception as inst:
                    print(type(inst))     # the exception instance
                    print(inst.args)      # arguments stored in .args
                    print(inst)
            else:
                func = plot_option["func"]
                data = data_df[key]
                proton = proton_df[key]
                if func:
                    print("Function:", func+"(data_df.{})".format(key))
                    data = eval(func+"(data_df.{})".format(key))
                    proton = eval(func+"(proton_df.{})".format(key))

                del plot_option["func"]
                del plot_option["xUnit"]

                print(plot_option)

                try:
                    plt.hist(data, histtype='step', normed=True, label="data", color="black", **plot_option)
                    plt.hist(proton, histtype='step', normed=True, label="proton", color="red", **plot_option)

                except Exception as inst:
                    print(type(inst))     # the exception instance
                    print(inst.args)      # arguments stored in .args
                    print(inst)

        plt.legend()
        # plt.show()
        pdf.savefig()
        plt.close()
