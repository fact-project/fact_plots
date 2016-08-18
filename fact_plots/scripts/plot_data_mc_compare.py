#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
import matplotlib
import datetime
import click
# matplotlib.rc_file("./matplotlibrc")
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib_hep import histpoints
from matplotlib.backends.backend_pdf import PdfPages
from cycler import cycler
from docopt import docopt
import pandas as pd
import logging
from IPython import embed
import gc
import os
import h5py

from ..default_plots import default_plots
from ..default_cuts import cuts as d_cuts
from ..utils import merge_dicts

default_plot_option = dict(
    # histtype='step',
    # normed=True,
    bottom=0,
    align='left',
)

color_cycle = cycler(color=['black', 'r', 'g', 'b'])

logger  = logging.getLogger(__name__)

logging.captureWarnings(True)
logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +  '%(message)s'), level=logging.INFO)


def aggregatePlottingCuts(cuts, default_cuts):
    plotting_cuts = list()
    if cuts:
        logger.info("will use given cuts")
        plotting_cuts.extend(cuts.split(","))

    if default_cuts:
        logger.info("will use given default cuts: {}".format(default_cuts))
        list_of_default_cuts = default_cuts.split(",")
        for cut_set in list_of_default_cuts:
            plotting_cuts.extend(d_cuts[cut_set])

    if plotting_cuts:
        logger.info("using cuts:{}".format( plotting_cuts))
        cuts = " & ".join(plotting_cuts)

    return cuts

def loadData(datatupels, cuts):
    datafiles = []
    df_list = []
    scales = []
    labels = []
    common_keys = None

    for i, datatupel in enumerate(datatupels):
        datafile, tablename, scale, label = datatupel

        datafiles.append(datafile)
        scales.append(scale)
        labels.append(label)

        logger.info("loading: {}, key={}".format(datafile, tablename))
        df=pd.DataFrame()
        try:
            df = pd.read_hdf(datafile, tablename)
        except KeyError:
            f = h5py.File(datafile)
            keys = list(f.keys())
            logger.error("Key '{}' not in datafile:\n{}\nPossible Keys are:{}".format(tablename, datafile, keys))
            f.close()
            exit()
        df.replace(np.inf, np.nan, inplace=True)
        df.dropna(inplace=True)
        df["filename"] = os.path.basename(datafile).split(".hdf")[0]
        logger.debug("{} Events in file".format(len(df)))
        if cuts:
            df = df.query(cuts)
        df_list.append(df)
        if i == 0:
            common_keys = df.keys()
        else:
            common_keys = set(common_keys).intersection(df.keys())

    return df_list, datafiles, scales, labels, sorted(common_keys)


def mkDirAtDestination(outputfile):
    fname = os.path.basename(outputfile).split(".")[0]
    dirname = os.path.dirname(outputfile)
    path = os.path.join(dirname, fname)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def calcDataRange(df_list, key):
    min_val = float("inf")
    max_val = float("-inf")
    for df in df_list:
        if df[key].max() > max_val:
            max_val = df[key].max()
        if df[key].min() < min_val:
            min_val = df[key].min()
    return min_val, max_val



# default plotting options for all comparison plots
@click.command()
@click.argument('outputfile', type=click.Path(exists=False, dir_okay=True, file_okay=True))
@click.option('--datatupels', '-d', multiple=True, nargs=4, type=click.Tuple([click.Path(exists=True, dir_okay=True), click.STRING, click.FLOAT, click.STRING]), help='tupels of: path to hdf5 file, table name, scale, label')
@click.option('--ignorekeys', '-i', type=click.STRING, default=None, help='comma seperated list of keys to ignore')
@click.option('--cuts', '-c', type=click.STRING, default=None, help='cuts for the pandas data frame as comma separated list')
@click.option('--default_cuts', type=click.STRING, default=None, help="choose predefined default cuts as comma separted list e.g. qualitycuts, precuts")
def main(outputfile, datatupels, ignorekeys, cuts, default_cuts):
    '''Plot Data MonteCarlo comparison plots from HDF5 files'''

    cuts = aggregatePlottingCuts(cuts, default_cuts)

    df_list, datafiles, scales, labels, common_keys = loadData(datatupels, cuts)

    if ignorekeys != None:
        common_keys = set(common_keys).difference(ignorekeys)
        for key in ignorekeys:
            logger.info("skipping column{}: on ignore list".format(key))

    picturePath = mkDirAtDestination(outputfile)

    with PdfPages(os.path.join(picturePath, os.path.basename(outputfile))) as pdf:
        logger.info("\nList of Keys:")
        for key in common_keys:
            logger.info(key)

            #skip tupples
            if isinstance(df_list[0][key].iloc[0], (list, tuple)):
                logger.info("skipping column {}: cannot interprete content".format(key))
                continue

            fig = plt.figure()
            plt.title(key)
            plot_option = None
            if key in default_plots:
                plot_option = default_plots[key]

                if plot_option == False:
                    plt.close()
                    continue


                data_range = calcDataRange(df_list, key)

                gc.collect()
                logger.info(default_plot_option)

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
                        logger.info("Function: {}({})".format(func_name, key))
                        xlabel = func_name+"({})".format(xlabel)

                    del plot_option["func"]
                    del plot_option["xUnit"]

                    plot_option = merge_dicts(default_plot_option, plot_option)
                    try:
                        if "bins" and "range" in plot_option:
                            if not plot_option["range"] == None:
                                plot_option["bins"] = np.linspace(*plot_option["range"], plot_option["bins"])
                            else:
                                plot_option["bins"] = np.linspace(*data_range, plot_option["bins"])
                    except:
                        embed()

                for df, scale, label, c in zip(df_list, scales, labels, color_cycle()):
                    data = df[key]

                    if func:
                        data = func(data)

                    try:
                        # plt.hist(data.values, label=df["filename"].iloc[0], normed=scale, color=c["color"], **plot_option)
                        ax = fig.gca()
                        ax.grid(True)
                        x, y, norm = histpoints(data.values, xerr='binwidth', yerr="sqrt", label=label,
                                                fmt='none', capsize=0, normed=scale, ecolor=c["color"], **plot_option)
                        ax.fill_between(x, y[1], 0, alpha=0.2, linewidth=0.01, step='mid', facecolor=c["color"])
                        if "log" in plot_option:
                            if plot_option["log"]:
                                ax.set_yscale("log", nonposy='clip')
                        if "range" in plot_option:
                            ax.set_xlim(plot_option["range"])

                    except Exception:
                        logger.exception("Plotting failed for {} in file {}".format(key, df["filename"]))


                    plt.xlabel(xlabel)
                    plt.ylabel("Frequency")

                plt.legend(loc='best')

                plt.savefig(os.path.join(picturePath, key+".png"))
                # plt.show()
                pdf.savefig()

                plt.close()
                # We can also set the file's metadata via the PdfPages object:
                d = pdf.infodict()
                d['Title'] = 'Data MC Comparison plots'
                d['Author'] = u'Jens Buss'
                d['Subject'] = 'Comparison'
                d['Keywords'] = 'Data:{}\nCuts:{}'.format(str(", ".join(datafiles)), str(cuts))
                d['CreationDate'] = datetime.datetime.today()
                d['ModDate'] = datetime.datetime.today()


if __name__ == '__main__':
    main()
