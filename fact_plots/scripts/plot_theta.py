"""Plot effective area for a dataset from HDF5 files. e.g.

Usage:
    plot_theta.py <datafile> <outputfile> [options]

Options:
    --tablename=<name>      [default: table]
    --bins=<bins>           number of bins [default: 100]
    --threshold=<threshold> threshold [default: '0.5']
    --title=<title>         [default: FACT Observations]
"""

import matplotlib
import matplotlib.pyplot as plt
from docopt import docopt
import numpy as np
import pandas as pd
import logging
from matplotlib_hep import histpoints
import matplotlib.patches as patches
from fact_plots.utils import li_ma_significance, theta_mm_to_theta_squared_deg
from IPython import embed

def main():

    logger  = logging.getLogger(__name__)
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +  '%(message)s'), level=logging.DEBUG)
    logger.debug(matplotlib.matplotlib_fname())

    args = docopt(__doc__)
    path = args["<datafile>"]
    out  = args["<outputfile>"]

    tablename   = args["--tablename"]
    bins        = int(args["--bins"])
    title       = args["--title"]
    threshold   = 0.5

    alpha = 1.0

    df = pd.read_hdf(path, key=tablename)
    df.columns = [c.replace(':', '_') for c in df.columns]

    theta_keys = ["signal_theta", "background_theta", "Theta", "Theta_Off_1","Theta_Off_2","Theta_Off_3","Theta_Off_4", "Theta_Off_5"]

    df[theta_keys] = df[theta_keys].apply(theta_mm_to_theta_squared_deg, 0)
    best_significance = 0
    best_threshold = 0
    best_theta_cut = 0
    for threshold in np.linspace(0.5, 1, 20):
        df_signal = df.query('(signal_prediction > {})'.format(threshold))
        df_background = df.query('(background_prediction > {})'.format(threshold))


        signal_theta = df_signal['signal_theta'].values
        background_theta = df_background['background_theta'].values
        theta_cuts = np.linspace(0.5, 0.001, 50)
        for theta_cut in theta_cuts:
            n_off = len(background_theta[background_theta < theta_cut])
            n_on =len(signal_theta[signal_theta < theta_cut])
            significance = li_ma_significance(n_on, n_off, alpha=alpha)
            if significance > best_significance:
                best_theta_cut = theta_cut
                best_significance = significance
                best_threshold = threshold


    logger.info('Best cut for predictction threshold {} has signifcance: {} with  a theta sqare cut of {}. '.format(best_threshold, best_significance, best_theta_cut))

    # from IPython import embed
    # embed()
    df_signal = df.query('(signal_prediction > {})'.format(best_threshold))
    df_background = df.query('(background_prediction > {})'.format(best_threshold))
    signal_theta = df_signal['signal_theta'].values
    background_theta = df_background['background_theta'].values

    excess_events = len(signal_theta) - len(background_theta)
    background_events = len(background_theta)

    theta_max = 0.3
    bins = np.linspace(0, theta_max, bins)

    #Define measures for plot and info box
    info_left = 0.0
    info_height = 0.3
    info_width = 1.0
    info_bottom = 1.
    info_top    = info_bottom + info_height
    info_right  = info_left + info_width
    plot_height = 1. - info_height
    plot_width  = theta_max


    fig = plt.figure()
    fig.subplots_adjust(top=plot_height)
    ax = fig.gca()

    #Plot the Theta2 Distributions
    sig_x, sig_y, sig_norm = histpoints(signal_theta, bins=bins, xerr='binwidth', label='On', fmt='none', ecolor='b', capsize=0 )
    back_x, back_y, back_norm = histpoints(background_theta, bins=bins, xerr='binwidth', label='Off', fmt='none', ecolor='r', capsize=0)

    #Fill area underneeth background
    ax.fill_between(back_x, back_y, 0, facecolor='grey', alpha=0.2, linewidth=0.0)

    #Mark theta cut with a line0.5*(info_left+info_right),
    ax.axvline(x=best_theta_cut, linewidth=1, color='k', linestyle='dashed')

    # embed()

    # Draw info Box
    p = patches.Rectangle( (info_left, 1.), info_width, info_height, fill=True, transform=ax.transAxes, clip_on=False, facecolor='0.9', edgecolor='black')
    ax.add_patch(p)

    info_text = 'Significance: {:.2f}\n'.format(best_significance)
    info_text += 'Confidence Cut: {:.2f}, Theta Sqare Cut: {:.2f} \n'.format(best_threshold, best_theta_cut)
    info_text += '{} excess events, {} background events \n'.format(excess_events, background_events)


    ax.text(0.5*(info_left+info_right), 0.5*(info_top+info_bottom)-0.05, info_text,
            horizontalalignment='center', verticalalignment='center', fontsize=12, transform=ax.transAxes)

    ax.text(0.5*(info_left+info_right), (info_top), title,
                bbox={'facecolor':'white', 'pad':10},
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14, color='black',
                transform=ax.transAxes)

        # hist_background, edges , _ = plt.hist(df_background.values, bins=edges, alpha=0.6, label='Off region')
    # plt.xlabel("$//Theta^2 in mm^2$")
    plt.xlabel("Theta^2 / mm^2")
    plt.ylabel("Counts")

    plt.legend(fontsize=12)
    # plt.show()
    plt.savefig(out)

if __name__ == '__main__':
    theta_square_plot()
