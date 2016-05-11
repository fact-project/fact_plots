"""Plot effective area for a dataset from HDF5 files. e.g.

Usage:
    plot_theta.py <datafile> <outputfile> [options]

Options:
    --tablename=<name>      [default: table]
    --bins=<bins>           number of bins [default: 100]
    --threshold=<threshold> threshold [default: '0.5']
"""

import matplotlib.pyplot as plt
from docopt import docopt
import numpy as np
import pandas as pd
import logging
from fact_plots.utils import li_ma_significance, theta_mm_to_theta_squared_deg

def main():

    logger  = logging.getLogger(__name__)
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +  '%(message)s'), level=logging.DEBUG)

    args = docopt(__doc__)
    path = args["<datafile>"]
    out  = args["<outputfile>"]

    tablename   = args["--tablename"]
    bins        = int(args["--bins"])
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
    hist_signal, edges, _ = plt.hist(signal_theta, bins=bins, alpha=0.6, label='On region', range=(0,0.3))
    hist_signal, _, _ = plt.hist(background_theta, bins=edges, alpha=0.6, label='Off region', range=(0,0.3))

    # hist_background, edges = np.histogram(background_theta, bins=edges)
    # hist_background = hist_background * alpha
    # plt.bar(edges[:-1], hist_background, width=edges[1]-edges[0], alpha=0.6, label='Off region')
    # t_off = pd.concat([df_background[["Theta_Off_1","Theta_Off_2","Theta_Off_3","Theta_Off_4", "Theta_Off_5"]]])
    # y, edges = np.histogram(t_off, bins=edges)
    # embed()
    # y =y * alpha
    # plt.bar(edges[:-1], y, width=edges[1]-edges[0], alpha=0.6, color='gray', label="huhrensohn")





        # hist_background, edges , _ = plt.hist(df_background.values, bins=edges, alpha=0.6, label='Off region')
    plt.xlabel("Theta^2 in mm^2")
    plt.legend()
    # plt.show()
    plt.savefig(out)

if __name__ == '__main__':
    theta_square_plot()
