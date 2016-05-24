"""Plot effective area for a dataset from HDF5 files. e.g.

Usage:
    plot_theta.py <datafile> <outputfile> [options]

Options:
    --tablename=<name>          tablename [default: table]
    --bins=<bins>               number of bins [default: 100]
    --threshold=<threshold>     threshold   [default: 0.5]
    --title=<title>             title   [default: FACT Observations]
    --first=<NIGHT>             first night
    --last=<NIGHT>              last night
    --theta2-cut=<c>            thetaˆ2 cut [default: 0.05]
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

    first_night = args["--first"]
    last_night  = args["--last"]

    theta_cut = float(args['--theta2-cut'])
    prediction_threshold = float(args['--threshold'])

    alpha = 0.2

    df = pd.read_hdf(path, key=tablename)
    df.columns = [c.replace(':', '_') for c in df.columns]

    if first_night:
        df = df.query('(NIGHT >= {})'.format(first_night)).copy()
        logger.info('Using only Data begining with night {}'.format(first_night))

    if last_night:
        df = df.query('(NIGHT <= {})'.format(last_night)).copy()
        logger.info('Using only Data until night {}'.format(last_night))


    night_stats = df.NIGHT.describe()
    logger.debug('Using Nights from {} to {}'.format(int(night_stats['min']), int(night_stats['max'])))
    period = 'Period: {} to {}'.format(int(night_stats['min']), int(night_stats['max']))

    theta_keys = ["Theta"] + ['Theta_Off_{}'.format(off_position) for off_position in range(1, 6)]


    df[theta_keys] = df[theta_keys].apply(theta_mm_to_theta_squared_deg, axis=0)


    # best_significance = 0
    # prediction_threshold = 0
    # theta_cut = 0
    # for threshold in np.linspace(0.5, 1, 20):
    #     df_signal = df.query('(prediction_on > {})'.format(threshold))
    #     df_background = df.query('(background_prediction > {})'.format(threshold))
    #
    #
    #     signal_theta = df_signal['signal_theta'].values
    #     background_theta = df_background['background_theta'].values
    #     theta_cuts = np.linspace(0.5, 0.001, 50)
    #     for theta_cut in theta_cuts:
    #         n_off = len(background_theta[background_theta < theta_cut])
    #         n_on =len(signal_theta[signal_theta < theta_cut])
    #         significance = li_ma_significance(n_on, n_off, alpha=alpha)
    #         if significance > best_significance:
    #             theta_cut = theta_cut
    #             best_significance = significance
    #             prediction_threshold = threshold

    theta_on = df['Theta'][df['prediction_on' ] > prediction_threshold]
    theta_off = pd.Series()
    for off_position in range(1, 6):
        mask = df['prediction_off_{}'.format(off_position)] > prediction_threshold
        theta_off = theta_off.append(df['Theta_Off_{}'.format(off_position)][mask])

    n_on = len(theta_on[theta_on < theta_cut])
    n_off = len(theta_off[theta_off < theta_cut])
    logger.info('N_on = {}, N_off = {}'.format(n_on, n_off))

    excess_events = n_on - alpha * n_off
    significance = li_ma_significance(n_on, n_off, alpha=alpha)

    logger.info(
        'Chosen cuts for prediction threshold {} has signifcance: {} with  a theta sqare cut of {}.'.format(
            prediction_threshold, significance, theta_cut
    ))

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
    sig_x, sig_y, sig_norm = histpoints(theta_on, bins=bins, xerr='binwidth', label='On', fmt='none', ecolor='b', capsize=0)
    back_x, back_y, back_norm = histpoints(theta_off, bins=bins, xerr='binwidth', label='Off', fmt='none', ecolor='r', capsize=0, weights=alpha * np.ones_like(theta_off))

    #Fill area underneeth background
    ax.fill_between(back_x, back_y[1], 0, facecolor='grey', alpha=0.2, linewidth=0.0)

    #Mark theta cut with a line0.5*(info_left+info_right),
    ax.axvline(x=theta_cut, linewidth=1, color='k', linestyle='dashed')

    # embed()

    # Draw info Box
    p = patches.Rectangle( (info_left, 1.), info_width, info_height, fill=True, transform=ax.transAxes, clip_on=False, facecolor='0.9', edgecolor='black')
    ax.add_patch(p)

    info_text = 'Significance: {:.2f}, Alpha: {:.2f}\n'.format(significance, alpha)
    if period:
        info_text = period + ',\n' + info_text
    info_text += 'Confidence Cut: {:.2f}, Theta Sqare Cut: {:.2f} \n'.format(prediction_threshold, theta_cut)
    info_text += '{:.2f} excess events, {:.2f} background events \n'.format(excess_events, n_off)

    ax.text(0.5*(info_left+info_right), 0.5*(info_top+info_bottom)-0.05, info_text,
            horizontalalignment='center', verticalalignment='center', fontsize=10, transform=ax.transAxes)

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
