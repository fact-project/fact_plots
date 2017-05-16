import matplotlib.pyplot as plt
from fact.analysis.statistics import li_ma_significance
import numpy as np
from matplotlib_hep import histpoints
from fact_plots.plotting import plotInfoBox, plotTitleBox
from fact_plots.utils import getTheta2Arrays, generate_theta_deg
from fact.analysis.core import (
    split_on_off_source_dependent,
    split_on_off_source_independent,
    )


def plot_theta2(events, nbins=100, on_key="Theta_deg",
                theta2_max=0.2,
                ax=None, alpha=0.2, period=None,
                title=None, prediction_threshold=None,
                theta_cut=None, x_range=None, is_source_dependend=False):
    ax = ax or plt.gca()

    theta2_cut = theta_cut**2

    generate_theta_deg(events)

    theta_on, theta_off = getTheta2Arrays(events, on_key=on_key,
                                          prediction_threshold=prediction_threshold,
                                          theta_cut=theta2_cut,
                                          is_source_dependend=prediction_threshold
                                          )
    n_on = len(theta_on)
    n_off = len(theta_off)

    excess_events = n_on - alpha * n_off
    excess_events_err = np.sqrt(n_on + alpha**2 * n_off)
    significance = li_ma_significance(n_on, n_off, alpha=alpha)

    bins = np.linspace(0, theta2_max, nbins)

    # Plot the Theta2 Distributions
    sig_x, sig_y, sig_norm = histpoints(
                                theta_on,
                                bins=bins,
                                xerr='binwidth',
                                label='On',
                                fmt='none',
                                ecolor='b',
                                capsize=0
                                )

    back_x, back_y, back_norm = histpoints(
                                theta_off,
                                bins=bins,
                                xerr='binwidth',
                                label='Off',
                                fmt='none',
                                ecolor='r',
                                capsize=0,
                                scale=alpha,
                                yerr='sqrt'
                                )

    # Fill area underneeth background
    ax.fill_between(back_x, back_y[1], 0, facecolor='grey', alpha=0.2, linewidth=0.0)

    # Mark theta cut with a line0.5*(info_left+info_right),
    if theta2_cut:
        ax.axvline(x=theta2_cut, linewidth=1, color='k', linestyle='dashed')

    info_text = 'Significance: {:.2f}, Alpha: {:.2f}\n'.format(significance, alpha)
    if period:
        info_text = period + ',\n' + info_text
    info_text += 'Prediction Threshold: {:.2f}, '.format(prediction_threshold)
    if theta_cut:
        info_text += 'Theta Sqare Cut: {:.2f}'.format(theta_cut)
    info_text += '\n'
    info_text += '{:.2f}+/-{:.2f} excess events, '.format(
        excess_events, excess_events_err)
    info_text += '{:.0f} background events \n'.format(n_off)
    info_text += '{:.0f} events in source region, '.format(n_on)

    # Summary Box
    plotInfoBox(info_text, ax=ax)

    # Title Box
    if title is not None:
        plotTitleBox(title, ax=ax)

    ax.set_xlabel(r'$\theta^2 / \deg^2$')
    ax.set_ylabel('Frequency')

    return ax
