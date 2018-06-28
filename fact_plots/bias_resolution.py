import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def plot_bias_resolution(
        df,
        bins,
        ax_bias=None,
        ax_resolution=None,
        prediction_key='gamma_energy_prediction',
        true_energy_key='corsika_event_header_total_energy',
        estimated=False,
        std=False,
        ):
    '''
    Plot energy bias and resolution vs true energy

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of simulated gamma events with columns `true_energy_key`
        and `prediction_key`
    bins: array-like
        bin edges
    ax_bias: matplotlib.axes.Axes
        axes for the bias plot
    ax_resolution: matplotlib.axes.Axes
        axes for the resolion plot
    prediction_key: str
        Column name for the energy prediction
    true_energy_key: str
        Column name for the true energy
    estimated: bool
        plot vs estimated instead of true energy
    std: bool
        If True, use standard deviation instead of 1-sigma percentiles
        to calculate resolution
    '''

    ax_bias = ax_bias or plt.gca()
    ax_res = ax_resolution or ax_bias.twinx()

    if estimated:
        df['bin'] = np.digitize(df[prediction_key], bins)
    else:
        df['bin'] = np.digitize(df[true_energy_key], bins)

    df['rel_error'] = (df[prediction_key] - df[true_energy_key]) / df[true_energy_key]

    binned = pd.DataFrame(index=np.arange(1, len(bins)))
    binned['center'] = 0.5 * (bins[:-1] + bins[1:])
    binned['width'] = np.diff(bins)

    resolution_quantiles = []
    resolution_stds = []
    bias = []

    for i in tqdm(range(100)):
        grouped = df.sample(len(df), replace=True).groupby('bin')
        bias.append(grouped['rel_error'].mean())
        lower_sigma = grouped['rel_error'].agg(lambda s: np.percentile(s, 15.87))
        upper_sigma = grouped['rel_error'].agg(lambda s: np.percentile(s, 84.13))
        resolution_quantiles.append(0.5 * (upper_sigma - lower_sigma))
        resolution_stds.append(grouped.rel_error.std())

    bias = pd.concat(bias, axis=1)
    resolution_quantiles = pd.concat(resolution_quantiles, axis=1)
    resolution_stds = pd.concat(resolution_stds, axis=1)

    binned['bias'] = bias.mean(axis=1)
    binned['bias_err'] = bias.std(axis=1)
    binned['resolution_quantiles'] = resolution_quantiles.mean(axis=1)
    binned['resolution_quantiles_err'] = resolution_quantiles.std(axis=1)
    binned['resolution'] = resolution_stds.mean(axis=1)
    binned['resolution_err'] = resolution_stds.std(axis=1)

    ax_bias.errorbar(
        binned['center'],
        binned['bias'],
        xerr=0.5 * binned['width'],
        yerr=binned['bias_err'],
        label='Bias',
        linestyle='',
        color='C0'
    )

    if std is False:
        ax_res.errorbar(
            binned['center'],
            binned['resolution_quantiles'],
            xerr=0.5 * binned['width'],
            yerr=binned['resolution_quantiles_err'],
            label='Resolution',
            linestyle='',
            color='C1',
        )
    else:
        ax_res.errorbar(
            binned['center'],
            binned['resolution'],
            yerr=binned['resolution_err'],
            xerr=0.5 * binned['width'],
            label='Resolution',
            linestyle='',
            color='C1',
        )

    ax_res.set_xscale('log')

    return ax_bias, ax_res
