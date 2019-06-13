import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_angular_resolution(
    df,
    bins,
    ax=None,
    theta_key='theta_deg',
    true_energy_key='corsika_event_header_total_energy',
    min_bin_count=200,
    **kwargs
):
    '''
    Plot the angular resolution from a dataframe of simulated
    gamma ray events.

    Parameters
    ----------

    df: pd.DataFrame
        DataFrame of simulated gamma-rays containing the
        columns `true_energy_key`, and `theta_key`
    n_bins: int
        number of bins in energy
    theta_key: str
        column name for theta
    true_energy_key: str
        column name for the true gamma energy
    min_bin_count: int
        Minimum number of events in an energy bin for the bin to be shown
    '''

    ax = ax or plt.gca()

    df['bin'] = np.digitize(df[true_energy_key], bins)
    df.drop(df[df['bin'] == len(bins)].index, axis=0, inplace=True)
    df.drop(df[df['bin'] == 0].index, axis=0, inplace=True)

    binned = pd.DataFrame(index=np.arange(1, len(bins)))
    binned['center'] = 0.5 * (bins[:-1] + bins[1:])
    binned['width'] = np.diff(bins)

    grouped = df.groupby('bin')

    values = []
    for i in range(100):
        sampled = df.sample(len(df), replace=True).groupby('bin')
        resolution = np.full(len(bins) - 1, np.nan)
        s = sampled[theta_key].agg(lambda s: np.percentile(s.values, 68))

        resolution[s.index.values - 1] = s.values
        values.append(resolution)

    binned['angular_resolution'] = np.nanmean(values, axis=0)
    binned['angular_resolution_err'] = np.nanstd(values, axis=0)
    binned['size'] = grouped.size()

    binned = binned.query('size > @min_bin_count')

    linestyle = kwargs.pop('ls', '')
    linestyle = kwargs.pop('linestyle', linestyle)

    ax.errorbar(
        binned['center'],
        binned['angular_resolution'],
        xerr=0.5 * binned['width'],
        yerr=binned['angular_resolution_err'],
        linestyle=linestyle,
        **kwargs
    )

    ax.set_xscale('log')

    return ax
