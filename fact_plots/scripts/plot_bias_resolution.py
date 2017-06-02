import click
import pandas as pd
import numpy as np
from fact.io import read_h5py
import matplotlib.pyplot as plt


@click.command()
@click.argument('gamma_path')
@click.option(
    '--std', default=False, is_flag=True,
    help='Use std instead of inter-percentile distance',
)
@click.option('--n-bins', default=20, type=int)
@click.option('--threshold', type=float)
@click.option('--theta2-cut', type=float)
@click.option('-o', '--output')
def main(gamma_path, std, n_bins, threshold, theta2_cut, output):
    df = read_h5py(
        gamma_path,
        key='events',
        columns=[
            'gamma_energy_prediction',
            'corsika_evt_header_total_energy',
            'gamma_prediction',
            'theta_deg'
        ],
    )

    if threshold:
        df = df.query('gamma_prediction >= @threshold').copy()
    if theta2_cut:
        df = df.query('theta_deg**2 <= @theta2_cut').copy()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    plot_bias_resolution(df, n_bins=n_bins, std=std, ax=ax)

    ax.set_xlabel(r'$\log_{10}(E_\mathrm{true} \,\, / \,\, \mathrm{GeV})$')

    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()


def plot_bias_resolution(
        df,
        n_bins=10,
        ax=None,
        prediction_key='gamma_energy_prediction',
        std=False,
        true_energy_key='corsika_evt_header_total_energy',
        ):

    ax = ax or plt.gca()

    bins = np.logspace(
        np.log10(df[true_energy_key].min()),
        np.log10(df[true_energy_key].max()),
        n_bins + 1
    )

    df['bin'] = np.digitize(df[true_energy_key], bins)
    df['rel_error'] = (df[prediction_key] - df[true_energy_key]) / df[true_energy_key]

    binned = pd.DataFrame(index=np.arange(1, len(bins)))
    binned['center'] = 0.5 * (bins[:-1] + bins[1:])
    binned['width'] = np.diff(bins)

    grouped = df.groupby('bin')
    binned['bias'] = grouped['rel_error'].mean()
    binned['bias_median'] = grouped['rel_error'].median()
    binned['lower_sigma'] = grouped['rel_error'].agg(lambda s: np.percentile(s, 15))
    binned['upper_sigma'] = grouped['rel_error'].agg(lambda s: np.percentile(s, 85))
    binned['resolution_quantiles'] = (binned.upper_sigma - binned.lower_sigma) / 2
    binned['resolution'] = grouped['rel_error'].std()

    ax.errorbar(
        binned['center'],
        binned['bias'],
        xerr=0.5 * binned['width'],
        label='Bias',
        linestyle='',
    )

    if std is False:
        ax.errorbar(
            binned['center'],
            binned['resolution_quantiles'],
            xerr=0.5 * binned['width'],
            label='Resolution',
            linestyle='',
        )
    else:
        ax.errorbar(
            binned['center'],
            binned['resolution'],
            xerr=0.5 * binned['width'],
            label='Resolution',
            linestyle='',
        )

    ax.legend()
    ax.set_xscale('log')

    return ax


if __name__ == '__main__':
    main()
