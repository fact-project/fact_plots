import click
import pandas as pd
import numpy as np
from fact.io import read_h5py
from ..plotting import add_preliminary
import matplotlib.pyplot as plt
import yaml


plot_config = {
    'xlabel': r'$\log_{10}(E_\mathrm{true} \,\, / \,\, \mathrm{GeV})$',
    'ylabel': r'$\theta_{0.68} \,\, / \,\, ^\circ$',
    'preliminary_position': 'upper left',
    'preliminary_size': 20,
    'preliminary_color': 'lightgray',
}


@click.command()
@click.argument('gamma_path')
@click.option(
    '--std', default=False, is_flag=True,
    help='Use std instead of inter-percentile distance',
)
@click.option('--n-bins', default=20, type=int)
@click.option('--threshold', type=float)
@click.option('-c', '--config', help='Path to yaml config file')
@click.option('-o', '--output')
@click.option('--preliminary', is_flag=True, help='add preliminary')
def main(gamma_path, std, n_bins, threshold, config, output, preliminary):
    if config:
        with open(config) as f:
            plot_config.update(yaml.safe_load(f))

    df = read_h5py(
        gamma_path,
        key='events',
        columns=[
            'corsika_evt_header_total_energy',
            'gamma_energy_prediction',
            'gamma_prediction',
            'theta_deg'
        ],
    )

    if threshold:
        df = df.query('gamma_prediction >= @threshold').copy()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()

    if preliminary:
        add_preliminary(
            plot_config['preliminary_position'],
            size=plot_config['preliminary_size'],
            color=plot_config['preliminary_color'],
            ax=ax,
        )

    plot_angular_resolution(df, n_bins=n_bins, ax=ax)
    ax.set_xlabel(plot_config['xlabel'])
    ax.set_ylabel(plot_config['ylabel'])

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()


def plot_angular_resolution(
        df,
        n_bins=10,
        ax=None,
        theta_key='theta_deg',
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

    binned = pd.DataFrame(index=np.arange(1, len(bins)))
    binned['center'] = 0.5 * (bins[:-1] + bins[1:])
    binned['width'] = np.diff(bins)

    grouped = df.groupby('bin')

    values = []
    for i in range(100):
        sampled = df.sample(len(df), replace=True).groupby('bin')
        resolution = np.full(n_bins, np.nan)
        s = sampled[theta_key].agg(lambda s: np.percentile(s.values, 68))

        resolution[s.index.values - 1] = s.values
        values.append(resolution)

    binned['angular_resolution'] = np.nanmean(values, axis=0)
    binned['angular_resolution_err'] = np.nanstd(values, axis=0)
    binned['size'] = grouped.size()

    binned = binned.query('size > 200')

    ax.errorbar(
        binned['center'],
        binned['angular_resolution'],
        xerr=0.5 * binned['width'],
        yerr=binned['angular_resolution_err'],
        linestyle='',
    )

    ax.set_xscale('log')

    return ax


if __name__ == '__main__':
    main()
