import click
import pandas as pd
import numpy as np
from fact.io import read_h5py
from ..plotting import add_preliminary
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml


plot_config = {
    'xlabel': r'$\log_{10}(E_\mathrm{true} \,\, / \,\, \mathrm{GeV})$',
    'preliminary_position': 'upper right',
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
@click.option('--theta2-cut', type=float)
@click.option('-c', '--config', help='Path to yaml config file')
@click.option('-o', '--output')
@click.option('--preliminary', is_flag=True, help='add preliminary')
def main(gamma_path, std, n_bins, threshold, theta2_cut, config, output, preliminary):
    if config:
        with open(config) as f:
            plot_config.update(yaml.safe_load(f))

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

    if preliminary:
        add_preliminary(
            plot_config['preliminary_position'],
            size=plot_config['preliminary_size'],
            color=plot_config['preliminary_color'],
            ax=ax,
        )

    ax_bias, ax_res = plot_bias_resolution(df, n_bins=n_bins, std=std, ax=ax)

    ax_bias.set_xlabel(plot_config['xlabel'])

    ax_bias.set_ylabel('Bias', color='C0')
    ax_res.set_ylabel('Resolution', color='C1')

    ax_res.set_ylim(*ax_bias.get_ylim())

    fig.tight_layout(pad=0)

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

    ax_bias = ax or plt.gca()
    ax_res = ax.twinx()

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


if __name__ == '__main__':
    main()
