import click
import numpy as np
from fact.io import read_h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ruamel.yaml import YAML

from ..plotting import add_preliminary

yaml = YAML(typ='safe')
plot_config = {
    'cmap': None,
    'logz': False,
    'xlabel': r'$E_\mathrm{true} \,\, / \,\, \mathrm{GeV}$',
    'ylabel': r'$(E_\mathrm{est} - E_\mathrm{true}) / E_\mathrm{true}$',
    'preliminary_position': 'lower right',
    'preliminary_size': 20,
    'preliminary_color': 'lightgray',
}


@click.command()
@click.argument('gamma_path')
@click.option(
    '--std', default=False, is_flag=True,
    help='Use std instead of inter-percentile distance',
)
@click.option('--n-bins', default=100, type=int)
@click.option('--threshold', type=float)
@click.option('--theta2-cut', type=float)
@click.option('--preliminary', is_flag=True, help='add preliminary')
@click.option('-c', '--config', help='Path to yaml config file')
@click.option('-o', '--output')
def main(gamma_path, std, n_bins, threshold, theta2_cut, preliminary, config, output):

    df = read_h5py(
        gamma_path,
        key='events',
        columns=[
            'gamma_energy_prediction',
            'corsika_event_header_total_energy',
            'gamma_prediction',
            'theta_deg'
        ],
    )

    if config:
        with open(config) as f:
            plot_config.update(yaml.load(f))

    if threshold:
        df = df.query('gamma_prediction >= @threshold').copy()
    if theta2_cut:
        df = df.query('theta_deg**2 <= @theta2_cut').copy()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.025)

    # ax.set_aspect(1)
    ax.set_xscale('log')
    # ax.set_yscale('log')

    e_min = df.corsika_event_header_total_energy.min()
    e_max = df.corsika_event_header_total_energy.max()

    limits = np.log10([e_min, e_max])
    xedges = np.logspace(limits[0], limits[1], n_bins + 1)
    yedges = np.linspace(-1, 5, n_bins + 1)

    e_true = df.corsika_event_header_total_energy.values
    e_est = df.gamma_energy_prediction.values
    rel_error = (e_est - e_true) / e_true

    hist, xedges, yedges = np.histogram2d(
        e_true,
        rel_error,
        bins=[xedges, yedges],
    )
    plot = ax.pcolormesh(
        xedges, yedges, hist.T,
        norm=LogNorm() if plot_config['logz'] else None,
        cmap=plot_config['cmap'],
    )
    plot.set_rasterized(True)

    fig.colorbar(plot, cax=cax)

    if preliminary:
        add_preliminary(
            plot_config['preliminary_position'],
            size=plot_config['preliminary_size'],
            color=plot_config['preliminary_color'],
            ax=ax,
        )

    ax.set_xlabel(plot_config['xlabel'])
    ax.set_ylabel(plot_config['ylabel'])

    fig.tight_layout(pad=0)

    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    main()
