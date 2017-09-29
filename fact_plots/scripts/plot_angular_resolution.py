import click
from fact.io import read_h5py
from ..plotting import add_preliminary
from ..angular_resolution import plot_angular_resolution
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
    '''
    Plot the 68% containment radius for different energy bins

    ARGUMENTS

    GAMMA_PATH: HDF5 file for simulated gamma rays containing the keys
        * gamma_prediction
        * theta_deg
        * corsika_evt_header_total_energy
    '''
    if config:
        with open(config) as f:
            plot_config.update(yaml.safe_load(f))

    df = read_h5py(
        gamma_path,
        key='events',
        columns=[
            'corsika_evt_header_total_energy',
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

    fig.tight_layout(pad=0)

    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    main()
