from fact.io import read_data, read_simulated_spectrum
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import yaml
import h5py
import click

from ..plotting import add_preliminary
from ..effective_area import plot_effective_area


plot_config = {
    'xlabel': r'$E_\mathrm{true} \,\,/\,\, \mathrm{GeV}$',
    'ylabel': r'$A_\mathrm{eff} \,\,/\,\, \mathrm{m}^2$',
    'preliminary_position': 'upper left',
    'preliminary_size': 20,
    'preliminary_color': 'lightgray',
}


@click.command()
@click.argument('CORSIKA_HEADERS')
@click.argument('ANALYSIS_OUTPUT')
@click.option('-f', '--fraction', type=float, help='Sample fraction for all_events')
@click.option('-t', '--threshold', type=float, default=[0.8], multiple=True, help='Prediction threshold to use')
@click.option('--theta2-cut', type=float, default=[0.03], multiple=True, help='Theta squared cut to use')
@click.option('--n-bins', type=int, default=20,  help='Number of bins for the area')
@click.option('--e-low', type=float, help='Lower energy limit in GeV')
@click.option('--e-high', type=float, help='Upper energy limit in GeV')
@click.option(
    '-i',
    '--impact',
    help='the maximum impact parameter used for the corsika simulations (in meter) '
)
@click.option('-c', '--config', help='Path to yaml config file')
@click.option('-o', '--output')
@click.option('--preliminary', is_flag=True, help='add preliminary')
def main(
    corsika_headers,
    analysis_output,
    fraction,
    threshold,
    theta2_cut,
    n_bins,
    impact,
    config,
    output,
    preliminary,
    e_low,
    e_high
):
    if config:
        with open(config) as f:
            plot_config.update(yaml.safe_load(f))

    all_events = read_data(corsika_headers, key='corsika_events')


    analysed = read_data(
        analysis_output,
        key='events',
        columns=[
            'corsika_event_header_total_energy',
            'gamma_prediction',
            'theta_deg'
        ]
    )

    if fraction is None:
        with h5py.File(analysis_output, 'r') as f:
            fraction = f.attrs.get('sample_fraction', 1.0)
            print('Using a sample fraction of', fraction)

    if impact is None:
        simulated_spectrum = read_simulated_spectrum(corsika_headers)
        impact = simulated_spectrum['x_scatter']
    else:
        impact = impact * u.m

    e_low = e_low or all_events.total_energy.min()
    e_high = e_high or all_events.total_energy.max()
    bins = np.logspace(np.log10(e_low), np.log10(e_high), n_bins + 1)

    assert len(theta2_cut) == len(threshold), 'Number of cuts has to be the same for theta and threshold'

    for threshold, theta2_cut in zip(threshold[:], theta2_cut[:]):
        selected = analysed.query(
            '(gamma_prediction >= @threshold) & (theta_deg**2 <= @theta2_cut)'
        ).copy()

        label = r'$p_\gamma \geq {}$'.format(threshold)
        if theta2_cut != np.inf:
            label += r', $\theta^2 \leq {:.3g}\,\mathrm{{deg}}^2$'.format(theta2_cut)

        plot_effective_area(
            all_events.total_energy,
            selected.corsika_event_header_total_energy,
            bins=bins,
            impact=impact,
            sample_fraction=fraction,
            label=label,
        )

    if preliminary:
        add_preliminary(
            plot_config['preliminary_position'],
            size=plot_config['preliminary_size'],
            color=plot_config['preliminary_color'],
        )

    plt.legend()
    plt.xlabel(plot_config['xlabel'])
    plt.ylabel(plot_config['ylabel'])

    plt.yscale('log')
    plt.xscale('log')

    plt.tight_layout(pad=0.02)
    if output is not None:
        plt.savefig(output, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    main()
