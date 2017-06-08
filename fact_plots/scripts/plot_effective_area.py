# coding: utf-8
from irf.collection_area import collection_area_energy
from fact.instrument import camera_distance_mm_to_deg
import pandas as pd
from fact.io import read_data
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import h5py
import yaml

import click

from ..plotting import add_preliminary


plot_config = {
    'xlabel': r'$\log_{10}(E_\mathrm{true} \,\,/\,\, \mathrm{GeV})$',
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
@click.option('-c', '--theta2-cut', type=float, default=[0.03], multiple=True, help='Theta squared cut to use')
@click.option('--n-bins', type=int, default=20,  help='Number of bins for the area')
@click.option(
    '-i',
    '--impact',
    default=270.0,
    show_default=True,
    help='the maximum impact parameter used for the corsika simulations (in meter) '
)
@click.option('-c', '--config', help='Path to yaml config file')
@click.option('-o', '--output')
@click.option('--preliminary', is_flag=True, help='add preliminary')
def main(corsika_headers, analysis_output, fraction, threshold, theta2_cut, n_bins, impact, config, output, preliminary):
    if config:
        with open(config) as f:
            plot_config.update(yaml.safe_load(f))

    all_events = pd.read_hdf(corsika_headers, 'table')

    analysed = read_data(analysis_output, key='events')

    impact = impact * u.m

    bins = np.logspace(
        np.log10(all_events.energy.min()),
        np.log10(all_events.energy.max()),
        n_bins + 1,
    )

    with h5py.File(analysis_output, 'r') as f:
        source_dependent = 'gamma_prediction_off_1' in f['events'].keys()

    if source_dependent:
        print('Separation used source dependent features, ignoring theta cut')
        theta2_cut = np.full_like(threshold, np.inf)
    else:
        assert len(theta2_cut) == len(threshold), 'Number of cuts has to be the same for theta and threshold'

    for threshold, theta2_cut in zip(threshold[:], theta2_cut[:]):
        selected = analysed.query(
            '(gamma_prediction >= @threshold) & (theta_deg**2 <= @theta2_cut)'
        ).copy()

        ret = collection_area_energy(
            all_events, selected, bins, impact, log=False,
            sample_fraction=fraction,
        )
        area, bin_centers, bin_width, lower_conf, upper_conf = ret

        label = r'$\mathtt{{gamma\_prediction}} \geq {}$'.format(threshold)
        if theta2_cut != np.inf:
            label += r', $\theta^2 \leq {:.3g}^{{\circ^2}}$'.format(theta2_cut)

        plt.errorbar(
            bin_centers,
            area.value,
            xerr=bin_width/2,
            yerr=[
                (area - lower_conf).value,
                (upper_conf - area).value
            ],
            linestyle='',
            label=label
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

    # plt.xlim(1e2, 1e5)

    plt.tight_layout(pad=0)
    if output is not None:
        plt.savefig(output, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    main()
