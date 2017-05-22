import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py

from fact.io import read_h5py
from fact.analysis import (
    li_ma_significance,
    split_on_off_source_independent,
    split_on_off_source_dependent,
)
from fact.instrument import camera_distance_mm_to_deg
import click

columns = [
    'signal_prediction',
    'theta',
    'theta_off_1',
    'theta_off_2',
    'theta_off_3',
    'theta_off_4',
    'theta_off_5',
]

stats_box_template = r'''Source: {source}, $t_\mathrm{{obs}} = {t_obs:.2f}\,\mathrm{{h}}$
$N_\mathrm{{On}} = {n_on}$, $N_\mathrm{{off}} = {n_off}$, $\alpha = {alpha}$
$S_\mathrm{{Li&Ma}} = {significance:.1f}\,\sigma$
'''


@click.command()
@click.argument('data_path')
@click.option('--threshold', type=float, help='prediction threshold', default=0.8, show_default=True)
@click.option('--theta2-cut', type=float, help='cut for theta^2 in deg^2', default=0.03, show_default=True)
@click.option('--key', help='Key for the hdf5 group', default='events')
@click.option('--bins', help='Number of bins in the histogram', default=40, show_default=True)
@click.option('--alpha', help='Ratio of on vs off region', default=0.2, show_default=True)
@click.option('-o', '--output', help='(optional) Output file for the plot')
def main(data_path, threshold, theta2_cut, key, bins, alpha, output):
    '''
    Given the DATA_PATH to a data hdf5 file (e.g. the output of ERNAs gather scripts)
    this script will create the infamous theta square plot.

    This plot shows the events of (selected gamma-like) events which have been
    reconstructed as coming from the source region and the one coming from a
    (more or less abritrary) off region.

    In a traditional IACT analysis this plot is used to calculate the significance of
    detection.

    The HDF files are expected to a have a group called 'runs' and a group called 'events'
    The events group has to have the columns:
        'gamma_prediction',
        'theta',
        'theta_off_1',
        'theta_off_2',
        'theta_off_3',
        'theta_off_4',
        'theta_off_5',

    The 'gamma_prediction' column can be added to the data using
    'klaas_apply_separation_model' for example.
    '''
    theta_cut = np.sqrt(theta2_cut)

    with h5py.File(data_path, 'r') as f:
        source_dependent = 'background_prediction_1' in f[key].keys()

    if source_dependent:
        print('Separation was using source dependent features')
        columns.extend('background_prediction_' + str(i) for i in range(1, 6))
        theta_cut = np.inf
        theta2_cut = np.inf

    events = read_h5py(data_path, key='events', columns=columns)
    runs = read_h5py(data_path, key='runs')

    for i in range(6):
        col = 'theta' if i == 0 else 'theta_off_{}'.format(i)
        events[col + '_deg'] = camera_distance_mm_to_deg(events[col])

    if source_dependent:
        on_data, off_data = split_on_off_source_dependent(events, threshold)
        theta_on = on_data.theta_deg
        theta_off = off_data.theta_deg
    else:
        selected = events.query('signal_prediction >= {}'.format(threshold))
        theta_on = selected.theta_deg
        theta_off = pd.concat([
            selected['theta_off_{}_deg'.format(i)]
            for i in range(1, 6)
        ])

    del events

    if source_dependent:
        limits = [
            0,
            max(np.percentile(theta_on, 99)**2, np.percentile(theta_off, 99)**2),
        ]
    else:
        limits = [0, 0.3]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    h_on, bin_edges = np.histogram(
        theta_on.apply(lambda x: x**2).values,
        bins=bins,
        range=limits
    )
    h_off, bin_edges, _ = ax.hist(
        theta_off.apply(lambda x: x**2).values,
        bins=bin_edges,
        range=limits,
        weights=np.full(len(theta_off), 0.2),
        histtype='stepfilled',
        color='lightgray',
    )

    bin_center = bin_edges[1:] - np.diff(bin_edges)* 0.5
    bin_width = np.diff(bin_edges)

    ax.errorbar(bin_center, h_on, yerr=np.sqrt(h_on)/2, xerr=bin_width/2, elinewidth=1, fmt='none', label='on')
    ax.errorbar(bin_center, h_off, yerr=alpha * np.sqrt(h_off)/2, xerr=bin_width/2, elinewidth=1, fmt='none', label='off')

    if not source_dependent:
        ax.axvline(theta_cut**2, color='gray', linestyle='--')

    n_on = np.sum(theta_on < theta_cut)
    n_off = np.sum(theta_off < theta_cut)
    significance = li_ma_significance(n_on, n_off, alpha=alpha)

    print('N_on', n_on)
    print('N_off', n_off)
    print('Li&Ma: {}'.format(significance))

    ax.text(
        0.5, 0.95,
        stats_box_template.format(
            source=runs.source.loc[0],
            t_obs=runs.ontime.sum() / 3600,
            n_on=n_on, n_off=n_off, alpha=0.2,
            significance=significance,
        ),
        transform=ax.transAxes,
        fontsize=12,
        va='top',
        ha='center',
    )

    ax.set_xlabel(r'$(\theta / {}^\circ )^2$')
    ax.legend()
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    main()
