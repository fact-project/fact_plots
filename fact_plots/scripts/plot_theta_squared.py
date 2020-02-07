import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML
from dateutil.parser import parse as parse_date

from fact.io import read_h5py
from fact.analysis import li_ma_significance
import click

from ..plotting import add_preliminary
from ..time import read_timestamp

yaml = YAML(typ='safe')

plot_config = {
    'xlabel': r'$(\theta \,\, / \,\, {}^\circ )^2$',
    'preliminary_position': 'right',
    'preliminary_size': 20,
    'preliminary_color': 'lightgray',
    'legend_loc': 'center right'
}

columns = [
    'theta_deg',
    'theta_deg_off_1',
    'theta_deg_off_2',
    'theta_deg_off_3',
    'theta_deg_off_4',
    'theta_deg_off_5',
]

tex = plt.rcParams['text.usetex'] or (plt.get_backend() == 'pgf')

stats_box_template = r'''Source: {source}, $t_{{\mathrm{{obs}}}} = {t_obs:.1f}\,\mathrm{{h}}$
$N_{{\mathrm{{On}}}} = {n_on}$, $N_{{\mathrm{{Off}}}} = {n_off}$, $\alpha = {alpha}$
$N_{{\mathrm{{Exc}}}} = {n_excess:.1f} \pm {n_excess_err:.1f}$, $S_{{\mathrm{{Li&Ma}}}} = {significance:.1f}\,\sigma$
'''

if tex:
    stats_box_template = stats_box_template.replace('&', '\&')


@click.command()
@click.argument('data_path')
@click.option('--threshold', type=float, help='prediction threshold', default=0.8, show_default=True)
@click.option('--theta2-cut', type=float, help='cut for theta^2 in deg^2', default=0.03, show_default=True)
@click.option('--key', help='Key for the hdf5 group', default='events')
@click.option('--bins', help='Number of bins in the histogram', default=40, show_default=True)
@click.option('--alpha', help='Ratio of on vs off region', default=0.2, show_default=True)
@click.option('--start', help='First timestamp to consider', type=parse_date)
@click.option('--end', help='last timestamp to consider', type=parse_date)
@click.option('--preliminary', is_flag=True, help='Add preliminary')
@click.option('--ymax', type=float, help='The upper ylim')
@click.option('-c', '--config', help='Path to yaml config file')
@click.option('-o', '--output', help='(optional) Output file for the plot')
def main(data_path, threshold, theta2_cut, key, bins, alpha, start, end, preliminary, ymax, config, output):
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
        'theta',
        'theta_deg_off_1',
        'theta_deg_off_2',
        'theta_deg_off_3',
        'theta_deg_off_4',
        'theta_deg_off_5',

    If a prediction threshold is to be used, also 'gamma_prediction',
    must be in the group.
    The 'gamma_prediction' column can be added to the data using
    'klaas_apply_separation_model' for example.
    '''
    if config:
        with open(config) as f:
            plot_config.update(yaml.safe_load(f))

    theta_cut = np.sqrt(theta2_cut)

    if threshold > 0.0:
        columns.append('gamma_prediction')

    events = read_h5py(data_path, key='events', columns=columns)

    if start or end:
        events['timestamp'] = read_timestamp(data_path)

    try:
        runs = read_h5py(data_path, key='runs')
        runs['run_start'] = pd.to_datetime(runs['run_start'])
        runs['run_stop'] = pd.to_datetime(runs['run_stop'])
    except IOError:
        runs = pd.DataFrame(columns=['run_start', 'run_stop', 'ontime', 'source'])

    if start is not None:
        events = events.query('timestamp >= @start')
        runs = runs.query('run_start >= @start')
    if end is not None:
        events = events.query('timestamp <= @end')
        runs = runs.query('run_stop <= @end')

    if threshold > 0:
        selected = events.query('gamma_prediction >= {}'.format(threshold))
    else:
        selected = events
    theta_on = selected.theta_deg
    theta_off = pd.concat([
        selected['theta_deg_off_{}'.format(i)]
        for i in range(1, 6)
    ])

    del events

    max_theta2 = 0.3
    width = max_theta2 / bins
    rounded_width = theta2_cut / np.round(theta2_cut / width)
    bins = np.arange(0, max_theta2 + 0.1 * rounded_width, rounded_width)

    print('Using {} bins to get theta_cut on a bin edge'.format(len(bins) - 1))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    h_on, bin_edges = np.histogram(
        theta_on.apply(lambda x: x**2).values,
        bins=bins,
    )
    h_off, bin_edges, _ = ax.hist(
        theta_off.apply(lambda x: x**2).values,
        bins=bin_edges,
        weights=np.full(len(theta_off), 0.2),
        histtype='stepfilled',
        color='lightgray',
        zorder=0,
    )

    bin_center = bin_edges[1:] - np.diff(bin_edges) * 0.5
    bin_width = np.diff(bin_edges)

    ax.errorbar(
        bin_center,
        h_on,
        yerr=np.sqrt(h_on),
        xerr=bin_width / 2,
        linestyle='',
        label='On',
    )

    ax.errorbar(
        bin_center,
        h_off,
        yerr=alpha * np.sqrt(h_off),
        xerr=bin_width / 2,
        linestyle='',
        label='Off',
        zorder=1
    )

    ax.axvline(theta_cut**2, color='black', alpha=0.3, linestyle='--')

    n_on = np.sum(theta_on < theta_cut)
    n_off = np.sum(theta_off < theta_cut)
    significance = li_ma_significance(n_on, n_off, alpha=alpha)

    print('N_on', n_on)
    print('N_off', n_off)
    print('Li&Ma: {}'.format(significance))

    ax.text(
        0.5, 0.95,
        stats_box_template.format(
            source=runs.source.iloc[0] if len(runs) > 0 else '',
            t_obs=runs.ontime.sum() / 3600,
            n_on=n_on, n_off=n_off, alpha=alpha,
            n_excess=n_on - alpha * n_off,
            n_excess_err=np.sqrt(n_on + alpha**2 * n_off),
            significance=significance,
        ),
        transform=ax.transAxes,
        va='top',
        ha='center',
    )

    if preliminary:
        add_preliminary(
            plot_config['preliminary_position'],
            size=plot_config['preliminary_size'],
            color=plot_config['preliminary_color'],
            ax=ax,
        )

    if ymax:
        ax.set_ylim(0, ymax)

    ax.set_xlim(0, bins.max())
    ax.set_xlabel(plot_config['xlabel'])
    ax.legend(loc=plot_config['legend_loc'])
    fig.tight_layout(pad=0)

    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    main()
