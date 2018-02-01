from fact import analysis
import fact.analysis.binning
from fact import plotting
from fact.io import read_h5py
import pandas as pd
from dateutil.parser import parse
import numpy as np

import h5py
import matplotlib.pyplot as plt
import click
from functools import partial

from ..plotting import add_preliminary


plot_config = {
    'xlabel': r'$(\theta \,\, / \,\, {}^\circ )^2$',
    'preliminary_position': 'lower right',
    'preliminary_size': 20,
    'preliminary_color': 'lightgray',
}


columns = [
    'gamma_prediction',
    'theta_deg',
    'theta_deg_off_1',
    'theta_deg_off_2',
    'theta_deg_off_3',
    'theta_deg_off_4',
    'theta_deg_off_5',
    'run_id',
    'night',
]


@click.command()
@click.argument('data_path')
@click.option('--threshold', type=float, help='prediction threshold', default=0.8, show_default=True)
@click.option('--theta2-cut', type=float, help='cut for theta^2 in deg^2', default=0.03, show_default=True)
@click.option('--key', help='Key for the hdf5 group', default='events')
@click.option('--binning', help='Ontime in one bin in minutes', default='20', show_default=True)
@click.option('--alpha', help='Ratio of on vs off region', default=0.2, show_default=True)
@click.option('--start', help='Date of first observation YYYY-MM-DD HH:SS or anything parseable by dateutil')
@click.option('--end', help='Date of first observation YYYY-MM-DD HH:SS or anything parseable by dateutil')
@click.option('-f', '--ontime-fraction', default=0.90, help='Discard bins with less ontime than fraction * binning')
@click.option('--preliminary', is_flag=True, help='Add preliminary')
@click.option('-o', '--output', help='(optional) output file for the plot')
def main(data_path, threshold, theta2_cut, key, binning, alpha, start, end, ontime_fraction, preliminary, output):
    '''
    Given the DATA_PATH to a data hdf5 file (e.g. the output of ERNAs gather scripts)
    this script will create a plot of excess rates over time.

    The HDF files are expected to a have a group called 'runs' and a group called 'events'
    The events group has to have the columns:
        'gamma_prediction',
        'theta_deg',
        'theta_deg_off_1',
        'theta_deg_off_2',
        'theta_deg_off_3',
        'theta_deg_off_4',
        'theta_deg_off_5',
        'run_id' and
        'night'.

    The 'gamma_prediction' column can be added to the data using
    'klaas_apply_separation_model' for example.
    '''

    with h5py.File(data_path, 'r') as f:
        source_dependent = 'gamma_prediction_off_1' in f[key].keys()

    if source_dependent:
        print('Separation was using source dependent features')
        columns.extend('gamma_prediction_off_' + str(i) for i in range(1, 6))
        theta2_cut = np.inf

    events = read_h5py(data_path, key='events', columns=columns)

    runs = read_h5py(data_path, key='runs')
    runs['run_start'] = pd.to_datetime(runs['run_start'])
    runs['run_stop'] = pd.to_datetime(runs['run_stop'])

    if start:
        start = parse(start)
        runs = runs.query('run_start >= @start')
    if end:
        end = parse(end)
        runs = runs.query('run_stop <= @end')

    if source_dependent:
        summary = analysis.calc_run_summary_source_dependent(
            events,
            runs,
            prediction_threshold=threshold,
        )
    else:
        summary = analysis.calc_run_summary_source_independent(
            events,
            runs,
            prediction_threshold=threshold,
            theta2_cut=theta2_cut,
        )

    if binning == 'nightly':
        binning_function = fact.analysis.binning.nightly_binning
    else:
        try:
            binning = float(binning)
        except ValueError:
            click.abort('--binning must be float or "nightly"')

        binning_function = partial(
            fact.analysis.binning.ontime_binning,
            bin_width_minutes=binning
        )

    bins = analysis.bin_runs(summary, alpha=alpha, binning_function=binning_function)
    if isinstance(binning, float):
        bins = bins.query('ontime >= (@ontime_fraction * @binning * 60)')

    ax_exc, ax_sig, ax_mjd = plotting.analysis.plot_excess_rate(bins)

    if preliminary:
        add_preliminary(
            plot_config['preliminary_position'],
            size=plot_config['preliminary_size'],
            color=plot_config['preliminary_color'],
            ax=ax_exc,
        )

    plt.tight_layout(pad=0)

    if output:
        plt.savefig(output, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    main()
