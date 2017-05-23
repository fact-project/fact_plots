from fact import analysis, instrument
from fact import  plotting
from fact.io import read_h5py
from fact.instrument import camera_distance_mm_to_deg
import pandas as pd
from dateutil.parser import parse

import h5py
import matplotlib.pyplot as plt
import click


columns = [
    'gamma_prediction',
    'theta',
    'theta_off_1',
    'theta_off_2',
    'theta_off_3',
    'theta_off_4',
    'theta_off_5',
    'run_id',
    'night',
]

@click.command()
@click.argument('data_path')
@click.option('--threshold', type=float, help='prediction threshold', default=0.8, show_default=True)
@click.option('--theta2-cut', type=float, help='cut for theta^2 in deg^2', default=0.03, show_default=True)
@click.option('--key', help='Key for the hdf5 group', default='events')
@click.option('--binning', help='Ontime in one bin in minutes', default=20, show_default=True)
@click.option('--alpha', help='Ratio of on vs off region', default=0.2, show_default=True)
@click.option('--start', help='Date of first observation YYYY-MM-DD HH:SS or anything parseable by dateutil')
@click.option('--end', help='Date of first observation YYYY-MM-DD HH:SS or anything parseable by dateutil')
@click.option('-o', '--output', help='(optional) output file for the plot')
def main(data_path, threshold, theta2_cut, key, binning, alpha, start, end, output):
    '''
    Given the DATA_PATH to a data hdf5 file (e.g. the output of ERNAs gather scripts)
    this script will create a plot of excess rates over time.

    The HDF files are expected to a have a group called 'runs' and a group called 'events'
    The events group has to have the columns:
        'gamma_prediction',
        'theta',
        'theta_off_1',
        'theta_off_2',
        'theta_off_3',
        'theta_off_4',
        'theta_off_5',
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
        theta_cut = np.inf
        theta2_cut = np.inf


    events = read_h5py(data_path, key='events', columns=columns)
    # events = events.rename(columns={'RUNID':'run_id', 'NIGHT':'night'})

    runs = read_h5py(data_path, key='runs')
    runs['run_start'] = pd.to_datetime(runs['run_start'])
    runs['run_stop'] = pd.to_datetime(runs['run_stop'])

    if start:
        start = parse(start)
        runs = runs.query('run_start >= @start')
    if end:
        end = parse(end)
        runs = runs.query('run_stop <= @end')

    for i in range(6):
        col = 'theta' if i == 0 else 'theta_off_{}'.format(i)
        events[col + '_deg'] = camera_distance_mm_to_deg(events[col])


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

    def f(df):
        return analysis.ontime_binning(df, bin_width_minutes=binning)

    d = analysis.bin_runs(summary, alpha=alpha, binning_function=f)

    ax = plotting.analysis.plot_excess_rate(d)
    if output:
        plt.savefig(output, dpi=300)
    else:
        plt.show()

if __name__ == '__main__':
    main()
