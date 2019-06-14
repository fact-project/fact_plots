from fact.io import read_h5py
from fact.analysis.statistics import calc_proton_obstime, calc_gamma_obstime
import astropy.units as u
import numpy as np
import click
import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
from tqdm import tqdm
from fnmatch import fnmatch


def wrap_angle(angle):
    angle = np.asanyarray(angle).copy()
    while np.any(angle < 0):
        angle[angle < 0] += 360

    while np.any(angle > 360):
        angle[angle > 360] -= 360

    return angle


def calc_limis(arrays):
    '''Calculate axis limits, try go get a nice range for visualization'''
    min_x = min(np.nanmin(a) for a in arrays)
    max_x = max(np.nanmax(a) for a in arrays)
    p1 = min(np.nanpercentile(a, 0.1) for a in arrays)
    p99 = max(np.nanpercentile(a, 99.9) for a in arrays)

    r = max_x - min_x

    if abs(max_x - p99) > (0.1 * r):
        max_x = p99

    if abs(min_x - p1) > (0.1 * r):
        min_x = p1

    limits = [min_x, max_x]

    return limits


def plot_hists(
    dfs,
    key,
    n_bins=100,
    limits=None,
    transform=None,
    xlabel=None,
    yscale='linear',
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    trans = {}
    for k, df in dfs.items():
        if transform is None:
            trans[k] = df[key]
        else:
            trans[k] = transform(df[key].values)

    if limits is None:
        limits = calc_limis(trans.values())

    if transform is np.log10 and xlabel is None:
        xlabel = 'log10(' + key + ')'

    for label, t in trans.items():
        ax.hist(
            t,
            bins=n_bins,
            range=limits,
            weights=dfs[label]['weight'],
            label=label,
            histtype='step',
        )

    ax.set_ylabel('Events / h')
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel or key)
    ax.legend()


@click.command()
@click.argument('config')
@click.argument('outputfile')
def main(config, outputfile):

    with open(config) as f:
        config = yaml.safe_load(f)

    # get columns available in all datasets and calculate weights
    weights = OrderedDict()
    for i, dataset in enumerate(config['datasets']):
        l = dataset['label']
        df = read_h5py(dataset['path'], key='events', last=1)

        if i == 0:
            common_columns = set(df.columns)
        else:
            common_columns = common_columns.intersection(df.columns)

        if dataset['kind'] == 'observations':
            runs = read_h5py(dataset['path'], key='runs', columns=['ontime'])
            ontime = runs['ontime'].sum() / 3600
            weights[l] = 1 / ontime

        elif dataset['kind'] == 'protons':

            sample_fraction = dataset.get('sample_fraction', 1.0)
            ontime = calc_proton_obstime(
                n_events=float(dataset['n_showers']),
                spectral_index=dataset['spectral_index'],
                max_impact=dataset['max_impact'] * u.m,
                viewcone=dataset['viewcone'] * u.deg,
                e_min=float(dataset['e_min']) * u.GeV,
                e_max=float(dataset['e_max']) * u.GeV,
            )
            weights[l] = 1 / (ontime.to_value(u.hour) * sample_fraction)

        elif dataset['kind'] == 'gammas':

            norm = dataset['phi_0']
            norm = u.Quantity(norm['value'], norm['unit'])
            sample_fraction = dataset.get('sample_fraction', 1.0)
            ontime = calc_gamma_obstime(
                n_events=float(dataset['n_showers']),
                spectral_index=dataset['spectral_index'],
                max_impact=dataset['max_impact'] * u.m,
                e_min=float(dataset['e_min']) * u.GeV,
                e_max=float(dataset['e_max']) * u.GeV,
                flux_normalization=norm,
            )
            weights[l] = 1 / (ontime.to_value(u.hour) * sample_fraction)

    # select columns
    columns = config.get('include_columns')
    if columns is not None:
        def included(column):
            return any(
                fnmatch(column, include)
                for include in columns
            )
        common_columns = list(filter(included, common_columns))

    columns = sorted(list(common_columns), key=str.lower)

    # exclude columns using glob pattern
    if config.get('exclude_columns') is not None:
        def excluded(column):
            return not any(
                fnmatch(column, exclude)
                for exclude in config['exclude_columns']
            )
        columns = list(filter(excluded, columns))

    fig = plt.figure()
    ax_hist = fig.add_subplot(1, 1, 1)

    index_weights = {}

    with PdfPages(outputfile) as pdf:
        for i, column in enumerate(tqdm(columns)):

            kwargs = config.get('columns').get(column, {})
            if 'transform' in kwargs:
                kwargs['transform'] = eval(kwargs['transform'])

            dfs = OrderedDict()
            for dataset in config['datasets']:
                l = dataset['label']
                dfs[l] = read_h5py(dataset['path'], key='events', columns=[column])
                dfs[l]['weight'] = weights[l]

                if dataset['kind'] == 'protons':
                    if not np.isclose(dataset['spectral_index'], -2.7):

                        if index_weights.get(l) is None:
                            print('Reweighting protons')
                            k = 'corsika_event_header_total_energy'
                            energy = read_h5py(
                                dataset['path'], key='events', columns=[k]
                            )[k]

                            index_weights[l] = calc_weight_change_index(
                                u.Quantity(energy, u.GeV, copy=False),
                                simulated_index=dataset['spectral_index'],
                                target_index=-2.7,
                                e_ref=1 * u.GeV,
                            ).to_value(u.dimensionless_unscaled)

                        dfs[l]['weight'] *= index_weights[l]


            if i == 0:
                for l, df in dfs.items():
                    print(f'{l: <15}', f'{df["weight"].sum() / 3600:.1f} Events/s')

            ax_hist.cla()
            try:
                plot_hists(dfs, column, ax=ax_hist, **kwargs)
            except Exception as e:
                print(f'Could not plot column {column}')
                print(e)

            pdf.savefig(fig)


if __name__ == '__main__':
    main()
