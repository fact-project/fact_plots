from fact.io import read_h5py
from fact.analysis.statistics import (
    calc_weights_cosmic_rays,
    calc_weights_powerlaw,
    calc_weights_logparabola,
)
import astropy.units as u
import numpy as np
import click
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from collections import OrderedDict
from tqdm import tqdm
from fnmatch import fnmatch

if plt.get_backend() == 'pgf':
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


ETRUE = 'corsika_event_header_total_energy'
yaml = YAML(typ='safe')


def wrap_angle(angle):
    angle = np.asanyarray(angle).copy()
    while np.any(angle < 0):
        angle[angle < 0] += 360

    while np.any(angle > 360):
        angle[angle > 360] -= 360

    return angle


def calc_limis(arrays):
    '''Calculate axis limits, try go get a nice range for visualization'''
    flat = []
    for data in arrays:
        if isinstance(data, dict):
            flat.extend(data.values())
        else:
            flat.append(data)

    min_x = min(np.nanmin(a) for a in flat)
    max_x = max(np.nanmax(a) for a in flat)
    p1 = min(np.nanpercentile(a, 0.1) for a in flat)
    p99 = max(np.nanpercentile(a, 99.9) for a in flat)

    r = max_x - min_x

    if abs(max_x - p99) > (0.1 * r):
        max_x = p99

    if abs(min_x - p1) > (0.1 * r):
        min_x = p1

    limits = [min_x, max_x]

    return limits


def unity(x):
    return x


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

    if transform is None:
        transform = unity

    trans = {}
    for k, data in dfs.items():
        if isinstance(data, dict):
            trans[k] = dict()
            for p, df in data.items():
                trans[k][p] = transform(df[key].values)
        else:
            trans[k] = transform(data[key].values)

    if limits is None:
        limits = calc_limis(trans.values())

    if transform is np.log10 and xlabel is None:
        xlabel = 'log10(' + key + ')'

    for label, transformed in trans.items():
        if isinstance(transformed, dict):
            total = np.concatenate(list(transformed.values()))
            weights_total = np.concatenate([dfs[label][p]['weight'] for p in transformed.keys()])
            ax.hist(
                total,
                bins=n_bins,
                range=limits,
                weights=weights_total,
                label=label + ' combined',
                histtype='step',
            )

            for part, values in transformed.items():
                ax.hist(
                    values,
                    bins=n_bins,
                    range=limits,
                    weights=dfs[label][part]['weight'],
                    label='  ' + part,
                    histtype='step',
                    alpha=0.5,
                )

        else:
            ax.hist(
                transformed,
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


def calc_weights(dataset):
    if dataset['kind'] == 'observations':
        runs = read_h5py(dataset['path'], key='runs', columns=['ontime'])
        ontime = runs['ontime'].sum() / 3600
        return 1 / ontime

    if dataset['kind'] == 'protons':
        energy = read_h5py(
            dataset['path'], key='events', columns=[ETRUE]
        )[ETRUE]

        return calc_weights_cosmic_rays(
            energy=u.Quantity(energy.values, u.GeV, copy=False),
            obstime=1 * u.hour,
            n_events=dataset['n_showers'],
            e_min=dataset['e_min'] * u.GeV,
            e_max=dataset['e_max'] * u.GeV,
            simulated_index=dataset['spectral_index'],
            scatter_radius=dataset['max_impact'] * u.m,
            sample_fraction=dataset.get('sample_fraction', 1.0),
            viewcone=dataset['viewcone'] * u.deg,
        )

    if dataset['kind'] == 'gammas':
        k = 'corsika_event_header_total_energy'
        energy = read_h5py(
            dataset['path'], key='events', columns=[k]
        )[k]
        spectrum = dataset['spectrum']

        if spectrum['function'] == 'power_law':
            return calc_weights_powerlaw(
                energy=u.Quantity(energy.values, u.GeV, copy=False),
                obstime=1 * u.hour,
                n_events=dataset['n_showers'],
                e_min=dataset['e_min'] * u.GeV,
                e_max=dataset['e_max'] * u.GeV,
                simulated_index=dataset['spectral_index'],
                scatter_radius=dataset['max_impact'] * u.m,
                sample_fraction=dataset.get('sample_fraction', 1.0),
                flux_normalization=u.Quantity(**spectrum['phi_0']),
                e_ref=u.Quantity(**spectrum['e_ref']),
                target_index=spectrum['spectral_index'],
            )

        if spectrum['function'] == 'log_parabola':
            return calc_weights_logparabola(
                energy=u.Quantity(energy.values, u.GeV, copy=False),
                obstime=1 * u.hour,
                n_events=dataset['n_showers'],
                e_min=dataset['e_min'] * u.GeV,
                e_max=dataset['e_max'] * u.GeV,
                simulated_index=dataset['spectral_index'],
                scatter_radius=dataset['max_impact'] * u.m,
                sample_fraction=dataset.get('sample_fraction', 1.0),
                flux_normalization=u.Quantity(**spectrum['phi_0']),
                e_ref=u.Quantity(**spectrum['e_ref']),
                target_a=spectrum['a'],
                target_b=spectrum['b'],
            )

    raise ValueError('Unknown dataset kind "{}"'.format(dataset['kind']))


def update_columns(dataset_config, common_columns):
    df = read_h5py(dataset_config['path'], key='events', last=1)

    if len(common_columns) == 0:
        return set(df.columns)
    return common_columns.intersection(df.columns)


def read_dfs_for_column(datasets, column):
    dfs = OrderedDict()
    for dataset in datasets:
        l = dataset['label']
        if 'parts' in dataset:
            dfs[l] = {}
            for part in dataset['parts']:
                dfs[l][part['label']] = read_h5py(
                    part['path'], key='events', columns=[column]
                )
        else:
            dfs[l] = read_h5py(dataset['path'], key='events', columns=[column])
    return dfs


def add_weights(dfs, weights):
    for l, data in dfs.items():
        if isinstance(data, dict):
            for p, df in data.items():
                df['weight'] = weights[l][p]
        else:
            data['weight'] = weights[l]


@click.command()
@click.argument('config')
@click.argument('outputfile')
def main(config, outputfile):

    with open(config) as f:
        config = yaml.load(f)

    n_bins = config.get('n_bins', 100)

    # get columns available in all datasets and calculate weights
    weights = OrderedDict()
    common_columns = set()
    for dataset in config['datasets']:
        l = dataset['label']
        if 'parts' in dataset:
            weights[dataset['label']] = {}
            for part in dataset['parts']:
                common_columns = update_columns(part, common_columns)
                weights[l][part['label']] = calc_weights(part)
        else:
            common_columns = update_columns(dataset, common_columns)
            weights[l] = calc_weights(dataset)

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

    with PdfPages(outputfile) as pdf:
        for i, column in enumerate(tqdm(columns)):

            kwargs = config.get('columns').get(column, {})
            kwargs['n_bins'] = kwargs.get('n_bins', n_bins)
            if 'transform' in kwargs:
                kwargs['transform'] = eval(kwargs['transform'])

            dfs = read_dfs_for_column(config['datasets'], column)
            add_weights(dfs, weights)

            if i == 0:
                for l, data in dfs.items():
                    if isinstance(data, dict):
                        print(f'{l: <15}')
                        total = 0
                        for p, df in data.items():
                            total += df['weight'].sum()
                            print(f'  {p: <15}', f'{df["weight"].sum() / 3600:6.2f} Events/s')
                        print(f'  {"total": <15}', f'{total / 3600:6.2f} Events/s')
                    else:
                        print(f'{l: <15}', f'{data["weight"].sum() / 3600:6.2f} Events/s')

            ax_hist.cla()
            try:
                plot_hists(dfs, column, ax=ax_hist, **kwargs)
            except Exception as e:
                print(f'Could not plot column {column}')
                print(e)

            fig.tight_layout(pad=0)
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
