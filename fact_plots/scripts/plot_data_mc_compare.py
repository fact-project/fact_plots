from fact.io import read_h5py
from fact.analysis.statistics import (
    calc_weights_cosmic_rays,
    calc_weights_powerlaw,
    calc_weights_logparabola,
    calc_weights_exponential_cutoff,
)
import astropy.units as u
import numpy as np
import click
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from collections import OrderedDict
from tqdm import tqdm
from fnmatch import fnmatch
from operator import lt, le, eq, ne, gt, ge


OPERATORS = {
    '<': lt, 'lt': lt,
    '<=': le, 'le': le,
    '==': eq, 'eq': eq,
    '=': eq,
    '!=': ne, 'ne': ne,
    '>': gt, 'gt': gt,
    '>=': ge, 'ge': ge,
}


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
        if isinstance(data, list):
            flat.extend(data)
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
    weights,
    key,
    datasets,
    n_bins=100,
    limits=None,
    transform=None,
    xlabel=None,
    yscale='linear',
    ax=None,
    legend_loc='best',
    colors=None,
):
    if ax is None:
        ax = plt.gca()

    if transform is None:
        transform = unity

    trans = []
    for d, dataset in enumerate(dfs):
        if isinstance(dataset, list):
            trans.append([])
            for part in dataset:
                trans[-1].append(transform(part[key].values))
        else:
            trans.append(transform(dfs[d].values))

    if limits is None:
        limits = calc_limis(trans)

    if transform is np.log10 and xlabel is None:
        xlabel = 'log10(' + key + ')'

    for d, dataset in enumerate(datasets):
        label = dataset['label']

        if 'parts' in dataset:
            ax.hist(
                np.concatenate(trans[d]),
                bins=n_bins,
                range=limits,
                weights=np.concatenate(weights[d]),
                label=label,
                histtype='step',
                color=dataset.get('color'),
            )

            if dataset.get('show_parts', True):
                for p, part in enumerate(dataset['parts']):
                    color = part.get('color')
                    alpha = part.get('alpha', 0.5 if not color else None)

                    ax.hist(
                        trans[d][p],
                        bins=n_bins,
                        range=limits,
                        weights=weights[d][p],
                        label=part['label'],
                        histtype='step',
                        color=color,
                        alpha=alpha
                    )

        else:
            ax.hist(
                trans[d],
                bins=n_bins,
                range=limits,
                weights=weights[d],
                label=label,
                histtype='step',
                color=dataset.get('color'),
                alpha=dataset.get('alpha', 1.0),
            )

    ax.set_ylabel('Events / h')
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel or key)
    ax.legend(loc=legend_loc)


def calc_weights(dataset, mask=None):
    # observed datasets
    if dataset['kind'] == 'observations':
        runs = read_h5py(dataset['path'], key='runs', columns=['ontime'])
        n_events = len(read_h5py(dataset['path'], key='events', columns=['event_num']))
        ontime = runs['ontime'].sum() / 3600
        return np.ones(n_events) / ontime

    # simulated datasets
    kind = dataset['kind']
    if kind in ('protons', 'gammas', 'electrons', 'helium'):

        energy = read_h5py(
            dataset['path'], key='events', columns=[ETRUE]
        )[ETRUE]

        if mask is not None:
            energy = energy[mask]

        if kind == 'gammas':
            viewcone = None
        else:
            viewcone = dataset['viewcone'] * u.deg

        spectrum = dataset.get('spectrum')
        kwargs = dict(
            energy=u.Quantity(energy.values, u.GeV, copy=False),
            obstime=1 * u.hour,
            n_events=dataset['n_showers'],
            e_min=dataset['e_min'] * u.GeV,
            e_max=dataset['e_max'] * u.GeV,
            simulated_index=dataset['spectral_index'],
            scatter_radius=dataset['max_impact'] * u.m,
            sample_fraction=dataset.get('sample_fraction', 1.0),
            viewcone=viewcone,
        )
        if spectrum is None:
            if kind == 'protons':
                return calc_weights_cosmic_rays(**kwargs).to_value(u.dimensionless_unscaled)

            raise ValueError(
                'Particle types other then protons require a "spectrum" in config'
            )

        if spectrum['function'] == 'power_law':
            return calc_weights_powerlaw(
                **kwargs,
                flux_normalization=u.Quantity(**spectrum['phi_0']),
                target_index=spectrum['spectral_index'],
                e_ref=u.Quantity(**spectrum['e_ref'])
            ).to_value(u.dimensionless_unscaled)

        if spectrum['function'] == 'log_parabola':
            return calc_weights_logparabola(
                flux_normalization=u.Quantity(**spectrum['phi_0']),
                e_ref=u.Quantity(**spectrum['e_ref']),
                target_a=spectrum['a'],
                target_b=spectrum['b'],
            ).to_value(u.dimensionless_unscaled)

        if spectrum['function'] == 'power_law_exponential_cutoff':
            return calc_weights_exponential_cutoff(
                **kwargs,
                flux_normalization=u.Quantity(**spectrum['phi_0']),
                target_index=spectrum['spectral_index'],
                target_e_cutoff=u.Quantity(**spectrum['e_cutoff']),
                e_ref=u.Quantity(**spectrum['e_ref'])
            ).to_value(u.dimensionless_unscaled)

        raise ValueError('Unknown spectral function {}'.format(spectrum['function']))

    raise ValueError('Unknown dataset kind "{}"'.format(dataset['kind']))


def update_columns(dataset_config, common_columns):
    df = read_h5py(dataset_config['path'], key='events', last=1)

    if len(common_columns) == 0:
        return set(df.columns)
    return common_columns.intersection(df.columns)


def read_dfs_for_column(datasets, column, masks=None):
    dfs = []
    for d, dataset in enumerate(datasets):
        if 'parts' in dataset:
            parts = []
            for p, part in enumerate(dataset['parts']):
                df = read_h5py(
                    part['path'], key='events', columns=[column]
                )
                if masks is not None:
                    mask = masks[d][p]
                    df = df.loc[mask].copy()
                parts.append(df)
            dfs.append(parts)
        else:
            df = read_h5py(dataset['path'], key='events', columns=[column])
            if masks is not None:
                mask = masks[d]
                df = df.loc[mask].copy()
            dfs.append(df)
    return dfs


def create_masks(config):
    masks = []
    mask_config = config['event_selection']
    for d, dataset in enumerate(config['datasets']):
        if 'parts' in dataset:
            parts = []
            for part in dataset['parts']:
                parts.append(create_mask(part['path'], mask_config))
            masks.append(parts)
        else:
            masks.append(create_mask(dataset['path'], mask_config))

    return masks


def create_mask(input_file, mask_config):
    columns = list(mask_config.keys())
    df = read_h5py(input_file, key='events', columns=columns)

    mask = np.ones(len(df), dtype='bool')
    for key, (op, val) in mask_config.items():
        mask &= OPERATORS[op](df[key], val)
    return mask


def calc_all_weights(datasets, masks=None):
    weights = []
    for d, dataset in enumerate(datasets):
        if 'parts' in dataset:
            parts = []
            for p, part in enumerate(dataset['parts']):
                mask = masks[d][p] if masks is not None else None
                parts.append(calc_weights(part, mask=mask))
            weights.append(parts)
        else:
            mask = masks[d] if masks is not None else None
            weights.append(calc_weights(dataset, mask=mask))
    return weights


def get_common_columns(datasets):
    common_columns = set()
    for dataset in datasets:
        if 'parts' in dataset:
            for part in dataset['parts']:
                common_columns = update_columns(part, common_columns)
        else:
            common_columns = update_columns(dataset, common_columns)
    return common_columns


def print_event_rates(weights, datasets):
    for d, dataset in enumerate(datasets):
        l = dataset['label']
        if 'parts' in dataset:
            print(f'{l: <15}')
            total = 0
            for p, part in enumerate(dataset['parts']):
                w = weights[d][p].sum()
                total += w
                print(f'  {part["label"]: <15}', f'{w / 3600:6.2f} Events/s')
            print(f'  {"total": <15}', f'{total / 3600:6.2f} Events/s')
        else:
            print(f'{l: <15}', f'{weights[d].sum() / 3600:6.2f} Events/s')


@click.command()
@click.argument('config')
@click.argument('outputfile')
def main(config, outputfile):

    with open(config) as f:
        config = yaml.load(f)

    datasets = config['datasets']

    n_bins = config.get('n_bins', 100)

    if config.get('event_selection') is not None:
        masks = create_masks(config)
    else:
        masks = None

    # get columns available in all datasets and calculate weights

    weights = calc_all_weights(datasets, masks)
    common_columns = get_common_columns(datasets)

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

    fig = plt.figure(constrained_layout=True)
    ax_hist = fig.add_subplot(1, 1, 1)

    with PdfPages(outputfile) as pdf:
        for i, column in enumerate(tqdm(columns)):

            kwargs = config.get('columns').get(column, {})
            kwargs['n_bins'] = kwargs.get('n_bins', n_bins)
            if 'transform' in kwargs:
                kwargs['transform'] = eval(kwargs['transform'])

            dfs = read_dfs_for_column(datasets, column, masks=masks)

            if i == 0:
                print_event_rates(weights, datasets)

            ax_hist.cla()
            try:
                plot_hists(dfs, weights, column, datasets, ax=ax_hist, **kwargs)
                # fig.tight_layout(pad=0)
                pdf.savefig(fig)
            except IOError as e:
                print(f'Could not plot column {column}')
                print(e)


if __name__ == '__main__':
    main()
