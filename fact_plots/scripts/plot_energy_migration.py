import click
import numpy as np
from fact.io import read_h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


@click.command()
@click.argument('gamma_path')
@click.option(
    '--std', default=False, is_flag=True,
    help='Use std instead of inter-percentile distance',
)
@click.option('--n-bins', default=100, type=int)
@click.option('--threshold', type=float)
@click.option('--theta2-cut', type=float)
@click.option('--logz', is_flag=True, help='Use a logarithmic color scale')
@click.option('--cmap', help='Matplotlib colormap to use')
@click.option('-o', '--output')
def main(gamma_path, std, n_bins, threshold, theta2_cut, logz, cmap, output):

    df = read_h5py(
        gamma_path,
        key='events',
        columns=[
            'gamma_energy_prediction',
            'corsika_evt_header_total_energy',
            'gamma_prediction',
            'theta_deg'
        ],
    )

    if threshold:
        df = df.query('gamma_prediction >= @threshold').copy()
    if theta2_cut:
        df = df.query('theta_deg**2 <= @theta2_cut').copy()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    e_min = min(
        df.gamma_energy_prediction.min(),
        df.corsika_evt_header_total_energy.min()
    )
    e_max = max(
        df.gamma_energy_prediction.max(),
        df.corsika_evt_header_total_energy.max()
    )

    limits = np.log10([e_min, e_max])

    ax.hist2d(
        np.log10(df.corsika_evt_header_total_energy.values),
        np.log10(df.gamma_energy_prediction.values),
        bins=n_bins,
        range=[limits, limits],
        norm=LogNorm() if logz else None,
        cmap=cmap,
    )

    ax.set_aspect(1)

    ax.set_xlabel(r'$\log_{10}(E_\mathrm{true} \,\, / \,\, \mathrm{GeV})$')
    ax.set_ylabel(r'$\log_{10}(E_\mathrm{est} \,\, / \,\, \mathrm{GeV})$')

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    main()
