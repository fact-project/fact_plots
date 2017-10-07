import click
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord

from fact.io import read_h5py

from ..skymap import plot_skymap
from ..plotting import add_preliminary

plot_config = {
    'xlabel': r'$(\theta \,\, / \,\, {}^\circ )^2$',
    'preliminary_position': 'lower center',
    'preliminary_size': 'xx-large',
    'preliminary_color': 'lightgray',
}

columns = [
    'reconstructed_source_position',
    'unix_time_utc',
    'az_tracking',
    'zd_tracking',
]


@click.command()
@click.argument('data_path')
@click.option('--threshold', type=float, help='prediction threshold', default=0.8, show_default=True)
@click.option('--key', help='Key for the hdf5 group', default='events')
@click.option('--bins', help='Number of bins in the histogram', default=100, show_default=True)
@click.option('--width', help='Extent of the histogram in degree ', default=4.0, show_default=True)
@click.option('--preliminary', is_flag=True, help='Add preliminary')
@click.option('-c', '--config', help='Path to yaml config file')
@click.option('-o', '--output', help='(optional) Output file for the plot')
@click.option('-s', '--source', help='Name of the source show')
def main(data_path, threshold, key, bins, width, preliminary, config, output, source):
    '''
    Plot a 2d histogram of the origin of the air showers in the
    given hdf5 file in ra, dec.
    '''
    if config:
        with open(config) as f:
            plot_config.update(yaml.safe_load(f))

    if threshold > 0.0:
        columns.append('gamma_prediction')

    events = read_h5py(data_path, key='events', columns=columns)

    events['time'] = pd.to_datetime(
        events['unix_time_utc_0'] * 1e6 + events['unix_time_utc_1'],
        unit='us',
    )

    if threshold > 0.0:
        events = events.query('gamma_prediction >= @threshold').copy()

    fig, ax = plt.subplots(1, 1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    if source:
        coord = SkyCoord.from_name(source)
        center_ra = coord.ra.deg
        center_dec = coord.dec.deg
    else:
        center_ra = center_dec = None

    ax, img = plot_skymap(
        events,
        width=width,
        bins=bins,
        center_ra=center_ra,
        center_dec=center_dec,
        ax=ax,
    )

    if source:
        ax.plot(
            center_ra,
            center_dec,
            label=source,
            color='r',
            marker='o',
            linestyle='',
            markersize=10,
            markerfacecolor='none',
        )
        ax.legend()

    fig.colorbar(img, cax=cax, label='Gamma-Like Events')

    if preliminary:
        add_preliminary(
            plot_config['preliminary_position'],
            size=plot_config['preliminary_size'],
            color=plot_config['preliminary_color'],
            ax=ax,
            zorder=3,
        )

    fig.tight_layout(pad=0)
    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()
