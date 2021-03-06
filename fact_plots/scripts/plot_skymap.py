import click
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from astropy.coordinates import SkyCoord

from fact.io import read_h5py

from ..skymap import plot_skymap
from ..plotting import add_preliminary

yaml = YAML(typ='safe')
plot_config = {
    'preliminary_position': 'lower center',
    'preliminary_size': 'xx-large',
    'preliminary_color': 'lightgray',
    'source_color': 'lightgray',
    'source_size': 10,
    'legend_font_color': 'lightgray',
    'legend': {
        'facecolor': '0.3',
        'edgecolor': '0.3',
        'markerscale': 0.5,
    }
}

columns = [
    'ra_prediction',
    'dec_prediction'
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
@click.option('-n', '--source-name', help='Name of the source show')
@click.option('-s', '--source', type=(str, str), default=(None, None), help='RA and DEC of the source')
def main(data_path, threshold, key, bins, width, preliminary, config, output, source_name, source):
    '''
    Plot a 2d histogram of the origin of the air showers in the
    given hdf5 file in ra, dec.
    '''
    if config:
        with open(config) as f:
            plot_config.update(yaml.load(f))

    if threshold > 0.0:
        columns.append('gamma_prediction')

    events = read_h5py(data_path, key='events', columns=columns)

    if threshold > 0.0:
        events = events.query('gamma_prediction >= @threshold').copy()

    fig, ax = plt.subplots(1, 1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    if source[0] and source_name:
        coord = SkyCoord(ra=source[0], dec=source[1])
        label = source_name
    elif source_name:
        coord = SkyCoord.from_name(source_name)
        label = source_name
    else:
        coord = None

    if coord:
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

    if coord:
        ax.plot(
            center_ra,
            center_dec,
            label=label,
            color=plot_config['source_color'],
            marker='o',
            linestyle='',
            markersize=plot_config['source_size'],
            markerfacecolor='none',
        )
        if label:
            l = ax.legend(**plot_config['legend'])
            if plot_config['legend_font_color']:
                for t in l.get_texts():
                    t.set_color(plot_config['legend_font_color'])

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
