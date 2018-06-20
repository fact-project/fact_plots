import matplotlib.pyplot as plt


def plot_skymap(df, width=4, bins=100, center_ra=None, center_dec=None, ax=None):
    '''
    Plot a 2d histogram of the reconstructed positions of air showers

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of the reconstructed events containing the columns
        `reconstructed_source_position_0`, `reconstructed_source_position_0`,
        `zd_tracking`, `az_tracking`, `time`, where time is the
        observation time as datetime
    width: float
        Extent of the plot in degrees
    bins: int
        number of bins
    center_ra: float
        right ascension of the center in degrees
    center_dec: float
        declination of the center in degrees
    ax: matplotlib.axes.Axes
        axes to plot into
    '''
    ax = ax or plt.gca()

    ra = df['ra_prediction']
    dec = df['dec_prediction']

    if center_ra is None:
        center_ra = ra.mean()
        center_ra *= 15  # conversion from hourangle to degree

    if center_dec is None:
        center_dec = dec.mean()

    bins, x_edges, y_deges, img = ax.hist2d(
        ra * 15,  # conversion from hourangle to degree
        dec,
        bins=bins,
        range=[
            [center_ra - width / 2, center_ra + width / 2],
            [center_dec - width / 2, center_dec + width / 2]
        ],
    )

    ax.set_xlabel('right ascension / degree')
    ax.set_ylabel('declination / degree')
    ax.set_aspect(1)

    return ax, img
