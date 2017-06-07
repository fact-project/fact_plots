import matplotlib.patches as patches
import matplotlib.pyplot as plt

default_margins = {
    'left': 0.0,
    'height': 0.3,
    'width': 1.0,
    'bottom': 1.,
}

positions = {
    'center': (0.5, 0.5, 'center', 'center'),
    'left': (0.05, 0.5, 'left', 'center'),
    'right': (0.975, 0.5, 'right', 'center'),
    'upper center': (0.5, 0.975, 'center', 'top'),
    'lower center': (0.5, 0.025, 'center', 'bottom'),
    'upper left': (0.025, 0.975, 'left', 'top'),
    'lower left': (0.025, 0.025, 'left', 'bottom'),
    'upper right': (0.975, 0.975, 'right', 'top'),
    'lower right': (0.975, 0.025, 'right', 'bottom'),
}


def add_preliminary(
        position,
        size=20,
        color='lightgray',
        ha='right',
        va='top',
        ax=None
        ):
    ax = ax or plt.gca()

    x, y, ha, va = positions[position]

    ax.text(
        x, y, 'PRELIMINARY',
        ha=ha, va=va,
        size=size,
        color=color,
        transform=ax.transAxes,
        weight='bold',
        zorder=-1,
    )


def plotInfoBox(text, ax=None,
                left=default_margins["left"],
                height=default_margins["height"],
                width=default_margins["width"],
                bottom=default_margins["bottom"]
                ):
    top = bottom + height
    right = left + width
    ax = ax or plt.gca()
    p = patches.Rectangle((left, 1.),
                          width, height,
                          fill=True, transform=ax.transAxes,
                          clip_on=False, facecolor='white', edgecolor='black')
    ax.add_patch(p)

    # Summary Box
    ax.text(0.5*(left+right),
            0.5*(top+bottom)-0.05,
            text,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10,
            transform=ax.transAxes
            )

    return ax


def plotTitleBox(title, ax=None,
                 left=default_margins["left"],
                 height=default_margins["height"],
                 width=default_margins["width"],
                 bottom=default_margins["bottom"]
                 ):
    top = bottom + height
    right = left + width
    ax = ax or plt.gca()
    ax.text(0.5*(left+right),
            (top),
            title,
            bbox={'facecolor': 'white', 'pad': 10},
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=14, color='black',
            transform=ax.transAxes)
    return ax
