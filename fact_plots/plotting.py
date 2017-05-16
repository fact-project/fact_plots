import matplotlib.patches as patches
import matplotlib.pyplot as plt

default_margins = {
    'left': 0.0,
    'height': 0.3,
    'width': 1.0,
    'bottom': 1.,
    }


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
