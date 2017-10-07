import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from irf.collection_area import collection_area


@u.quantity_input(impace=u.meter)
def plot_effective_area(
        all_events,
        selected_events,
        impact,
        bins=10,
        sample_fraction=1.0,
        ax=None,
        **kwargs
        ):

    ax = ax or plt.gca()

    if isinstance(bins, int):
        bins = np.logspace(
            np.log10(all_events.min()),
            np.log10(all_events.max()),
            bins + 1,
        )

    ret = collection_area(
        all_events,
        selected_events,
        impact=impact,
        bins=bins,
        log=False,
        sample_fraction=sample_fraction,
    )
    area, bin_centers, bin_width, lower_conf, upper_conf = ret

    linestyle = kwargs.pop('ls', '')
    linestyle = kwargs.pop('linestyle', linestyle)

    line = ax.errorbar(
        bin_centers,
        area.value,
        xerr=bin_width / 2,
        yerr=[
            (area - lower_conf).value,
            (upper_conf - area).value
        ],
        linestyle=linestyle,
        **kwargs,
    )

    return line
