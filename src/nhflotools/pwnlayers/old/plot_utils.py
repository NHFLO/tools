import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def get_figure(extent, title=None, figsize=(10, 8)):
    # get a figure and an ax of a map of the model domain
    if type(extent) not in [list, tuple]:
        # assume extent is a flopy model m
        extent = extent.modelgrid.extent

    f, ax = plt.subplots(figsize=figsize)
    ax.grid()
    if title:
        ax.set_title(title)
    ax.axis("scaled")
    ax.axis(extent)
    plt.yticks(rotation=90, va="center")
    f.tight_layout(pad=0.0)
    return f, ax
