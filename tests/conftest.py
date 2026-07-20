"""Shared fixtures for the nhflotools test suite."""

import matplotlib.pyplot as plt
import pytest
import xarray as xr

from .util import make_gwf_disv, make_rect_vertex_ds


@pytest.fixture(autouse=True)
def _hygiene():
    """Keep tests independent: no figures or open netCDF handles leak between tests.

    ``FILE_CACHE.clear()`` already closes the netCDF handles xarray holds open; an
    additional ``gc.collect()`` here cost ~50 ms per test (two thirds of the whole
    suite's runtime) and closed nothing further, so it is deliberately absent.
    """
    yield
    plt.close("all")
    xr.backends.file_manager.FILE_CACHE.clear()


@pytest.fixture
def vertex_ds():
    """A fresh 2x2 vertex model dataset.

    Function-scoped on purpose: the functions under test mutate the dataset in place
    (adding ``northsea``, ``sfw_*``, ``drn_*``, ``thickness``), so sharing one would
    couple tests. Building it costs well under a millisecond.
    """
    return make_rect_vertex_ds()


@pytest.fixture
def gwf_disv(vertex_ds, tmp_path):
    """``(ds, gwf)`` with sim/tdis/gwf/disv built in memory from ``vertex_ds``."""
    return make_gwf_disv(vertex_ds, tmp_path)
