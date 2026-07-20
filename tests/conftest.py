"""Shared fixtures for the nhflotools test suite."""

import os

import matplotlib.pyplot as plt
import pytest
import xarray as xr

from .util import make_gwf_disv, make_rect_vertex_ds

_DATA_LOCATION_ENV = "NHFLODATA_LOCATION"
_saved_data_location = None


def pytest_configure(config):  # noqa: ARG001
    """Suppress ``NHFLODATA_LOCATION`` for the whole session, restoring it afterwards.

    ``nhflodata.get_abs_data_path`` resolves against that variable when it is set and only
    *warns* when the resulting path is missing, so a developer with a real data mount would
    otherwise run these tests against their own data instead of the mockup data shipped in
    the wheel -- the same data CI resolves. The tests would then pass or fail depending on
    which datasets happen to be mounted locally.

    This runs in ``pytest_configure`` rather than in a fixture deliberately: it fires before
    collection, so parametrisation and any session- or module-scoped fixture also see the
    variable unset. A function-scoped fixture would run too late for both.
    """
    global _saved_data_location  # noqa: PLW0603
    _saved_data_location = os.environ.pop(_DATA_LOCATION_ENV, None)


def pytest_unconfigure(config):  # noqa: ARG001
    """Put ``NHFLODATA_LOCATION`` back, for in-process runners that outlive the session."""
    if _saved_data_location is not None:
        os.environ[_DATA_LOCATION_ENV] = _saved_data_location


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
