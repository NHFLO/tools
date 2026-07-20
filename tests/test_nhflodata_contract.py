"""Contract tests: every nhflodata file the 09pwnmodel2 closure reads must exist.

``nhflodata.get_paths.get_abs_data_path`` only *warns* when a dataset path is missing, so a
restructure of the NHFLO/data repository would otherwise surface as a ``FileNotFoundError``
deep inside a model run. These tests stat every (dataset, relative file) pair that nhflotools
-- or ``modelscripts/09pwnmodel2/01_pwnmodel2.py`` -- hardcodes, against the mockup data
packaged with nhflodata (``pyproject.toml`` sets ``NHFLODATA_LOCATION=`` through pytest-env,
so resolution always lands on the mockup).
"""

from pathlib import Path

import pandas as pd
import pytest

from nhflotools.pwnlayers3.layers import layer_names

get_paths = pytest.importorskip("nhflodata.get_paths")
get_abs_data_path = get_paths.get_abs_data_path

# Hydrogeological units of pwnlayers3/layers.py:get_gdf_boundaries.
_BOUNDARY_UNITS = ("11", "12", "13", "21", "22", "31", "32")

# Aquitards whose kh comes from NHDZ transmissivity points (pwnlayers3/layers.py:get_kh).
_KD_NHDZ_LAYERS = ("S12", "S13", "S21", "S22", "S31")

_KOPPELTABEL_CSV = "bodemlagenvertaaltabelv2.csv"


def _bodemlagen_2024_files():
    """Yield the relative paths read from the ``bodemlagen_pwn_2024`` dataset.

    Derived from the path expressions in ``nhflotools/pwnlayers3/layers.py`` and
    ``nhflotools/pwnlayers3/plot.py`` rather than from a directory listing, so a file the
    code needs but the dataset stopped shipping is a failure, not a silent omission.

    Yields
    ------
    str
        Path relative to the dataset root, POSIX-style.
    """
    yield "botm/botm.geojson"  # get_botm + plot._load_and_project_source_botm
    yield "boundaries/triwaco_model_nhdz.geojson"  # get_kh
    for unit in _BOUNDARY_UNITS:  # get_gdf_boundaries
        yield f"boundaries/S{unit}/S{unit}.geojson"
    for name in layer_names:
        # get_kh reads K<aquifer>_combined; aquitards read C<aquitard>_combined (get_kh + get_kv)
        prefix = "K" if name.startswith("W") else "C"
        yield f"conductances/{prefix}{name}_combined.geojson"
    for name in _KD_NHDZ_LAYERS:  # get_kh, NHDZ branch
        yield f"conductances/KD{name}_NHDZ.geojson"


def _data_files():
    """Yield ``(dataset_name, relative_path)`` for every hardcoded read.

    Yields
    ------
    tuple of (str, str)
        Dataset name as passed to ``get_abs_data_path`` and the relative file joined onto it.
    """
    for relative_path in _bodemlagen_2024_files():
        yield "bodemlagen_pwn_2024", relative_path
    # panden.py reads the .shp; geopandas additionally requires the .dbf/.shx sidecars and
    # needs the .prj to attach EPSG:28992 before the grid intersection.
    for suffix in ("shp", "dbf", "shx", "prj"):
        yield "oppervlaktewater_pwn_shapes_panden", f"Panden_ICAS_IKIEF.{suffix}"
    yield "nhi_chloride_concentration", "chloride_p50.nc"  # nhi_chloride.py:32
    yield "wells_pwn", "pumping_infiltration_wells.geojson"  # well.py:36
    yield "wells_pwn", "sec_flows.feather"  # well.py:46
    yield "wells_tata", "tata_zoutwaterbronnen.geojson"  # well.py:108
    yield "wells_tata", "tata_zoetwaterbronnen.geojson"  # well.py:120
    # Read by 01_pwnmodel2.py itself (lines 126, 363, 400).
    yield "lakes_pwn", "lakes_pwn.geojson"
    yield "drains_pwn", "drains_pwn.geojson"
    yield "hfb_pwn", "hfb_pwn.geojson"


_DATA_FILES = tuple(_data_files())


@pytest.mark.parametrize(("dataset", "relative_path"), _DATA_FILES, ids=[f"{d}:{r}" for d, r in _DATA_FILES])
def test_mockup_data_file_resolves_and_is_nonempty(dataset, relative_path):
    """The resolved path exists and holds bytes.

    ``get_abs_data_path`` warns instead of raising on a missing dataset, and a zero-byte
    placeholder would satisfy a bare existence check while breaking every reader, so both
    are asserted.
    """
    root = Path(get_abs_data_path(name=dataset, version="latest", location="get_from_env"))
    path = root / relative_path

    assert path.is_file(), f"{dataset} is missing {relative_path} (resolved to {path})"
    assert path.stat().st_size > 0, f"{dataset}/{relative_path} is empty"


def test_paths_resolve_to_the_packaged_mockup():
    """Every path check below assumes resolution lands on the packaged mockup data.

    ``get_abs_data_path`` resolves against ``NHFLODATA_LOCATION`` whenever it holds a
    non-empty value, and only warns when the result is missing. On a machine with a real
    data mount that would silently point this whole module at local data, so it would pass
    or fail on whichever datasets happen to be mounted rather than on what NHFLO/data ships.

    ``pyproject.toml`` therefore sets ``NHFLODATA_LOCATION=`` through pytest-env, which
    overrides any inherited value with the empty string that ``get_paths`` reads as "use
    mockup" (``get_paths.py:69`` defaults to ``""``, ``:96`` branches on it being falsy).
    Asserting the resolved location rather than the variable keeps this honest if that
    empty-means-mockup contract ever changes.
    """
    mockup_root = Path(get_paths.__file__).parent / "data" / "mockup"
    resolved = Path(get_abs_data_path(name="bodemlagen_pwn_regis_koppeltabel", version="latest"))
    assert mockup_root in resolved.parents


def test_koppeltabel_columns_and_layer_coverage():
    """The koppeltabel exposes the columns and the layer names the merge indexes by.

    ``merge_layer_models.combine_two_layer_models`` defaults to the column names
    ``'Regis II v2.2'`` / ``'ASSUMPTION1'`` and then does
    ``layer_model_other.sel(layer=<ASSUMPTION1 values>)``. So the non-null ASSUMPTION1
    entries must be exactly the pwnlayers3 layer set: an extra name raises inside ``sel``,
    a missing name silently drops that PWN layer from the merged model. Rows without a
    REGIS name can never be coupled, so that column must be complete.
    """
    root = Path(get_abs_data_path(name="bodemlagen_pwn_regis_koppeltabel", version="latest", location="get_from_env"))
    # Same read as pwnlayers3/layers.py:166.
    df = pd.read_csv(root / _KOPPELTABEL_CSV, skiprows=0, index_col=0)

    assert {"Regis II v2.2", "ASSUMPTION1"}.issubset(df.columns)
    assert set(df["ASSUMPTION1"].dropna()) == set(layer_names)
    assert df["Regis II v2.2"].notna().all()
