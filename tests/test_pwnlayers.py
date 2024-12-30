import logging
import os
import tempfile

import nlmod
import pytest
from nhflodata.get_paths import get_abs_data_path

from nhflotools.pwnlayers.layers import (
    get_pwn_layer_model,
    get_top_from_ahn,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

tmpdir = tempfile.gettempdir()


# %% create folder structure
@pytest.mark.slow
def test_create_pwn_model_grid_only(tmpdir, model_name="pwn"):
    tmpdir = str(tmpdir)
    _, cachedir = nlmod.util.get_model_dirs(tmpdir)
    data_path_mensink = get_abs_data_path(name="bodemlagen_pwn_nhdz", version="latest", location="get_from_env")
    data_path_2024 = get_abs_data_path(name="bodemlagen_pwn_2024", version="latest", location="get_from_env")
    data_path_bergen = get_abs_data_path(name="bodemlagen_pwn_bergen", version="latest", location="get_from_env")
    data_path_koppeltabel = get_abs_data_path(
        name="bodemlagen_pwn_regis_koppeltabel", version="latest", location="get_from_env"
    )
    fname_koppeltabel = os.path.join(data_path_koppeltabel, "bodemlagenvertaaltabelv2.csv")

    transition_length = 1000.0

    layer_model_regis_struc = nlmod.read.regis.get_combined_layer_models(
        (99000, 103000, 500000, 505000),
        use_regis=True,
        use_geotop=False,
        remove_nan_layers=False,
        cachedir=cachedir,
        cachename="layer_model",
    )

    ds_regis = nlmod.to_model_ds(
        layer_model_regis_struc,
        model_name,
        tmpdir,
        remove_nan_layers=False,
        transport=False,
    )

    ds_regis = nlmod.grid.refine(ds_regis, model_ws=tmpdir, remove_nan_layers=False)

    ahn = nlmod.read.ahn.get_ahn4(extent=ds_regis.extent, cachedir=cachedir, cachename="ahn")
    ds_regis["ahn"] = nlmod.dims.resample.structured_da_to_ds(ahn, ds_regis, method="average")
    ds_regis["top"] = get_top_from_ahn(
        ds=ds_regis,
        replace_surface_water_with_peil=True,
        replace_northsea_with_constant=0.0,
        method_elsewhere="nearest",
        cachedir=cachedir,
    )
    assert ds_regis.top.isnull().sum() == 0, "Variable top should not contain nan values"

    if (ds_regis.layer != nlmod.read.regis.get_layer_names()).any():
        msg = "All REGIS layers should be present in `ds_regis`. Use `get_regis(.., remove_nan_layers=False)`."
        raise ValueError(msg)

    ds = get_pwn_layer_model(
        ds_regis=ds_regis,
        data_path_mensink=data_path_mensink,
        data_path_bergen=data_path_bergen,
        data_path_2024=data_path_2024,
        fname_koppeltabel=fname_koppeltabel,
        top=ds_regis["top"],
        length_transition=transition_length,
        cachedir=cachedir,
        cachename="pwn_layer_model",
    )

    for k in ["botm", "kh", "kv"]:
        assert k in ds, f"Variable {k} should be present in `ds`."
        assert ~(ds[k].values == ds_regis[k].values).all(), f"Variable {k} should not be equal to `ds_regis`."
