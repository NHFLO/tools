import json
import logging
import sys
from pathlib import Path

import nlmod
from flopy.utils.get_modflow import flopy_appdata_path

logger = logging.getLogger(__name__)


def get_exe_path(bindir=None, exe_name="mf6"):
    """Get the full path of the executable.

    Searching for the executables is done in the following order:
    1. The directory specified by the user.
    2. The directory used by nlmod installed in this environment.
    3. If the executables were downloaded with flopy/nlmod from an other env,
        most recent installation location of MODFLOW is found in flopy metadata.

    Else:
    4. Download the executables.

    The returned directory is checked to contain exe_name if exe_name is provided.

    Parameters
    ----------
    bindir : Path, optional
        The directory where the executables are stored, by default None
    exe_name : str, optional
        The name of the executable, by default "mf6".

    Returns
    -------
    exe_full_path : str
        full path of the executable.
    """
    if sys.platform.startswith("win") and not exe_name.endswith(".exe"):
        exe_name += ".exe"

    exe_full_path = get_bin_directory(bindir=bindir, exe_name=exe_name) / exe_name
    msg = f"Executable path: {exe_full_path}"

    logger.debug(msg)

    return exe_full_path


def get_bin_directory(bindir=None, exe_name="mf6") -> Path:
    """
    Get the directory where the executables are stored.

    Searching for the executables is done in the following order:
    1. The directory specified by the user.
    2. The directory used by nlmod installed in this environment.
    3. If the executables were downloaded with flopy/nlmod from an other env,
        most recent installation location of MODFLOW is found in flopy metadata.

    Else:
    4. Download the executables.

    The returned directory is checked to contain exe_name if exe_name is provided.

    Parameters
    ----------
    bindir : Path, optional
        The directory where the executables are stored, by default "mf6".
    exe_name : str, optional
        The name of the executable, by default None.
    """
    bindir = Path(bindir) if bindir is not None else None

    if sys.platform.startswith("win") and not exe_name.endswith(".exe"):
        exe_name += ".exe"

    # If bindir is provided
    if (bindir is not None and exe_name is not None and Path(bindir / exe_name).exists()) or (
        bindir is not None and exe_name is None and Path(bindir).exists()):
        return bindir

    # If the executables are in the nlmod directory
    nlmod_bindir = Path(nlmod.__file__).parent / "bin"

    if (exe_name is not None and Path(nlmod_bindir / exe_name).exists()) or (
        exe_name is None and Path(nlmod_bindir).exists()):
        return nlmod_bindir

    # If the executables are in the flopy directory
    flopy_bindir = _get_flopy_bin_directory()

    if (flopy_bindir is not None and exe_name is not None and Path(flopy_bindir / exe_name).exists()) or (
        flopy_bindir is not None and exe_name is None and Path(flopy_bindir).exists()):
        return flopy_bindir

    # Else download the executables
    nlmod.download_mfbinaries(bindir=bindir)

    if bindir is not None and exe_name is not None and not Path(bindir / exe_name).exists():
        msg = f"Could not find {exe_name} in {bindir}."
        raise FileNotFoundError(msg)
    if bindir is None and exe_name is not None and not Path(nlmod_bindir / exe_name).exists():
        msg = f"Could not find {exe_name} in {nlmod_bindir} and {flopy_bindir}."
        raise FileNotFoundError(msg)
    if bindir is None:
        return nlmod_bindir
    return bindir


def _get_flopy_bin_directory() -> Path:
    flopy_metadata_fp = flopy_appdata_path / "get_modflow.json"

    if not flopy_metadata_fp.exists():
        return None

    try:
        meta_raw = flopy_metadata_fp.read_text()

        # Remove trailing characters that are not part of the JSON.
        while len(meta_raw) > 0 and meta_raw[-3:] != "}\n]":
            meta_raw = meta_raw[:-1]

        # get metadata of most all installations
        meta_list = json.loads(meta_raw)

        # get path to the most recent installation. Appended to end of get_modflow.json
        Path(meta_list[-1]["bindir"])
    except:  # noqa: E722
        return None
