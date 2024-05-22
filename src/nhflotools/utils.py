import json
import logging
import os
import sys
from pathlib import Path

import nlmod
from flopy.utils import get_modflow
from flopy.utils.get_modflow import flopy_appdata_path, get_release

logger = logging.getLogger(__name__)

nlmod_bindir = Path(nlmod.__file__).parent / "bin"


def get_exe_path(bindir=None, exe_name="mf6", download_if_not_found=True, version_tag="latest", repo="executables"):
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
    download_if_not_found : bool, optional
        Download the executables if they are not found, by default True.
    repo : str, default "executables"
        Name of GitHub repository. Choose one of "executables" (default),
        "modflow6", or "modflow6-nightly-build". Used only in search destinations 3 and 4.
    version_tag : str, default "latest"
        GitHub release ID. Used only in search destinations 3 and 4.

    Returns
    -------
    exe_full_path : str
        full path of the executable.
    """
    if sys.platform.startswith("win") and not exe_name.endswith(".exe"):
        exe_name += ".exe"

    exe_full_path = (
        get_bin_directory(
            bindir=bindir,
            exe_name=exe_name,
            download_if_not_found=download_if_not_found,
            version_tag=version_tag,
            repo=repo,
        )
        / exe_name
    )

    msg = f"Executable path: {exe_full_path}"
    logger.debug(msg)

    return exe_full_path


def get_bin_directory(
    bindir=None, exe_name="mf6", download_if_not_found=True, version_tag="latest", repo="executables"
) -> Path:
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
    download_if_not_found : bool, optional
        Download the executables if they are not found, by default True.
    repo : str, default "executables"
        Name of GitHub repository. Choose one of "executables" (default),
        "modflow6", or "modflow6-nightly-build". Used only if download is needed.
    version_tag : str, default "latest"
        GitHub release ID. Used only if download is needed.

    Returns
    -------
    Path
        The directory where the executables are stored.

    Raises
    ------
    FileNotFoundError
        If the executables are not found in the specified directories.
    """
    bindir = Path(bindir) if bindir is not None else None

    if sys.platform.startswith("win") and not exe_name.endswith(".exe"):
        exe_name += ".exe"

    # If bindir is provided
    use_bindir = bindir is not None and exe_name is not None and Path(bindir / exe_name).exists()
    use_bindir |= bindir is not None and exe_name is None and Path(bindir).exists()

    if use_bindir:
        return bindir

    # If the executables are in the nlmod directory
    use_nlmod_bindir = exe_name is not None and Path(nlmod_bindir / exe_name).exists()
    use_nlmod_bindir |= exe_name is None and Path(nlmod_bindir).exists()

    if use_nlmod_bindir:
        return nlmod_bindir

    # If the executables are in the flopy directory
    flopy_bindir = _get_flopy_bin_directory()

    use_flopy_bindir = flopy_bindir is not None and exe_name is not None and Path(flopy_bindir / exe_name).exists()
    use_flopy_bindir |= flopy_bindir is not None and exe_name is None and Path(flopy_bindir).exists()

    if use_flopy_bindir:
        return flopy_bindir

    # Else download the executables
    if download_if_not_found:
        download_mfbinaries(bindir=bindir, version_tag=version_tag, repo=repo)
    else:
        msg = f"Could not find {exe_name} in {bindir}, {nlmod_bindir} and {flopy_bindir}."
        raise FileNotFoundError(msg)

    if bindir is not None and exe_name is not None and not Path(bindir / exe_name).exists():
        msg = f"Could not find {exe_name} in {bindir}."
        raise FileNotFoundError(msg)
    if bindir is None and exe_name is not None and not Path(nlmod_bindir / exe_name).exists():
        msg = f"Could not find {exe_name} in {nlmod_bindir} and {flopy_bindir}."
        raise FileNotFoundError(msg)
    if bindir is None:
        return nlmod_bindir
    return bindir


def _get_flopy_bin_directory(version_tag="latest", repo="executables") -> Path:
    flopy_metadata_fp = flopy_appdata_path / "get_modflow.json"

    if not flopy_metadata_fp.exists():
        return None

    version_tag_pin = get_release(tag=version_tag, repo=repo, quiet=True)["tag_name"]

    try:
        meta_raw = flopy_metadata_fp.read_text()

        # Remove trailing characters that are not part of the JSON.
        while meta_raw[-3:] != "}\n]":
            meta_raw = meta_raw[:-1]

        # get metadata of most all installations
        meta_list = json.loads(meta_raw)

        # get path to the most recent installation. Appended to end of get_modflow.json
        meta_list = [meta for meta in meta_list if (meta["release_id"] == version_tag_pin) and (meta["repo"] == repo)]
        Path(meta_list[-1]["bindir"])

    except:  # noqa: E722
        return None


def download_mfbinaries(bindir=None, version_tag="latest", repo="executables"):
    """Download and unpack platform-specific modflow binaries.

    Source: USGS

    Parameters
    ----------
    binpath : str, optional
        path to directory to download binaries to, if it doesnt exist it
        is created. Default is None which sets dir to nlmod/bin.
    repo : str, default "executables"
        Name of GitHub repository. Choose one of "executables" (default),
        "modflow6", or "modflow6-nightly-build".
    version_tag : str, default "latest"
        GitHub release ID.

    """
    if bindir is None:
        # Path objects are immutable so a copy is implied
        bindir = nlmod_bindir

    if not os.path.isdir(bindir):
        os.makedirs(bindir)

    get_modflow(bindir=str(bindir), release_id=version_tag, repo=repo)

    if sys.platform.startswith("win"):
        # download the provisional version of modpath from Github
        nlmod.util.download_modpath_provisional_exe(bindir)
