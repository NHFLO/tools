from importlib.metadata import version


def show_nhflo_versions():
    """Show the versions of the nhflo suite."""
    msg = ""
    for pkg in ["nhflodata", "nhflotools", "nhflomodels"]:
        try:
            pkg_version = version(pkg)
        except Exception:
            pkg_version = "Not installed"
        msg += f"{pkg :<13s}: {pkg_version}\n"
    print(msg)
