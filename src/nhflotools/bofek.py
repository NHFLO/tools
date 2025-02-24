import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pedon as pe


def bofek_to_boring(profiles: pd.DataFrame, soil_mapper, iprofile: int):
    """Convert a bofek profile to a boring object.

    Parameters
    ----------
    profiles : pd.DataFrame
        DataFrame with the profiles
    soil_mapper : dict
        dictionary with the soil types
    iprofile : int
        profile number

    Returns
    -------
    boring
        boring object
    """
    profile = profiles.loc[[iprofile]]
    depth = (
        profile.loc[:, ["iZ1", "iZ2", "iZ3", "iZ4", "iZ5", "iZ6", "iZ7", "iZ8", "iZ9"]].squeeze().rename("Depth [cm]")
    )
    soil = (
        profile.loc[:, ["iSoil1", "iSoil2", "iSoil3", "iSoil4", "iSoil5", "iSoil6", "iSoil7", "iSoil8", "iSoil9"]]
        .squeeze()
        .rename("Soil")
    )
    soil.index = depth

    return soil.map(soil_mapper).dropna().rename(iprofile)


def get_bofek_profile(bofek_gdf, bofek_table, bofek_profiles):
    """Get the genuchten parameters from the bofek profiles.

    Parameters
    ----------
    bofek_gdf : geopandas.GeoDataFrame
        bofek data
    bofek_table : pd.DataFrame
        bofek table
    bofek_profiles : pd.DataFrame
        bofek profiles

    Returns
    -------
    dict
        dictionary with the genuchten parameters
    """
    # map unique profiles to soil types
    soil_mapper = {i + 1: pe.Soil(val).from_staring("2018").model for i, val in enumerate(bofek_table.columns[12:])}
    profiel_cluster = dict(zip(bofek_table["Cluster"], bofek_table["Profiel"], strict=False))
    bofek_gdf["profile"] = bofek_gdf["BOFEK2020"].astype(int).map(profiel_cluster)
    unique_profiles = bofek_gdf["profile"].unique()
    genuchten_params = {
        iprofile: bofek_to_boring(bofek_profiles, soil_mapper, iprofile) for iprofile in unique_profiles
    }

    return genuchten_params


def get_bofek_berging(genpar, depth_gw, dh=0.1):
    """Get the berging from a bofek profile.

    Parameters
    ----------
    genpar : dict
        genuchten parameters
    depth_gw : float
        depth to groundwater
    dh : float, optional
        step size, by default 0.1

    Returns
    -------
    float
        berging
    """
    if depth_gw <= 0:
        return 0
    if np.isnan(depth_gw):
        return np.nan

    h = np.arange(0, depth_gw, dh)
    berging = 0
    starth = depth_gw
    for depth, genobj in genpar.items():
        endh = depth_gw - depth

        traject_h = h[(starth > h) & (h > endh)]
        theta = genobj.theta(h=traject_h)

        berging += np.sum((genobj.theta_s - theta) * dh) / 100  # meter

        starth = endh

        if depth > depth_gw:
            break

    return berging


def plot_bofek_profile(genpar, depth_gw, ax=None, dh=0.1, plot_profile_log=False):
    """Plot a bofek profile.

    Parameters
    ----------
    genpar : dict
        genuchten parameters
    depth_gw : float
        depth to groundwater
    ax : _type_, optional
        axis, by default None
    dh : float, optional
        step size, by default 0.1
    plot_profile_log : bool, optional
        plot the profile in log scale, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with the berging
    """
    if depth_gw < 0:
        raise ValueError("cannot plot a profile when there is no berging")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    df = genpar.to_frame()
    df[["berging [m]"]] = np.nan
    colorcycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    h = np.arange(0, depth_gw, dh)

    starth = depth_gw
    for i, (depth, genobj) in enumerate(genpar.items()):
        endh = depth_gw - depth

        traject_h = h[(starth > h) & (h > endh)]
        theta = genobj.theta(h=traject_h)

        berging = np.sum((genobj.theta_s - theta) * dh) / 100  # meter

        df.loc[depth, "berging [m]"] = berging
        df.loc[depth, "starth [m]"] = starth
        df.loc[depth, "endh [m]"] = endh

        ax.plot(theta, traject_h, color=colorcycle[i], label=f"{genobj.alpha=}")
        ax.vlines(genobj.theta_s, traject_h[0], starth, color=colorcycle[i], ls="--")
        ax.fill_betweenx(traject_h, theta, genobj.theta_s, color=colorcycle[i], alpha=0.5)

        if plot_profile_log:
            ax.semilogy(genobj.theta(h=h), h, color=colorcycle[i], ls="--", alpha=0.5, label=f"{genobj.alpha}")

        starth = endh

        if depth > depth_gw:
            break

    ax.legend()
    return df


def get_brooks(gen: pe.Genuchten, h: np.ndarray[float] | None) -> pe.Brooks:
    """Get the Brooks-Corey parameters from the van Genuchten parameters.

    Parameters
    ----------
    gen : pe.Genuchten
        van Genuchten parameters
    h : np.ndarray[float] | None
        heads to fit the Brooks-Corey parameters to, by default None

    Returns
    -------
    pe.Brooks
        Brooks-Corey parameters
    """
    if h is not None:
        k = gen.k(h)
        theta = gen.theta(h)
        soilsample = pe.SoilSample(h=h, k=k, theta=theta)
        pbounds = pe._params.pBrooks.copy()
        pbounds.loc["theta_r"] = (
            gen.theta_r,
            max(gen.theta_r - 0.1, 0.0),
            gen.theta_r + 0.1,
        )
        pbounds.loc["theta_s"] = (gen.theta_s, gen.theta_s - 0.1, gen.theta_s + 0.1)
        pbounds.loc["l"] = (5.0, 0.01, 20.0)
        pbounds.loc["h_b"] = (1.0, 0.1, 100.0)
        bc = soilsample.fit(pe.Brooks, pbounds=pbounds, k_s=gen.k_s)
    else:
        # Morel-Seytoux (1996) - Parameter equivalence for the Brooks-Corey and van Genuchten soil characteristics
        eps = 1 + 2 / gen.m  # eq 16b
        h_b = (
            1
            / gen.alpha
            * (eps + 3)
            / (2 * eps * (eps - 1))
            * (147.8 + 8.1 * eps + 0.0928 * eps**2)
            / (55.6 + 7.4 * eps + eps**2)
        )  # eq 17
        length = 2 / (eps - 3)  # because eps = 3 + 2 / l
        bc = pe.Brooks(k_s=gen.k_s, theta_r=gen.theta_r, theta_s=gen.theta_s, h_b=h_b, l=length)
    return bc
