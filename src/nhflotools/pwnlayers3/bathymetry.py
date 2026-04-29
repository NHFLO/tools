"""Helpers for fetching Rijkswaterstaat bathymetry GeoTIFFs.

The Rijkswaterstaat ``bodemhoogte_20mtr_*.tif`` rasters are distributed with an
ISO 19115 metadata sidecar (``.tif.xml``). Streaming the GeoTIFF through GDAL's
``/vsicurl/`` driver triggers spurious ``TIFFReadDirectory`` warnings caused by
chunked HTTP fetches that drop bytes around the overview IFDs near the end of
the file. Downloading the file once to local disk avoids the warnings entirely;
the file itself is well-formed.

This module exposes :func:`open_rws_bathymetry`, which caches both files under
a local directory, opens the raster with xarray/rasterio, and merges the most
useful ISO 19115 fields into ``Dataset.attrs``.
"""

from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET  # noqa: S405
from pathlib import Path  # noqa: TC003
from urllib.parse import urlparse

import xarray as xr

_NS = {
    "gmd": "http://www.isotc211.org/2005/gmd",
    "gco": "http://www.isotc211.org/2005/gco",
    "gml": "http://www.opengis.net/gml",
    "gmx": "http://www.isotc211.org/2005/gmx",
    "xlink": "http://www.w3.org/1999/xlink",
}

URL = "https://downloads.rijkswaterstaatdata.nl/bodemhoogte_20mtr/bodemhoogte_20mtr.tif"


def _txt(elem: ET.Element | None, path: str) -> str | None:
    """Return stripped text of the first child matching ``path``, or ``None``."""
    if elem is None:
        return None
    node = elem.find(path, _NS)
    if node is None or node.text is None:
        return None
    text = node.text.strip()
    return text or None


def _float(elem: ET.Element | None, path: str) -> float | None:
    """Return ``float(_txt(elem, path))`` or ``None`` if the text is missing."""
    text = _txt(elem, path)
    return float(text) if text is not None else None


def parse_iso19115(xml_path: Path, source_url: str | None = None) -> dict:
    """Extract a flat dict of CF-friendly attributes from an ISO 19115 metadata file.

    Parameters
    ----------
    xml_path : pathlib.Path
        Path to the ``.xml`` sidecar produced by Rijkswaterstaat.
    source_url : str, optional
        URL the data was downloaded from. Stored under ``source_url`` if given.

    Returns
    -------
    dict
        Subset of ISO 19115 fields suitable for ``xarray.Dataset.attrs``. Keys
        with empty or missing values are omitted.
    """
    root = ET.parse(xml_path).getroot()  # noqa: S314
    ident = root.find(".//gmd:MD_DataIdentification", _NS)
    bbox = root.find(".//gmd:EX_GeographicBoundingBox", _NS)
    crs_codes = [a.get(f"{{{_NS['xlink']}}}href") for a in root.findall(".//gmd:referenceSystemInfo//gmx:Anchor", _NS)]
    pos_acc = root.find(".//gmd:DQ_AbsoluteExternalPositionalAccuracy//gco:Record", _NS)
    license_anchor = root.find(".//gmd:MD_LegalConstraints/gmd:otherConstraints/gmx:Anchor", _NS)

    attrs = {
        "title": _txt(ident, ".//gmd:citation//gmd:title/gco:CharacterString"),
        "summary": _txt(ident, ".//gmd:abstract/gco:CharacterString"),
        "purpose": _txt(ident, ".//gmd:purpose/gco:CharacterString"),
        "topic_category": _txt(ident, ".//gmd:topicCategory/gmd:MD_TopicCategoryCode"),
        "creation_date": _txt(ident, ".//gmd:citation//gmd:date/gco:Date"),
        "metadata_date": _txt(root, ".//gmd:dateStamp/gco:Date"),
        "identifier": _txt(root, ".//gmd:fileIdentifier/gco:CharacterString"),
        "institution": _txt(root, ".//gmd:contact//gmd:organisationName/gco:CharacterString"),
        "horizontal_crs_ref": next((c for c in crs_codes if c and "EPSG/0/28992" in c), None),
        "vertical_crs_ref": next((c for c in crs_codes if c and "EPSG/0/5709" in c), None),
        "geospatial_lon_min": _float(bbox, "gmd:westBoundLongitude/gco:Decimal"),
        "geospatial_lon_max": _float(bbox, "gmd:eastBoundLongitude/gco:Decimal"),
        "geospatial_lat_min": _float(bbox, "gmd:southBoundLatitude/gco:Decimal"),
        "geospatial_lat_max": _float(bbox, "gmd:northBoundLatitude/gco:Decimal"),
        "time_coverage_start": _txt(root, ".//gml:TimePeriod/gml:beginPosition"),
        "lineage": _txt(root, ".//gmd:lineage//gmd:statement/gco:CharacterString"),
        "positional_accuracy_m": float(pos_acc.text) if pos_acc is not None and pos_acc.text else None,
        "license": license_anchor.get(f"{{{_NS['xlink']}}}href") if license_anchor is not None else None,
        "source_url": source_url,
    }
    return {k: v for k, v in attrs.items() if v is not None}


def open_rws_bathymetry(cachedir: Path, url: str = URL) -> xr.DataArray:
    """Download a Rijkswaterstaat bathymetry GeoTIFF and open it as an xarray DataArray.

    The function caches both the ``.tif`` and its ``.tif.xml`` sidecar under
    ``cachedir``. Subsequent calls reuse the cached files. CRS information
    is already embedded in the GeoTIFF tags (EPSG:28992 / RD New); no ``.prj``
    sidecar is required.

    Parameters
    ----------
    cachedir : pathlib.Path
        Local directory used to cache both files. Created if missing. Required
        as streaming the GeoTIFF directly from the URL triggers spurious errors.
    url : str
        URL of the ``.tif`` file. The ``.xml`` sidecar is fetched from
        ``url + ".xml"``.

    Returns
    -------
    xarray.DataArray
        2-D bed-elevation array (``y``, ``x``) opened from the local copy. The
        single-band coordinate is dropped. ISO 19115 metadata and CF-style
        attributes (``units``, ``standard_name``, ``vertical_datum``, ...) are
        merged into ``da.attrs``; the ``spatial_ref`` coordinate carries the
        CRS (EPSG:28992 / RD New).
    """
    if urlparse(url).scheme not in {"http", "https"}:
        msg = f"only http(s) URLs are supported, got: {url!r}"
        raise ValueError(msg)

    cachedir.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    tif = cachedir / name
    xml = cachedir / (name + ".xml")

    if not tif.exists():
        urllib.request.urlretrieve(url, tif)  # noqa: S310
    if not xml.exists():
        urllib.request.urlretrieve(url + ".xml", xml)  # noqa: S310

    ds = xr.open_dataset(tif, engine="rasterio")
    da = ds["band_data"].isel(band=0, drop=True)
    da.attrs = parse_iso19115(xml, source_url=url)
    da.attrs.update(
        long_name="bed elevation",
        standard_name="height_above_reference_ellipsoid",
        units="m",
        vertical_datum="NAP (EPSG:5709)",
        positive="up",
    )
    return da
