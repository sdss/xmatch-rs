#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: test_hats_tools.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

"""Tests for the HATS tools implemented in Rust."""

from __future__ import annotations

import pathlib

import healpy
import numpy
import polars
import pytest

from xmatch_rs._xmatch_rs_lib import hats as _rs_hats
from xmatch_rs.hats.index import create_index

from .conftest import random_patch


def _make_catalogue(n: int = 300) -> polars.DataFrame:
    ra, dec = random_patch(n, 10.0, 10.0, size=5.0)
    return polars.DataFrame({"a_id": numpy.arange(n), "ra": ra, "dec": dec})


@pytest.mark.parametrize("map_order", [1, 2, 4])
def test_get_healpix_map(make_hats, map_order):
    """The HEALPix map is a UInt32 series in RING order with the target counts."""

    cat = _make_catalogue()
    hats = make_hats("cat_a", cat, "a_id")

    healpix_map = _rs_hats.get_healpix_map(
        str(hats.path / "object" / "dataset"),
        map_order,
    )

    assert isinstance(healpix_map, polars.Series)
    assert healpix_map.dtype == polars.UInt32

    nside = 2**map_order
    assert healpix_map.len() == healpy.nside2npix(nside)

    ring_pixels = healpy.ang2pix(
        nside,
        cat["ra"].to_numpy(),
        cat["dec"].to_numpy(),
        lonlat=True,
    )
    expected = numpy.bincount(ring_pixels, minlength=healpy.nside2npix(nside))

    numpy.testing.assert_array_equal(healpix_map.to_numpy(), expected)


def _write_pixel(dataset: pathlib.Path, order: int, npix: int, n_rows: int):
    """Writes a pixel file with ``n_rows`` rows spread evenly inside the pixel."""

    path = dataset / f"Norder={order}" / "Dir=0" / f"Npix={npix}"
    path.mkdir(parents=True)

    pixel_size = 4 ** (29 - order)
    healpix_29 = npix * pixel_size + numpy.arange(n_rows) * (pixel_size // n_rows)
    df = polars.DataFrame(
        {
            "_healpix_29": polars.Series(healpix_29, dtype=polars.UInt64),
            "value": numpy.arange(n_rows),
        }
    )
    df.write_parquet(path / "part-0.parquet")


def test_repartition_does_not_mix_orders(tmp_path: pathlib.Path):
    """Pixels with the same Npix at a different order are not counted together."""

    dataset = tmp_path / "dataset"

    # Pixel 5 has 10 rows at order 2 and 10 rows at order 3. Only order 2 counts.
    _write_pixel(dataset, 2, 5, 10)
    _write_pixel(dataset, 3, 5, 10)

    # Pixel 6 at order 2 is over the limit.
    _write_pixel(dataset, 2, 6, 20)

    pixels = _rs_hats.repartition(
        str(dataset),
        order_0=2,
        order_1=3,
        pixels=None,
        max_rows_per_pixel=15,
        max_kilobytes_per_pixel=None,
        n_workers=1,
        progress_file=None,
        keep_initial_order_directories=False,
    )

    assert pixels == [6]

    assert (dataset / "Norder=2/Dir=0/Npix=5").exists()
    assert not (dataset / "Norder=2/Dir=0/Npix=6").exists()

    new_pixels = polars.scan_parquet(dataset / "Norder=3").collect()
    assert sorted(new_pixels["Npix"].unique().to_list()) == [5, 24, 25, 26, 27]
    assert new_pixels.filter(polars.col.Npix != 5).height == 20


@pytest.mark.parametrize("check_duplicates", [False, True])
def test_create_index_full(make_hats, tmp_path, check_duplicates):
    """The full index has one row per target with or without the duplicate check."""

    cat = _make_catalogue().with_columns(extra=polars.col.a_id * 2)
    hats = make_hats("cat_a", cat, "a_id")

    _rs_hats.create_index(
        str(hats.path),
        "a_id_index",
        "a_id",
        ["extra"],
        "full",
        check_duplicates,
        False,
        None,
        None,
    )

    index = polars.scan_parquet(
        hats.path / "a_id_index" / "dataset" / "index", hive_partitioning=False
    ).collect()
    assert index.columns == ["Norder", "Dir", "Npix", "a_id", "extra"]
    assert index.height == cat.height

    index = index.sort("a_id")
    assert index["a_id"].to_list() == cat["a_id"].to_list()
    assert (index["extra"] == index["a_id"] * 2).all()


@pytest.mark.parametrize("check_duplicates", [False, True])
def test_create_index_python(make_hats, tmp_path, check_duplicates):
    """The Python index has one row per target with or without the duplicate check."""

    cat = _make_catalogue().with_columns(extra=polars.col.a_id * 2)
    hats = make_hats("cat_a", cat, "a_id")

    index_path = tmp_path / "index"
    create_index(
        hats.path / "object" / "dataset",
        index_path,
        "a_id",
        extra_columns=["extra"],
        check_duplicates=check_duplicates,
    )

    index = polars.scan_parquet(index_path, hive_partitioning=False).collect()
    assert index.columns == ["Norder", "Dir", "Npix", "a_id", "extra"]
    assert index.height == cat.height

    index = index.sort("a_id")
    assert index["a_id"].to_list() == cat["a_id"].to_list()
    assert (index["extra"] == index["a_id"] * 2).all()


def test_create_index_duplicates(make_hats, tmp_path):
    """The duplicate check fails if an index value appears twice in a pixel."""

    cat = _make_catalogue()
    cat = cat.with_columns(dup_id=polars.col.a_id // 2)

    # Put both rows of each pair at the same position so they are in the same pixel.
    cat = cat.with_columns(
        ra=polars.col.ra.first().over("dup_id"),
        dec=polars.col.dec.first().over("dup_id"),
    )
    hats = make_hats("cat_a", cat, "a_id")

    with pytest.raises(Exception, match="Duplicate"):
        _rs_hats.create_index(
            str(hats.path),
            "dup_index",
            "dup_id",
            None,
            "full",
            True,
            False,
            None,
            None,
        )
