#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: test_memory_python.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

"""Regression tests for the memory-reduction changes in the Python package."""

from __future__ import annotations

import types

import numpy
import polars
import pytest
from astropy.io import fits
from astropy.table import Table
from polars.testing import assert_frame_equal

from xmatch_rs.convert import (
    convert_to_parquet,
    read_astropy_to_dataframe,
    read_fits_to_dataframe,
)
from xmatch_rs.database.load import _write_pixel_to_database
from xmatch_rs.hats.object import add_sequential_primary_key
from xmatch_rs.hats.tools import calculate_max_proper_motion
from xmatch_rs.schema import Schema

from .conftest import make_schema, offset_coords, random_patch, read_catalog_to


def _pm_hats(df: polars.DataFrame, **schema_kwargs) -> types.SimpleNamespace:
    """Returns a minimal object with the HATS interface used for proper motions."""

    schema = Schema(make_schema("pm", df, "id", pm=True, **schema_kwargs))

    return types.SimpleNamespace(schema=schema, get_as_lazyframe=df.lazy)


@pytest.fixture
def pm_df() -> polars.DataFrame:
    # At dec=0 the total proper motion is sqrt(pmra**2 + pmdec**2) mas/yr.
    return polars.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "ra": [10.0] * 5,
            "dec": [0.0] * 5,
            "pmra": [3.0, 0.0, 6.0, 0.0, None],
            "pmdec": [4.0, 1.0, 8.0, 2.0, 1000.0],
            "ref_epoch": [2000.0, 2000.0, 2010.0, 1990.0, 2000.0],
        }
    )


def test_max_proper_motion_max(pm_df: polars.DataFrame):
    hats = _pm_hats(pm_df)

    # Maximum per year, in arcsec. The row with a null pmra is dropped.
    assert calculate_max_proper_motion(hats) == pytest.approx(10e-3)

    # Constant reference epoch.
    pm_epoch = calculate_max_proper_motion(hats, epoch=2016.0)
    assert pm_epoch == pytest.approx(10e-3 * 16)


def test_max_proper_motion_percentile(pm_df: polars.DataFrame):
    hats = _pm_hats(pm_df)

    # Total proper motions are 5, 1, 10, 2 mas/yr.
    expected = polars.Series([5.0, 1.0, 10.0, 2.0]).quantile(0.5)
    assert expected is not None

    pm = calculate_max_proper_motion(hats, percentile=50)
    assert pm == pytest.approx(expected / 1000)


def test_max_proper_motion_epoch_column(pm_df: polars.DataFrame):
    hats = _pm_hats(pm_df, epoch_column="ref_epoch")

    # pm * |epoch - ref_epoch| is 5*20, 1*20, 10*10, 2*30 mas.
    pm = calculate_max_proper_motion(hats, epoch=2020.0)
    assert pm == pytest.approx(100e-3)

    pm = calculate_max_proper_motion(hats, epoch=2020.0, percentile=0)
    assert pm == pytest.approx(20e-3)


def test_write_pixel_to_database_batches(tmp_path, monkeypatch):
    pixel_path = tmp_path / "dataset" / "Norder=2" / "Dir=0" / "Npix=10"
    pixel_path.mkdir(parents=True)

    data = polars.DataFrame(
        {
            "id": numpy.arange(10, dtype=numpy.int64),
            "_healpix_29": numpy.arange(10, dtype=numpy.uint64),
        }
    )
    data.write_parquet(pixel_path / "0.parquet")

    written: list[tuple[polars.DataFrame, str]] = []

    def write_database(self, table_name, **kwargs):
        written.append((self, table_name))

    monkeypatch.setattr(polars.DataFrame, "write_database", write_database)

    _write_pixel_to_database(
        (2, 10),
        dataset_path=tmp_path,
        database_uri="postgresql://",
        schema="catalogdb",
        table_name="test",
        load_hats_columns=True,
        chunk_size=4,
    )

    assert len(written) > 1
    assert {table_name for _, table_name in written} == {"catalogdb.test"}

    result = polars.concat([df for df, _ in written])
    expected = data.rename({"_healpix_29": "healpix_29"}).with_columns(
        polars.lit(2, dtype=polars.UInt8).alias("hats_norder"),
        polars.lit(0, dtype=polars.UInt32).alias("hats_dir"),
        polars.lit(10, dtype=polars.UInt64).alias("hats_npix"),
    )
    assert_frame_equal(result, expected)

    written.clear()
    _write_pixel_to_database(
        (2, 10),
        dataset_path=tmp_path,
        database_uri="postgresql://",
        schema="catalogdb",
        table_name="test",
        load_hats_columns=False,
    )
    assert_frame_equal(
        polars.concat([df for df, _ in written]), data.drop("_healpix_29")
    )


def test_describe(make_hats, make_xmatch, run_id):
    ra, dec = random_patch(200, 10.0, 10.0)
    cat_a = polars.DataFrame({"a_id": numpy.arange(200), "ra": ra, "dec": dec})

    ra_b, dec_b = offset_coords(ra[:100], dec[:100], 0.3)
    cat_b = polars.DataFrame({"b_id": numpy.arange(100), "ra": ra_b, "dec": dec_b})

    xm = make_xmatch()
    xm.add_catalogue(
        make_hats("cat_a", cat_a, "a_id").path, name="cat_a", run_id=run_id
    )
    xm.add_catalogue(make_hats("cat_b", cat_b, "b_id").path, name="cat_b")
    xm.run()

    describe = xm.describe()
    assert describe["catalogue"].to_list() == ["cat_a", "cat_b"]
    assert describe["n_rows"].to_list() == [200, 100]

    for name, pk in [("cat_a", "a_id"), ("cat_b", "b_id")]:
        catalog_to = read_catalog_to(xm, name)
        row = describe.filter(polars.col.catalogue == name)
        for phase in catalog_to["added_by_phase"].unique():
            in_phase = catalog_to.filter(polars.col.added_by_phase == phase)
            matched = in_phase.filter(polars.col.best)[pk].unique().len()
            assert row[f"phase_{phase}_entries"].item() == in_phase.height
            assert row[f"phase_{phase}_matched"].item() == matched


def _read_fits_reference(path) -> polars.DataFrame:
    """The previous implementation of ``read_fits_to_dataframe``."""

    data = Table.read(path)
    data.convert_bytestring_to_unicode()

    return polars.DataFrame(
        {
            col: data[col].astype(data.dtype[col].type, copy=False)
            for col in data.colnames
        }
    )


@pytest.mark.parametrize("gzip", [False, True])
def test_read_fits_to_dataframe(tmp_path, gzip: bool):
    columns = [
        fits.Column(
            name="name",
            format="8A",
            array=numpy.array([b"abc", b"de  f ", b"", b"   ", b"x"]),
        ),
        fits.Column(name="id", format="K", array=numpy.arange(1, 6)),
        fits.Column(
            name="inull", format="J", null=-99, array=numpy.array([1, -99, 3, 4, 5])
        ),
        fits.Column(
            name="u16",
            format="I",
            bzero=32768,
            array=numpy.array([0, 1, 65535, 7, 8], dtype=numpy.uint16),
        ),
        fits.Column(
            name="flux", format="D", array=numpy.array([1.5, numpy.nan, -2, 3, 0])
        ),
        fits.Column(name="f32", format="E", array=numpy.array([1.5, 2, -2, 3, 0])),
        fits.Column(
            name="flag", format="L", array=numpy.array([True, False, True, False, True])
        ),
    ]

    path = tmp_path / ("test.fits.gz" if gzip else "test.fits")
    fits.BinTableHDU.from_columns(columns).writeto(path)

    expected = polars.DataFrame(
        {
            "name": ["abc", "de  f", None, None, "x"],
            "id": [1, 2, 3, 4, 5],
            "inull": polars.Series([1, None, 3, 4, 5], dtype=polars.Int32),
            "u16": polars.Series([0, 1, 65535, 7, 8], dtype=polars.UInt16),
            "flux": [1.5, None, -2.0, 3.0, 0.0],
            "f32": polars.Series([1.5, 2, -2, 3, 0], dtype=polars.Float32),
            "flag": [True, False, True, False, True],
        }
    )

    df = read_fits_to_dataframe(path)
    assert_frame_equal(df, expected)
    assert_frame_equal(df, _read_fits_reference(path))


def test_read_astropy_to_dataframe(tmp_path):
    table = Table(
        {
            "name": ["abc", "de f", "x"],
            "id": numpy.array([1, 2, 3], dtype=numpy.int64),
            "flux": [1.5, -2.0, 3.25],
        }
    )
    path = tmp_path / "test.tbl"
    table.write(path, format="ipac")

    expected = polars.DataFrame(
        {"name": ["abc", "de f", "x"], "id": [1, 2, 3], "flux": [1.5, -2.0, 3.25]}
    )
    assert_frame_equal(read_astropy_to_dataframe(path, format="ipac"), expected)


def test_convert_direct_row_index(tmp_path):
    dfs = [
        polars.DataFrame({"ra": [1.0, 2.0, 3.0], "dec": [0.0, 1.0, 2.0]}),
        polars.DataFrame({"ra": [4.0, 5.0], "dec": [3.0, 4.0]}),
    ]
    schema = make_schema("test", dfs[0], "id")

    convert_to_parquet(
        dfs,
        tmp_path,
        schema=schema,
        mode="direct",
        add_healpix_columns=False,
        add_row_index_column=True,
    )

    files = sorted(tmp_path.glob("*.parquet"))
    assert [polars.read_parquet(ff)["id"].to_list() for ff in files] == [
        [1, 2, 3],
        [4, 5],
    ]


def test_add_sequential_primary_key(tmp_path):
    data = [
        polars.DataFrame({"ra": [1.0, 2.0, 3.0], "dec": [0.0, 1.0, 2.0]}),
        polars.DataFrame({"ra": [4.0, 5.0], "dec": [3.0, 4.0]}),
    ]
    for ii, df in enumerate(data):
        (tmp_path / f"Npix={ii}").mkdir()
        df.write_parquet(tmp_path / f"Npix={ii}" / "0.parquet")

    add_sequential_primary_key(tmp_path, "pk")

    files = sorted(tmp_path.glob("**/*"))
    assert [ff.name for ff in files if ff.is_file()] == ["0.parquet", "0.parquet"]

    for ii, (pks, df) in enumerate(zip([[1, 2, 3], [4, 5]], data)):
        result = polars.read_parquet(tmp_path / f"Npix={ii}" / "0.parquet")
        expected = df.select(
            polars.Series("pk", pks, dtype=polars.UInt64), polars.all()
        )
        assert_frame_equal(result, expected)
