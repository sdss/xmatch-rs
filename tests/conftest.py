#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: conftest.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

"""Fixtures to build small synthetic HATS catalogues and cross-match products."""

from __future__ import annotations

import itertools
import pathlib

from typing import Any, Callable

import numpy
import polars
import pytest

from xmatch_rs.hats.hats import HATS
from xmatch_rs.xmatch.xmatch import XMatch


# Epoch and pixelation order used for all the test cross-match products.
XMATCH_EPOCH = 2015.5
XMATCH_ORDER = 5

# Each test gets a different run_id. The Rust code keeps a process-global cache of
# the maximum catalogid per run_id, so sharing run_ids between tests would make them
# depend on each other.
_RUN_IDS = itertools.count(100)


def make_schema(
    name: str,
    df: polars.DataFrame,
    primary_key: str,
    pm: bool = False,
    epoch: float = 2000.0,
    parallax: bool = False,
    epoch_column: str | None = None,
) -> dict[str, Any]:
    """Returns a schema dictionary for a synthetic catalogue."""

    columns = {col: {"dtype": str(dtype)} for col, dtype in df.schema.items()}

    schema: dict[str, Any] = {
        "catalogue_name": name,
        "short_name": name,
        "url": "https://example.com",
        "primary_key": primary_key,
        "ra_column": "ra",
        "dec_column": "dec",
        "source_catalogue": {
            "data_url": "https://example.com",
            "format": "parquet",
            "columns": columns,
        },
        "database": {"table_name": name, "schema": "catalogdb"},
    }

    if pm:
        schema.update(
            pmra_column="pmra",
            pmdec_column="pmdec",
            pmra_is_cosdec=True,
            epoch=epoch,
        )

    if epoch_column is not None:
        schema.pop("epoch", None)
        schema["epoch_column"] = epoch_column

    if parallax:
        schema["parallax_column"] = "parallax"

    return schema


@pytest.fixture
def make_hats(tmp_path: pathlib.Path) -> Callable[..., HATS]:
    """Returns a function that converts a data frame to a HATS collection."""

    def _make_hats(
        name: str,
        df: polars.DataFrame,
        primary_key: str,
        **schema_kwargs,
    ) -> HATS:
        src = tmp_path / "src" / name
        src.mkdir(parents=True)
        df.write_parquet(src / "data.parquet")

        return HATS.convert_to_hats(
            src / "data.parquet",
            tmp_path / "hats" / name,
            schema=make_schema(name, df, primary_key, **schema_kwargs),
            format="parquet",
            initial_order=5,
            max_rows_per_pixel=None,
            create_density_map=False,
            show_progress=False,
            quiet=True,
            n_workers=1,
        )

    return _make_hats


@pytest.fixture
def run_id() -> int:
    """Returns a run_id unique to the test."""

    return next(_RUN_IDS)


@pytest.fixture
def make_xmatch(tmp_path: pathlib.Path) -> Callable[..., XMatch]:
    """Returns a function that creates an empty cross-match product."""

    def _make_xmatch(name: str = "xmatch") -> XMatch:
        return XMatch(
            tmp_path / name,
            version="test",
            epoch=XMATCH_EPOCH,
            pixelation_order=XMATCH_ORDER,
            quiet=True,
            logfile=False,
        )

    return _make_xmatch


def read_catalog(xm: XMatch) -> polars.DataFrame:
    """Reads the ``catalog`` dataset of a cross-match product."""

    path = xm.path / "catalog" / "dataset"
    if not path.exists() or not any(path.rglob("*.parquet")):
        return polars.DataFrame(schema={"catalogid": polars.UInt64, "added_by": str})

    return polars.scan_parquet(path, hive_partitioning=False).collect()


def read_catalog_to(xm: XMatch, name: str) -> polars.DataFrame:
    """Reads the ``catalog_to_<name>`` dataset of a cross-match product."""

    path = xm.path / f"catalog_to_{name}" / "dataset"
    if not path.exists() or not any(path.rglob("*.parquet")):
        return polars.DataFrame()

    return polars.scan_parquet(path, hive_partitioning=False).collect()


def random_patch(
    n: int,
    ra0: float,
    dec0: float,
    size: float = 1.0,
    seed: int = 42,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Returns random coordinates in a square patch of side ``2*size`` degrees.

    Points are at least ~10 arcsec apart so that 1 arcsec matches are unambiguous.

    """

    rng = numpy.random.default_rng(seed)
    grid = int(numpy.ceil(numpy.sqrt(n)))
    step = 2 * size / grid

    ii, jj = numpy.divmod(numpy.arange(n), grid)
    jitter = rng.uniform(-0.25, 0.25, (2, n)) * step

    ra = ra0 - size + (jj + 0.5) * step + jitter[0]
    dec = dec0 - size + (ii + 0.5) * step + jitter[1]

    return ra, dec


def offset_coords(
    ra: numpy.ndarray,
    dec: numpy.ndarray,
    arcsec: float,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Offsets coordinates in Dec by a number of arcsec."""

    return ra, dec + arcsec / 3600.0
