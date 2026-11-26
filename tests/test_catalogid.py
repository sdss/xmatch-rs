#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: test_catalogid.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

"""Tests for the assignment of catalogids."""

from __future__ import annotations

import datetime

import polars

from xmatch_rs._xmatch_rs_lib import xmatch as xmatch_lib


RUN_ID_BIT_SHIFT = 53


def write_catalog(xm, catalogids: list[int]):
    """Writes a catalog dataset with the given catalogids in a single file."""

    n = len(catalogids)
    now = datetime.datetime.now(tz=datetime.timezone.utc)

    df = polars.DataFrame(
        {
            "catalogid": polars.Series(catalogids, dtype=polars.UInt64),
            "_healpix_29": polars.Series([0] * n, dtype=polars.UInt64),
            "ra": polars.Series([0.0] * n, dtype=polars.Float64),
            "dec": polars.Series([0.0] * n, dtype=polars.Float64),
            "pmra": polars.Series([None] * n, dtype=polars.Float32),
            "pmdec": polars.Series([None] * n, dtype=polars.Float32),
            "epoch": polars.Series([2015.5] * n, dtype=polars.Float32),
            "parallax": polars.Series([None] * n, dtype=polars.Float32),
            "added_by": ["test"] * n,
            "added_on": polars.Series([now] * n).dt.cast_time_unit("ms"),
            "version": ["test"] * n,
        }
    )

    path = xm.path / "catalog/dataset/Norder=5/Dir=0/Npix=0"
    path.mkdir(parents=True)
    df.write_parquet(path / "part-00.parquet")


def test_get_max_catalogid_run_id(make_xmatch, run_id):
    """The maximum catalogid for a run_id is not the global maximum."""

    other_run_id = run_id + 1000

    base = run_id << RUN_ID_BIT_SHIFT
    other_base = other_run_id << RUN_ID_BIT_SHIFT

    xm = make_xmatch()

    # The first row for run_id is not its maximum, and the global maximum
    # belongs to a different run_id.
    write_catalog(xm, [base + 5, base + 9, base + 3, other_base + 1])

    max_catalogid = xmatch_lib.get_max_catalogid(str(xm.path), run_id)
    assert max_catalogid == base + 9


def test_get_max_catalogid_new_run_id(make_xmatch, run_id):
    """A run_id without catalogids starts at the base for that run_id."""

    xm = make_xmatch()
    write_catalog(xm, [((run_id + 1500) << RUN_ID_BIT_SHIFT) + 1])

    max_catalogid = xmatch_lib.get_max_catalogid(str(xm.path), run_id)
    assert max_catalogid == run_id << RUN_ID_BIT_SHIFT
