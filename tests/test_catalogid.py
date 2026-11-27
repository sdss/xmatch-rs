#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: test_catalogid.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

"""Tests for the assignment of catalogids."""

from __future__ import annotations

import datetime
import subprocess
import sys

import numpy
import polars

from xmatch_rs._xmatch_rs_lib import xmatch as xmatch_lib
from xmatch_rs.xmatch.xmatch import XMatch

from .conftest import (
    XMATCH_EPOCH,
    XMATCH_ORDER,
    offset_coords,
    random_patch,
    read_catalog,
)


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


def test_get_max_catalogid_run_id_boundaries(make_xmatch, run_id):
    """Only catalogids in the bit range of the run_id are considered."""

    base = run_id << RUN_ID_BIT_SHIFT
    next_base = (run_id + 1) << RUN_ID_BIT_SHIFT

    xm = make_xmatch()

    # The largest catalogid for run_id, the first one for the next run_id (which is
    # the global maximum), and the last one for the previous run_id.
    write_catalog(xm, [base, next_base - 1, next_base, base - 1])

    max_catalogid = xmatch_lib.get_max_catalogid(str(xm.path), run_id)
    assert max_catalogid == next_base - 1


BUILD_XMATCH_SCRIPT = """
import sys
from xmatch_rs.xmatch.xmatch import XMatch

path, hats_path, run_id, epoch, order = sys.argv[1:]
xm = XMatch(
    path,
    version="test",
    epoch=float(epoch),
    pixelation_order=int(order),
    quiet=True,
    logfile=False,
)
xm.add_catalogue(hats_path, name="cat_a", run_id=int(run_id))
xm.run()
"""


def test_catalogid_cache_two_products(make_hats, make_xmatch, run_id, tmp_path):
    """Catalogids are unique when two xmatch products are run in one process."""

    ra, dec = random_patch(200, 10.0, 10.0)
    cat_a = polars.DataFrame({"a_id": numpy.arange(200), "ra": ra, "dec": dec})
    hats_a = make_hats("cat_a", cat_a, "a_id")

    # A small catalogue for the first product.
    cat_s = cat_a.head(50).rename({"a_id": "s_id"})
    hats_s = make_hats("cat_s", cat_s, "s_id")

    # Even B targets are 0.3 arcsec from A and match in phase 2. Odd ones are
    # 30 arcsec away and are added in phase 3.
    ra_b, dec_b = offset_coords(ra, dec, 0.3)
    _, dec_far = offset_coords(ra, dec, 30)
    dec_b[1::2] = dec_far[1::2]
    cat_b = polars.DataFrame({"b_id": numpy.arange(200), "ra": ra_b, "dec": dec_b})
    hats_b = make_hats("cat_b", cat_b, "b_id")

    # Product 2 is created in a different process so that this process does not
    # know about its catalogids.
    path_2 = tmp_path / "xmatch_2"
    subprocess.run(
        [
            sys.executable,
            "-c",
            BUILD_XMATCH_SCRIPT,
            str(path_2),
            str(hats_a.path),
            str(run_id),
            str(XMATCH_EPOCH),
            str(XMATCH_ORDER),
        ],
        check=True,
        capture_output=True,
    )

    # Product 1 uses the same run_id in this process.
    xm_1 = make_xmatch("xmatch_1")
    xm_1.add_catalogue(hats_s.path, name="cat_s", run_id=run_id)
    xm_1.run()
    assert read_catalog(xm_1).height == 50

    # Now add B to product 2 in this process.
    xm_2 = XMatch(path_2, quiet=True, logfile=False)
    xm_2.add_catalogue(hats_b.path, name="cat_b", query_radius=1.0)
    xm_2.run()

    catalog_2 = read_catalog(xm_2)
    assert catalog_2.height == 300
    assert catalog_2["catalogid"].n_unique() == 300
