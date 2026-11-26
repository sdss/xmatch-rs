#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: test_xmatch.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

"""End-to-end tests for the cross-match phases on synthetic catalogues."""

from __future__ import annotations

import numpy
import polars

from .conftest import offset_coords, random_patch, read_catalog, read_catalog_to


def test_first_catalogue_phase_3(make_hats, make_xmatch, run_id):
    """The first catalogue only runs phase 3 and gets one catalogid per row."""

    ra, dec = random_patch(200, 10.0, 10.0)
    cat_a = polars.DataFrame({"a_id": numpy.arange(200), "ra": ra, "dec": dec})
    hats_a = make_hats("cat_a", cat_a, "a_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.run()

    assert [phase.phase for phase in xm.model.get_catalogue("cat_a").phases] == [3]

    catalog = read_catalog(xm)
    assert catalog.height == 200
    assert catalog["catalogid"].n_unique() == 200
    assert (catalog["catalogid"] // 2**53).unique().to_list() == [run_id]

    catalog_to_a = read_catalog_to(xm, "cat_a")
    assert catalog_to_a.height == 200
    assert catalog_to_a["best"].all()
    assert set(catalog_to_a["a_id"]) == set(range(200))


def test_phase_2_matches(make_hats, make_xmatch, run_id):
    """Phase 2 matches a second catalogue spatially; phase 3 adds the rest."""

    ra, dec = random_patch(200, 10.0, 10.0)
    cat_a = polars.DataFrame({"a_id": numpy.arange(200), "ra": ra, "dec": dec})

    # The first 100 targets in B are counterparts of A offset by 0.3 arcsec.
    # The other 100 are in the same patch but far from any A target.
    ra_b, dec_b = offset_coords(ra[:100], dec[:100], 0.3)
    ra_new, dec_new = random_patch(100, 10.0, 10.0, seed=1)
    ra_new, dec_new = offset_coords(ra_new, dec_new, 30)

    cat_b = polars.DataFrame(
        {
            "b_id": numpy.arange(200),
            "ra": numpy.concatenate([ra_b, ra_new]),
            "dec": numpy.concatenate([dec_b, dec_new]),
        }
    )

    hats_a = make_hats("cat_a", cat_a, "a_id")
    hats_b = make_hats("cat_b", cat_b, "b_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.add_catalogue(hats_b.path, name="cat_b", query_radius=1.0)
    xm.run()

    catalog_to_b = read_catalog_to(xm, "cat_b")

    phase_2 = catalog_to_b.filter(polars.col.added_by_phase == 2)
    assert set(phase_2["b_id"]) == set(range(100))
    assert phase_2["distance"].max() * 3600 < 0.31

    phase_3 = catalog_to_b.filter(polars.col.added_by_phase == 3)
    assert set(phase_3["b_id"]) == set(range(100, 200))

    catalog = read_catalog(xm)
    assert catalog.height == 300
    assert catalog["catalogid"].n_unique() == 300


def test_phase_1_join_table_uses_best(make_hats, make_xmatch, run_id, tmp_path):
    """Join-table paths only follow the best associations of the left catalogue."""

    ra, dec = random_patch(50, 10.0, 10.0)

    # Pairs of A targets 0.8 arcsec apart: A[2i] at P and A[2i+1] at P + 0.8".
    ra_a = numpy.repeat(ra, 2)
    dec_a = numpy.repeat(dec, 2)
    dec_a[1::2] += 0.8 / 3600.0
    cat_a = polars.DataFrame({"a_id": numpy.arange(100), "ra": ra_a, "dec": dec_a})

    # L[2i] at P + 0.2" matches A[2i] (0.2") and A[2i+1] (0.6"). L[2i+1] at
    # P + 0.9" is the closest to A[2i+1]. The L[2i] -> A[2i+1] association is not
    # best regardless of whether best is defined per target or per catalogid.
    dec_l = numpy.repeat(dec, 2)
    dec_l[0::2] += 0.2 / 3600.0
    dec_l[1::2] += 0.9 / 3600.0
    cat_l = polars.DataFrame({"l_id": numpy.arange(100), "ra": ra_a, "dec": dec_l})

    # C links to L[2i] through a join table.
    cat_c = polars.DataFrame(
        {"c_id": numpy.arange(50), "ra": ra, "dec": dec, "c_ref": numpy.arange(50)}
    )

    join_table_path = tmp_path / "join_l_c"
    join_table_path.mkdir()
    polars.DataFrame(
        {"jt_l_id": numpy.arange(0, 100, 2), "jt_c_ref": numpy.arange(50)}
    ).write_parquet(join_table_path / "join.parquet")

    hats_a = make_hats("cat_a", cat_a, "a_id")
    hats_l = make_hats("cat_l", cat_l, "l_id")
    hats_c = make_hats("cat_c", cat_c, "c_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.add_catalogue(hats_l.path, name="cat_l", query_radius=1.0, skip_phases=[3])
    xm.add_catalogue(
        hats_c.path,
        name="cat_c",
        skip_phases=[2, 3],
        join_paths=[
            {
                "join_type": "join_table",
                "left_catalogue": "cat_l",
                "left_on": "l_id",
                "right_on": "c_ref",
                "join_table_path": join_table_path,
                "join_table_left_on": "jt_l_id",
                "join_table_right_on": "jt_c_ref",
            }
        ],
    )
    xm.run()

    # Sanity check: L[2i] has a non-best association.
    catalog_to_l = read_catalog_to(xm, "cat_l")
    assert catalog_to_l.filter(~polars.col.best, polars.col.l_id % 2 == 0).height > 0

    # Each C target is only linked to the catalogid of A[2i].
    catalog_to_c = read_catalog_to(xm, "cat_c")
    assert catalog_to_c.height == 50

    catalog_to_a = read_catalog_to(xm, "cat_a").select("catalogid", "a_id")
    joined = catalog_to_c.join(catalog_to_a, on="catalogid")
    assert (joined["a_id"] == 2 * joined["c_id"]).all()
