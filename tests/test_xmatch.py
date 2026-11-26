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


def test_phase_1_gaia_neighbourhood_best(make_hats, make_xmatch, run_id, tmp_path):
    """Only the highest-score neighbour of each target is marked as best."""

    ra, dec = random_patch(100, 10.0, 10.0)

    # Pairs of A targets 5 arcsec apart. A[2i] and A[2i+1] are both neighbours of C[i].
    ra_a = numpy.repeat(ra, 2)
    dec_a = numpy.repeat(dec, 2)
    dec_a[1::2] += 5 / 3600.0
    cat_a = polars.DataFrame({"a_id": numpy.arange(200), "ra": ra_a, "dec": dec_a})

    cat_c = polars.DataFrame(
        {
            "c_id": numpy.arange(100),
            "ra": ra,
            "dec": dec,
            "c_ref": numpy.arange(100),
        }
    )

    join_table_path = tmp_path / "neighbourhood"
    join_table_path.mkdir()
    polars.DataFrame(
        {
            "jt_a_id": numpy.arange(200),
            "jt_c_id": numpy.repeat(numpy.arange(100), 2),
            "score": numpy.tile([0.5, 0.9], 100),
        }
    ).write_parquet(join_table_path / "join.parquet")

    hats_a = make_hats("cat_a", cat_a, "a_id")
    hats_c = make_hats("cat_c", cat_c, "c_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.add_catalogue(
        hats_c.path,
        name="cat_c",
        skip_phases=[2, 3],
        join_paths=[
            {
                "join_type": "join_table",
                "join_table_type": "gaia_neighbourhood",
                "left_catalogue": "cat_a",
                "left_on": "a_id",
                "right_on": "c_ref",
                "join_table_path": join_table_path,
                "join_table_left_on": "jt_a_id",
                "join_table_right_on": "jt_c_id",
            }
        ],
    )
    xm.run()

    # Both neighbours are kept but only one is best for each C target.
    catalog_to_c = read_catalog_to(xm, "cat_c")
    assert catalog_to_c.height == 200

    best = catalog_to_c.filter(polars.col.best)
    assert best.height == 100
    assert set(best["c_id"]) == set(range(100))

    # The best neighbour is the one with the highest score (A[2i+1]).
    catalog_to_a = read_catalog_to(xm, "cat_a").select("catalogid", "a_id")
    best_a_id = best.join(catalog_to_a, on="catalogid").sort("c_id")["a_id"]
    assert (best_a_id == 2 * numpy.arange(100) + 1).all()
