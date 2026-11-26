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


def test_phase_1_multiple_join_paths(make_hats, make_xmatch, run_id, tmp_path):
    """Associations from a later join path are not duplicated."""

    ra, dec = random_patch(100, 10.0, 10.0)
    cat_a = polars.DataFrame({"a_id": numpy.arange(100), "ra": ra, "dec": dec})

    # B links to A through a column and through a join table with the same pairs.
    cat_b = polars.DataFrame(
        {
            "b_id": numpy.arange(1000, 1100),
            "ra": ra,
            "dec": dec,
            "a_ref": numpy.arange(100),
            "b_ref": numpy.arange(1000, 1100),
        }
    )

    join_table_path = tmp_path / "join_a_b"
    join_table_path.mkdir()
    polars.DataFrame(
        {"jt_a_id": numpy.arange(100), "jt_b_id": numpy.arange(1000, 1100)}
    ).write_parquet(join_table_path / "join.parquet")

    hats_a = make_hats("cat_a", cat_a, "a_id")
    hats_b = make_hats("cat_b", cat_b, "b_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.add_catalogue(
        hats_b.path,
        name="cat_b",
        skip_phases=[2, 3],
        join_paths=[
            {"left_catalogue": "cat_a", "left_on": "a_id", "right_on": "a_ref"},
            {
                "join_type": "join_table",
                "left_catalogue": "cat_a",
                "left_on": "a_id",
                "right_on": "b_ref",
                "join_table_path": join_table_path,
                "join_table_left_on": "jt_a_id",
                "join_table_right_on": "jt_b_id",
            },
        ],
    )
    xm.run()

    catalog_to_b = read_catalog_to(xm, "cat_b")
    phase_1 = catalog_to_b.filter(polars.col.added_by_phase == 1)

    assert phase_1.height == 100
    assert phase_1.select("catalogid", "b_id").is_duplicated().sum() == 0
    assert set(phase_1["b_id"]) == set(range(1000, 1100))
