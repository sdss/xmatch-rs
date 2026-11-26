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


def test_phase_1_join_table_same_pk_names(make_hats, make_xmatch, run_id, tmp_path):
    """The right primary key is correct when both catalogues use the same pk name."""

    ra, dec = random_patch(100, 10.0, 10.0)

    # Like Gaia DR3 -> DR2: both primary keys are called source_id and the join
    # table maps one to the other. The right catalogue is joined on its pk.
    cat_l = polars.DataFrame({"source_id": numpy.arange(100), "ra": ra, "dec": dec})
    cat_r = polars.DataFrame(
        {"source_id": numpy.arange(1000, 1100), "ra": ra, "dec": dec}
    )

    join_table_path = tmp_path / "neighbourhood"
    join_table_path.mkdir()
    polars.DataFrame(
        {"dr3_source_id": numpy.arange(100), "dr2_source_id": numpy.arange(1000, 1100)}
    ).write_parquet(join_table_path / "join.parquet")

    hats_l = make_hats("cat_l", cat_l, "source_id")
    hats_r = make_hats("cat_r", cat_r, "source_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_l.path, name="cat_l", run_id=run_id)
    xm.add_catalogue(
        hats_r.path,
        name="cat_r",
        skip_phases=[2, 3],
        join_paths=[
            {
                "join_type": "join_table",
                "left_catalogue": "cat_l",
                "left_on": "source_id",
                "right_on": "source_id",
                "join_table_path": join_table_path,
                "join_table_left_on": "dr3_source_id",
                "join_table_right_on": "dr2_source_id",
            }
        ],
    )
    xm.run()

    catalog_to_r = read_catalog_to(xm, "cat_r")
    assert catalog_to_r.height == 100
    assert set(catalog_to_r["source_id"]) == set(range(1000, 1100))

    catalog_to_l = read_catalog_to(xm, "cat_l").rename({"source_id": "l_id"})
    joined = catalog_to_r.join(catalog_to_l.select("catalogid", "l_id"), on="catalogid")
    assert (joined["source_id"] == joined["l_id"] + 1000).all()


def test_phase_1_column_right_on_pk(make_hats, make_xmatch, run_id):
    """A column join can use the right primary key as the join column."""

    ra, dec = random_patch(100, 10.0, 10.0)

    # A has a column with the pk of R. The column join is A.r_ref -> R.r_id.
    cat_a = polars.DataFrame(
        {
            "a_id": numpy.arange(100),
            "ra": ra,
            "dec": dec,
            "r_ref": numpy.arange(2000, 2100),
        }
    )
    cat_r = polars.DataFrame({"r_id": numpy.arange(2000, 2100), "ra": ra, "dec": dec})

    hats_a = make_hats("cat_a", cat_a, "a_id")
    hats_r = make_hats("cat_r", cat_r, "r_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.add_catalogue(
        hats_r.path,
        name="cat_r",
        skip_phases=[2, 3],
        join_paths=[
            {"left_catalogue": "cat_a", "left_on": "r_ref", "right_on": "r_id"}
        ],
    )
    xm.run()

    catalog_to_r = read_catalog_to(xm, "cat_r")
    assert catalog_to_r.height == 100

    catalog_to_a = read_catalog_to(xm, "cat_a").select("catalogid", "a_id")
    joined = catalog_to_r.join(catalog_to_a, on="catalogid")
    assert (joined["r_id"] == joined["a_id"] + 2000).all()
