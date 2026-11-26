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


def test_phase_2_new_sky_area(make_hats, make_xmatch, run_id):
    """Phase 2 does not fail for catalogue pixels with no xmatch data around them."""

    ra, dec = random_patch(100, 10.0, 10.0)
    cat_a = polars.DataFrame({"a_id": numpy.arange(100), "ra": ra, "dec": dec})

    # B has counterparts of all of A plus 100 targets in a far patch of sky that
    # no previous catalogue covers.
    ra_b, dec_b = offset_coords(ra, dec, 0.3)
    ra_far, dec_far = random_patch(100, 200.0, -30.0, seed=2)

    cat_b = polars.DataFrame(
        {
            "b_id": numpy.arange(200),
            "ra": numpy.concatenate([ra_b, ra_far]),
            "dec": numpy.concatenate([dec_b, dec_far]),
        }
    )

    hats_a = make_hats("cat_a", cat_a, "a_id")
    hats_b = make_hats("cat_b", cat_b, "b_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.add_catalogue(hats_b.path, name="cat_b", query_radius=1.0, skip_phases=[3])
    xm.run()

    catalog_to_b = read_catalog_to(xm, "cat_b")
    assert set(catalog_to_b["b_id"]) == set(range(100))
    assert (catalog_to_b["added_by_phase"] == 2).all()


def test_phase_3_new_sky_area(make_hats, make_xmatch, run_id):
    """Phase 3 adds targets in pixels where nothing matched in phases 1 or 2."""

    ra, dec = random_patch(100, 10.0, 10.0)
    cat_a = polars.DataFrame({"a_id": numpy.arange(100), "ra": ra, "dec": dec})

    # B links to all of A through a column join, and has 100 extra targets in a
    # far patch of sky that has no catalog_to_cat_b entries after phase 1.
    ra_far, dec_far = random_patch(100, 200.0, -30.0, seed=2)

    cat_b = polars.DataFrame(
        {
            "b_id": numpy.arange(200),
            "ra": numpy.concatenate([ra, ra_far]),
            "dec": numpy.concatenate([dec, dec_far]),
            "a_ref": numpy.concatenate([numpy.arange(100), numpy.full(100, -1)]),
        }
    )

    hats_a = make_hats("cat_a", cat_a, "a_id")
    hats_b = make_hats("cat_b", cat_b, "b_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.add_catalogue(
        hats_b.path,
        name="cat_b",
        skip_phases=[2],
        join_paths=[
            {"left_catalogue": "cat_a", "left_on": "a_id", "right_on": "a_ref"}
        ],
    )
    xm.run()

    catalog_to_b = read_catalog_to(xm, "cat_b")
    assert set(catalog_to_b["b_id"]) == set(range(200))

    phase_1 = catalog_to_b.filter(polars.col.added_by_phase == 1)
    assert set(phase_1["b_id"]) == set(range(100))

    phase_3 = catalog_to_b.filter(polars.col.added_by_phase == 3)
    assert set(phase_3["b_id"]) == set(range(100, 200))

    catalog = read_catalog(xm)
    assert catalog.filter(polars.col.added_by == "cat_b").height == 100


def test_phase_3_proper_motions(make_hats, make_xmatch, run_id):
    """Phase 3 propagates coordinates and _healpix_29 for catalogues with pm."""

    ra, dec = random_patch(100, 10.0, 10.0)
    pmdec = numpy.full(100, 100.0)  # mas/yr

    cat_a = polars.DataFrame(
        {
            "a_id": numpy.arange(100),
            "ra": ra,
            "dec": dec,
            "pmra": numpy.zeros(100),
            "pmdec": pmdec,
        }
    )
    hats_a = make_hats("cat_a", cat_a, "a_id", pm=True, epoch=2000.0)

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.run()

    catalog = read_catalog(xm).join(
        read_catalog_to(xm, "cat_a").select("catalogid", "a_id"),
        on="catalogid",
    )
    catalog = catalog.sort("a_id")
    assert catalog.height == 100

    # 15.5 years at 100 mas/yr in Dec.
    expected_dec = dec + 15.5 * 100.0 / 1000.0 / 3600.0
    numpy.testing.assert_allclose(catalog["dec"].to_numpy(), expected_dec, atol=1e-9)
    numpy.testing.assert_allclose(catalog["ra"].to_numpy(), ra, atol=1e-9)

    # _healpix_29 must correspond to the propagated coordinates.
    order_29_size = 0.0004 / 3600.0  # ~0.4 mas in deg.
    for row in catalog.head(10).iter_rows(named=True):
        lon, lat = _healpix_29_center(row["_healpix_29"])
        assert abs(lat - row["dec"]) < order_29_size


def _healpix_29_center(pixel: int) -> tuple[float, float]:
    """Returns the centre of an order 29 HEALPix pixel in degrees."""

    import healpy

    theta, phi = healpy.pix2ang(2**29, pixel, nest=True)
    return float(numpy.degrees(phi)), float(90.0 - numpy.degrees(theta))


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


def test_parallax_column(make_hats, make_xmatch, run_id):
    """The parallax column of the catalogue is copied to catalog."""

    ra, dec = random_patch(100, 10.0, 10.0)
    parallax = numpy.linspace(0.1, 10.0, 100)
    cat_a = polars.DataFrame(
        {"a_id": numpy.arange(100), "ra": ra, "dec": dec, "parallax": parallax}
    )
    hats_a = make_hats("cat_a", cat_a, "a_id", parallax=True)

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    assert xm.model.get_catalogue("cat_a").parallax_column == "parallax"

    xm.run()

    catalog = read_catalog(xm).join(
        read_catalog_to(xm, "cat_a").select("catalogid", "a_id"),
        on="catalogid",
    )
    catalog = catalog.sort("a_id")

    assert catalog["parallax"].null_count() == 0
    numpy.testing.assert_allclose(catalog["parallax"].to_numpy(), parallax, rtol=1e-6)
