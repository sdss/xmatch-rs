#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: test_xmatch.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

"""End-to-end tests for the cross-match phases on synthetic catalogues."""

from __future__ import annotations

import numpy
import polars
import pytest

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


def test_delete_catalogue_with_dependents(make_hats, make_xmatch, run_id):
    """Deleting a catalogue that later catalogues depend on requires force=True."""

    ra, dec = random_patch(100, 10.0, 10.0)
    cat_a = polars.DataFrame({"a_id": numpy.arange(100), "ra": ra, "dec": dec})

    # Even B targets match A in phase 2. Odd ones are new.
    ra_b, dec_b = offset_coords(ra, dec, 0.3)
    _, dec_far = offset_coords(ra, dec, 30)
    dec_b[1::2] = dec_far[1::2]
    cat_b = polars.DataFrame({"b_id": numpy.arange(100), "ra": ra_b, "dec": dec_b})

    hats_a = make_hats("cat_a", cat_a, "a_id")
    hats_b = make_hats("cat_b", cat_b, "b_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.add_catalogue(hats_b.path, name="cat_b", query_radius=1.0)
    xm.run()

    # catalog_to_cat_b points to catalogids added by cat_a, so cat_a cannot be
    # deleted while cat_b is there.
    with pytest.raises(RuntimeError, match="processed after it"):
        xm.delete_catalogue_data("cat_a")

    assert read_catalog(xm).filter(polars.col.added_by == "cat_a").height == 100

    # Phase 1 and 2 deletions do not touch catalog and are allowed.
    xm.delete_catalogue_data("cat_a", phase=1)

    # Deleting the last catalogue is allowed, and then cat_a can be deleted.
    xm.delete_catalogue_data("cat_b", delete_from_schema=True)
    assert read_catalog(xm).filter(polars.col.added_by == "cat_b").height == 0

    xm.delete_catalogue_data("cat_a")
    assert read_catalog(xm).filter(polars.col.added_by == "cat_a").height == 0


def test_delete_catalogue_force(make_hats, make_xmatch, run_id):
    """force=True keeps the previous behaviour."""

    ra, dec = random_patch(50, 10.0, 10.0)
    cat_a = polars.DataFrame({"a_id": numpy.arange(50), "ra": ra, "dec": dec})
    ra_b, dec_b = offset_coords(ra, dec, 0.3)
    cat_b = polars.DataFrame({"b_id": numpy.arange(50), "ra": ra_b, "dec": dec_b})

    hats_a = make_hats("cat_a", cat_a, "a_id")
    hats_b = make_hats(
        "cat_b",
        cat_b,
        "b_id",
    )

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.add_catalogue(hats_b.path, name="cat_b", query_radius=1.0, skip_phases=[3])
    xm.run()

    xm.delete_catalogue_data("cat_a", force=True)
    assert read_catalog(xm).filter(polars.col.added_by == "cat_a").height == 0
    assert not xm.model.get_catalogue("cat_a").complete


def test_phase_1_left_catalogue_partial_footprint(make_hats, make_xmatch, run_id):
    """Phase 1 works when the left catalogue does not cover all the xmatch pixels."""

    ra_a, dec_a = random_patch(100, 10.0, 10.0)
    cat_a = polars.DataFrame({"a_id": numpy.arange(100), "ra": ra_a, "dec": dec_a})

    # B covers a different patch, so the catalog dataset has pixels with no
    # catalog_to_cat_a entries.
    ra_b, dec_b = random_patch(100, 200.0, -30.0, seed=2)
    cat_b = polars.DataFrame({"b_id": numpy.arange(100), "ra": ra_b, "dec": dec_b})

    # C links to A through a column. It also has unlinked targets in B's patch,
    # where there are no catalog_to_cat_a entries.
    cat_c = polars.DataFrame(
        {
            "c_id": numpy.arange(200),
            "ra": numpy.concatenate([ra_a, ra_b]),
            "dec": numpy.concatenate([dec_a, dec_b]),
            "a_ref": numpy.concatenate([numpy.arange(100), numpy.full(100, -1)]),
        }
    )

    hats_a = make_hats("cat_a", cat_a, "a_id")
    hats_b = make_hats("cat_b", cat_b, "b_id")
    hats_c = make_hats("cat_c", cat_c, "c_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.add_catalogue(hats_b.path, name="cat_b", skip_phases=[2])
    xm.add_catalogue(
        hats_c.path,
        name="cat_c",
        skip_phases=[2, 3],
        join_paths=[
            {"left_catalogue": "cat_a", "left_on": "a_id", "right_on": "a_ref"}
        ],
    )
    xm.run()

    catalog_to_c = read_catalog_to(xm, "cat_c")
    assert set(catalog_to_c["c_id"]) == set(range(100))
    assert (catalog_to_c["added_by_phase"] == 1).all()


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


def test_phase_1_join_table_left_on_not_pk(make_hats, make_xmatch, run_id, tmp_path):
    """A join table can join on a left catalogue column that is not the primary key."""

    ra, dec = random_patch(100, 10.0, 10.0)
    cat_a = polars.DataFrame(
        {
            "a_id": numpy.arange(100),
            "ra": ra,
            "dec": dec,
            "a_name": numpy.arange(5000, 5100),
        }
    )

    cat_c = polars.DataFrame(
        {
            "c_id": numpy.arange(100),
            "ra": ra,
            "dec": dec,
            "c_ref": numpy.arange(100),
        }
    )

    # The join table maps the A non-pk column a_name to C.
    join_table_path = tmp_path / "join_a_c"
    join_table_path.mkdir()
    polars.DataFrame(
        {"jt_a_name": numpy.arange(5000, 5100), "jt_c_ref": numpy.arange(100)}
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
                "left_catalogue": "cat_a",
                "left_on": "a_name",
                "right_on": "c_ref",
                "join_table_path": join_table_path,
                "join_table_left_on": "jt_a_name",
                "join_table_right_on": "jt_c_ref",
            }
        ],
    )
    xm.run()

    catalog_to_c = read_catalog_to(xm, "cat_c")
    assert catalog_to_c.height == 100

    # Each C target is linked to the catalogid of the A target with the same index.
    catalog_to_a = read_catalog_to(xm, "cat_a").select("catalogid", "a_id")
    joined = catalog_to_c.join(catalog_to_a, on="catalogid")
    assert (joined["c_id"] == joined["a_id"]).all()


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


def test_get_as_lazyframe(make_hats, make_xmatch, run_id):
    """get_as_lazyframe() reads the datasets written by the cross-match."""

    ra, dec = random_patch(100, 10.0, 10.0)
    cat_a = polars.DataFrame(
        {
            "a_id": numpy.arange(100),
            "ra": ra,
            "dec": dec,
            "pmra": numpy.zeros(100),
            "pmdec": numpy.zeros(100),
        }
    )
    hats_a = make_hats("cat_a", cat_a, "a_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)

    # Before running, the frames are empty but have the right columns.
    assert xm.get_as_lazyframe().collect().height == 0
    assert "a_id" in xm.get_as_lazyframe("cat_a").collect_schema()

    xm.run()

    catalog = xm.get_as_lazyframe().collect()
    assert catalog.height == 100
    assert catalog.equals(read_catalog(xm).select(catalog.columns))

    catalog_to_a = xm.get_as_lazyframe("cat_a").collect()
    assert catalog_to_a.height == 100
    assert catalog_to_a.columns[:2] == ["catalogid", "a_id"]
    assert set(catalog_to_a["a_id"]) == set(range(100))


def test_phase_3_catalog_and_catalog_to_consistent(make_hats, make_xmatch, run_id):
    """catalog and catalog_to_<cat> agree on the catalogid of each new target."""

    ra, dec = random_patch(400, 10.0, 10.0, size=2.0)
    cat_a = polars.DataFrame({"a_id": numpy.arange(400), "ra": ra, "dec": dec})

    # Even B targets match A in phase 2. Odd ones are new and added in phase 3.
    ra_b, dec_b = offset_coords(ra, dec, 0.3)
    _, dec_far = offset_coords(ra, dec, 30)
    dec_b[1::2] = dec_far[1::2]
    cat_b = polars.DataFrame({"b_id": numpy.arange(400), "ra": ra_b, "dec": dec_b})

    hats_a = make_hats("cat_a", cat_a, "a_id")
    hats_b = make_hats("cat_b", cat_b, "b_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.add_catalogue(hats_b.path, name="cat_b", query_radius=1.0)
    xm.run()

    catalog = read_catalog(xm).filter(polars.col.added_by == "cat_b")
    catalog_to_b = read_catalog_to(xm, "cat_b").filter(polars.col.added_by_phase == 3)
    assert catalog.height == 200
    assert set(catalog_to_b["b_id"]) == set(range(1, 400, 2))

    # The coordinates in catalog for each catalogid are those of the B target that
    # catalog_to_cat_b associates with it.
    joined = catalog.join(catalog_to_b.select("catalogid", "b_id"), on="catalogid")
    assert joined.height == 200

    joined = joined.join(cat_b, on="b_id", suffix="_b")
    numpy.testing.assert_allclose(joined["ra"], joined["ra_b"], atol=1e-9)
    numpy.testing.assert_allclose(joined["dec"], joined["dec_b"], atol=1e-9)


def test_phase_2_best_per_catalogid(make_hats, make_xmatch, run_id):
    """In phase 2, each catalogid has a single best match: the closest target."""

    ra, dec = random_patch(100, 10.0, 10.0)
    cat_a = polars.DataFrame({"a_id": numpy.arange(100), "ra": ra, "dec": dec})

    # Two B targets near each A target, at 0.2 and 0.5 arcsec. Both match the same
    # catalogid. B[2i] is the closest.
    _, dec_near = offset_coords(ra, dec, 0.2)
    _, dec_far = offset_coords(ra, dec, 0.5)
    dec_b = numpy.empty(200)
    dec_b[0::2] = dec_near
    dec_b[1::2] = dec_far

    cat_b = polars.DataFrame(
        {"b_id": numpy.arange(200), "ra": numpy.repeat(ra, 2), "dec": dec_b}
    )

    hats_a = make_hats("cat_a", cat_a, "a_id")
    hats_b = make_hats("cat_b", cat_b, "b_id")

    xm = make_xmatch()
    xm.add_catalogue(hats_a.path, name="cat_a", run_id=run_id)
    xm.add_catalogue(hats_b.path, name="cat_b", query_radius=1.0, skip_phases=[3])
    xm.run()

    catalog_to_b = read_catalog_to(xm, "cat_b")
    assert catalog_to_b.height == 200
    assert catalog_to_b["catalogid"].n_unique() == 100

    best = catalog_to_b.filter(polars.col.best)
    assert best.height == 100
    assert best["catalogid"].n_unique() == 100
    assert (best["b_id"] % 2 == 0).all()


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
