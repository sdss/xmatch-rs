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
