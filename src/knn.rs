/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-10
 *  @Filename: knn.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::Result;
use cdshealpix::haversine_dist;
use kdtree::KdTree;
use rayon::prelude::*;

/// Calculates the spherical distance between two points. Units are in radians.
#[inline]
pub fn spherical_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    let (ra1, dec1) = (a[0], a[1]);
    let (ra2, dec2) = (b[0], b[1]);

    haversine_dist(ra1, dec1, ra2, dec2)
}

/// Matches two sets of spherical coordinates using a k-nearest neighbors search with Haversine distance.
///
/// This function is a simple wrapper arond the [kdtree-rs](https://github.com/mrhooray/kdtree-rs)
/// crate with the Haversine distance as the metric. Tree lookups are parallelised using rayon.
///
/// # Arguments
/// - `coords1`: A vector of tuples containing the right ascension and declination
///     of the first set of coordinates. Units are in radians.
/// - `coords2`: A vector of tuples containing the right ascension and declination
///     of the second set of coordinates. Units are in radians.
/// - `k`: The number of nearest neighbors to search for.
///
/// # Returns
/// A vector with the same length as `coords2`, in which the i-th element is a vector of tuples for
/// all the neighbours of `coords2[i]` in `coords1` up to the k-th neighbour. Each tuple contains
/// the distance to the neighbour and the index of the neighbour in `coords1`.
///
pub fn match_sky(
    coords1: &Vec<(f64, f64)>,
    coords2: &Vec<(f64, f64)>,
    k: usize,
) -> Result<Vec<Vec<(f64, usize)>>> {
    // Initialise the k-d tree.
    let mut kdtree = KdTree::new(2);

    // Build the tree using the first set of coordinates.
    for (i, coord) in coords1.iter().enumerate() {
        kdtree.add([coord.0, coord.1], i)?;
    }

    // For each coordinate in the second set, find the k nearest neighbors in the first set.
    // We use rayon to parallelise the lookups.
    let matches = coords2
        .par_iter()
        .map(|coord| {
            let neighbours = kdtree
                .nearest(&[coord.0, coord.1], k, &spherical_distance)
                .unwrap();

            neighbours
                .into_iter()
                .map(|(dist, idx)| (dist, *idx))
                .collect()
        })
        .collect();

    Ok(matches)
}

/// Matches two sets of spherical coordinates using a radius search with Haversine distance.
///
/// This function is a simple wrapper arond the [kdtree-rs](https://github.com/mrhooray/kdtree-rs)
/// crate with the Haversine distance as the metric. Tree lookups are parallelised using rayon.
///
/// # Arguments
/// - `coords1`: A vector of tuples containing the right ascension and declination
///     of the first set of coordinates. Units are in radians.
/// - `coords2`: A vector of tuples containing the right ascension and declination
///     of the second set of coordinates. Units are in radians.
/// - `radius`: The radius within which to search for neighbors. Units are in radians.
///
/// # Returns
/// A vector with the same length as `coords2`, in which the i-th element is a vector of tuples for
/// all the neighbours of `coords2[i]` in `coords1` within the specified radius. Each tuple contains
/// the distance to the neighbour and the index of the neighbour in `coords1`.
///
pub fn match_sky_radius(
    coords1: &Vec<(f64, f64)>,
    coords2: &Vec<(f64, f64)>,
    radius: f64,
) -> Result<Vec<Vec<(f64, usize)>>> {
    // Initialise the k-d tree.
    let mut kdtree = KdTree::new(2);

    // Build the tree using the first set of coordinates.
    for (i, coord) in coords1.iter().enumerate() {
        kdtree.add([coord.0, coord.1], i)?;
    }

    // For each coordinate in the second set, find the neighbors within the radius in the first set.
    let matches = coords2
        .par_iter()
        .map(|coord| {
            let neighbours = kdtree
                .within(&[coord.0, coord.1], radius, &spherical_distance)
                .unwrap();
            neighbours
                .into_iter()
                .map(|(dist, idx)| (dist, *idx))
                .collect()
        })
        .collect();

    Ok(matches)
}

/// A single match returned by [`match_sky_radius_flat`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyMatch {
    /// Index of the coordinate in the second set (the query).
    pub index2: usize,
    /// Index of the neighbour in the first set.
    pub index1: usize,
    /// Distance between the two coordinates, in radians.
    pub distance: f64,
}

/// Same as [`match_sky_radius`] but returns a flat vector of matches.
///
/// [`match_sky_radius`] allocates one vector per coordinate in `coords2`, even if it has no
/// matches. This function only stores the matches. They are ordered by `index2` and, for each
/// `index2`, by increasing distance, as returned by [`match_sky_radius`]. Ties in distance are
/// broken by `index1`.
///
/// The k-d tree is built on the smaller of the two sets of coordinates, which reduces the memory
/// used by the tree. The distance is symmetric so the matches do not depend on which set is
/// used to build the tree.
///
pub fn match_sky_radius_flat(
    coords1: &[(f64, f64)],
    coords2: &[(f64, f64)],
    radius: f64,
) -> Result<Vec<SkyMatch>> {
    let mut matches = if coords1.len() <= coords2.len() {
        radius_search(
            coords1,
            coords2,
            radius,
            |tree_index, query_index, distance| SkyMatch {
                index2: query_index,
                index1: tree_index,
                distance,
            },
        )?
    } else {
        radius_search(
            coords2,
            coords1,
            radius,
            |tree_index, query_index, distance| SkyMatch {
                index2: tree_index,
                index1: query_index,
                distance,
            },
        )?
    };

    // Sort by index2, then by distance and index1. If the tree was built on coords1 the
    // matches are already sorted by index2 and distance, but sorting also makes the order of
    // ties deterministic.
    matches.par_sort_unstable_by(|a, b| {
        a.index2
            .cmp(&b.index2)
            .then(a.distance.total_cmp(&b.distance))
            .then(a.index1.cmp(&b.index1))
    });

    Ok(matches)
}

/// Builds a k-d tree with `tree_coords` and finds all the neighbours of each coordinate in
/// `query_coords` within `radius`. `to_match` receives the index in `tree_coords`, the index in
/// `query_coords` and the distance, and returns the match.
fn radius_search<F>(
    tree_coords: &[(f64, f64)],
    query_coords: &[(f64, f64)],
    radius: f64,
    to_match: F,
) -> Result<Vec<SkyMatch>>
where
    F: Fn(usize, usize, f64) -> SkyMatch + Sync,
{
    // Initialise the k-d tree.
    let mut kdtree = KdTree::new(2);

    for (i, coord) in tree_coords.iter().enumerate() {
        kdtree.add([coord.0, coord.1], i)?;
    }

    let matches = query_coords
        .par_iter()
        .enumerate()
        .flat_map_iter(|(query_index, coord)| {
            kdtree
                .within(&[coord.0, coord.1], radius, &spherical_distance)
                .unwrap()
                .into_iter()
                .map(|(distance, tree_index)| to_match(*tree_index, query_index, distance))
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_sky_radius_flat_matches_nested() {
        let to_rad = |ra: f64, dec: f64| (ra.to_radians(), dec.to_radians());

        let coords1 = vec![
            to_rad(10.0, 10.0),
            to_rad(10.0, 10.0001),
            to_rad(20.0, -5.0),
            to_rad(10.001, 10.0),
        ];
        let coords2 = vec![
            to_rad(10.00001, 10.0),
            to_rad(50.0, 50.0),
            to_rad(20.0, -5.00001),
        ];
        let radius = (1.0f64 / 3600.0).to_radians();

        let nested = match_sky_radius(&coords1, &coords2, radius).unwrap();
        let flat = match_sky_radius_flat(&coords1, &coords2, radius).unwrap();

        let expected = nested
            .iter()
            .enumerate()
            .flat_map(|(index2, matches)| {
                matches.iter().map(move |(distance, index1)| SkyMatch {
                    index2,
                    index1: *index1,
                    distance: *distance,
                })
            })
            .collect::<Vec<_>>();

        assert_eq!(flat, expected);

        // Target 0 matches 0 and 1 (closest first), target 1 has no matches, target 2 matches 2.
        let pairs = flat
            .iter()
            .map(|m| (m.index2, m.index1))
            .collect::<Vec<_>>();
        assert_eq!(pairs, vec![(0, 0), (0, 1), (2, 2)]);
    }

    /// Returns pseudo-random coordinates in radians in a small patch around the RA=0 line.
    fn random_coords(n: usize, seed: u64) -> Vec<(f64, f64)> {
        // Simple linear congruential generator to avoid adding a dependency.
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        (0..n)
            .map(|_| {
                // RA in [359.99, 360.01) wrapped to [0, 360), Dec in [-0.01, 0.01).
                let ra = (359.99 + next() * 0.02) % 360.0;
                let dec = -0.01 + next() * 0.02;
                (ra.to_radians(), dec.to_radians())
            })
            .collect()
    }

    #[test]
    fn test_match_sky_radius_flat_tree_side() {
        // The radius is large enough that each target has several neighbours.
        let radius = (5.0f64 / 3600.0).to_radians();

        let small = random_coords(500, 1);
        let large = random_coords(3000, 2);

        let expected = |coords1: &Vec<(f64, f64)>, coords2: &Vec<(f64, f64)>| {
            let mut expected = match_sky_radius(coords1, coords2, radius)
                .unwrap()
                .iter()
                .enumerate()
                .flat_map(|(index2, matches)| {
                    matches.iter().map(move |(distance, index1)| SkyMatch {
                        index2,
                        index1: *index1,
                        distance: *distance,
                    })
                })
                .collect::<Vec<_>>();
            expected.sort_by(|a, b| {
                a.index2
                    .cmp(&b.index2)
                    .then(a.distance.total_cmp(&b.distance))
                    .then(a.index1.cmp(&b.index1))
            });
            expected
        };

        // Tree built on coords1 (smaller) and on coords2 (smaller).
        for (coords1, coords2) in [(&small, &large), (&large, &small)] {
            let flat = match_sky_radius_flat(coords1, coords2, radius).unwrap();
            let expected = expected(coords1, coords2);

            assert!(flat.len() > coords1.len().min(coords2.len()));
            assert_eq!(flat, expected);
        }
    }
}
