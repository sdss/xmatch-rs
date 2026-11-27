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
/// `index2`, by increasing distance, as returned by [`match_sky_radius`].
///
pub fn match_sky_radius_flat(
    coords1: &[(f64, f64)],
    coords2: &[(f64, f64)],
    radius: f64,
) -> Result<Vec<SkyMatch>> {
    // Initialise the k-d tree.
    let mut kdtree = KdTree::new(2);

    // Build the tree using the first set of coordinates.
    for (i, coord) in coords1.iter().enumerate() {
        kdtree.add([coord.0, coord.1], i)?;
    }

    // Rayon's collect preserves the order of the input, so the matches are sorted by index2.
    let matches = coords2
        .par_iter()
        .enumerate()
        .flat_map_iter(|(index2, coord)| {
            kdtree
                .within(&[coord.0, coord.1], radius, &spherical_distance)
                .unwrap()
                .into_iter()
                .map(move |(distance, index1)| SkyMatch {
                    index2,
                    index1: *index1,
                    distance,
                })
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
}
