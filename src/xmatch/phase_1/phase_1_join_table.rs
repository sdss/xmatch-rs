/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-10-26
 *  @Filename: phase_1_join_table.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::Result;
use polars::prelude::*;
use std::sync::Mutex;
use std::thread::available_parallelism;

use crate::hats::dataset::{AsDataset, Dataset};
use crate::tools::to_pl_path;
use crate::xmatch;
use crate::xmatch::models::{JoinDirection, JoinPathModel, JoinPathTableType, XMatchData};
use crate::xmatch::tools::get_xmatch_path;

use super::phase_1_column::{
    Phase1Datasets, RIGHT_ON_COLUMN, RIGHT_PK_COLUMN, remove_already_xmatched,
};

/// Reserved name for the join table right column when ranking the candidates of a target.
const JOIN_TABLE_RIGHT_ON_COLUMN: &str = "__xmatch_join_table_right_on";
/// Reserved name for the column that flags the candidates whose catalogid is in the pixel
/// being processed.
const CENTRAL_COLUMN: &str = "__xmatch_central";

// Process one join path on the primary key.
// Here "left" is the catalogue that has already been cross-matched and "right" is the
// new catalogue being processed.
pub(crate) fn xmatch_phase1_join_type_table(
    xmatch_data: &XMatchData,
    join_path_model: &JoinPathModel,
) -> Result<()> {
    let log_prefix = format!(
        "[{}] - Phase 1 -",
        xmatch_data.catalogue_name.to_uppercase()
    );

    // Collect join table information. We just unwrap here since we have verified that the
    // fields are not none.
    let join_table_path = join_path_model.join_table_path.clone().unwrap();
    let join_table_left_on = join_path_model.join_table_left_on.clone().unwrap();
    let join_table_right_on = join_path_model.join_table_right_on.clone().unwrap();
    let join_table_type = join_path_model.join_table_type.clone();

    // Log the join path information.
    log::info!(
        "{} Processing join path of type 'join_table' with join table in '{}'.",
        log_prefix,
        join_table_path.clone().to_str().unwrap()
    );
    log::debug!(
        "{} Left table: {}.",
        log_prefix,
        join_path_model.left_catalogue
    );
    log::debug!(
        "{} Left join column: {}.",
        log_prefix,
        join_path_model.left_on
    );
    log::debug!(
        "{} Right join column: {}.",
        log_prefix,
        join_path_model.right_on
    );
    log::debug!("{} Join table left_on: {}.", log_prefix, join_table_left_on);
    log::debug!(
        "{} Join table right_on: {}.",
        log_prefix,
        join_table_right_on
    );
    if join_table_type.is_some() {
        log::debug!(
            "{} Join table type: {:?}.",
            log_prefix,
            serde_json::to_string(&join_table_type.unwrap_or(JoinPathTableType::Null))
                .unwrap()
                .trim_matches('"')
        );
    }

    check_join_table_ranking(join_path_model)?;

    if !join_path_model.use_partitioning {
        log::debug!(
            "{} Partitioning is disabled. No progress bar will be shown.",
            log_prefix
        );

        // Call the "pixel" function without a pixel, which will process all the joins without
        // partitioning.
        let Some(datasets) = Phase1Datasets::new(xmatch_data, join_path_model)? else {
            return Ok(());
        };
        xmatch_phase1_join_type_table_pixel(None, xmatch_data, join_path_model, &datasets).unwrap();
        return Ok(());
    }

    // Get the list of pixels in the from catalogue.
    let xmatch_dataset = Dataset::new(
        get_xmatch_path(xmatch_data.path.clone(), None).unwrap(),
        false,
    );
    let xmatch_partition_info = xmatch_dataset.partition();
    let n_pixels = xmatch_partition_info.len() as u64;

    let Some(datasets) = Phase1Datasets::new(xmatch_data, join_path_model)? else {
        return Ok(());
    };

    // Prepare a progress tracker.
    let progress_tracker = if let Some(progress_file) = &xmatch_data.options.progress_file {
        Some(xmatch::tools::ProgressTracker::new(
            progress_file.clone(),
            n_pixels as i32,
            true,
        ))
    } else {
        None
    };

    let progress_tracker = Mutex::new(progress_tracker);

    // Parallelise the processing of each pixel in the xmatch prtition.
    let right_cat_model = xmatch_data
        .get_catalogue_model(xmatch_data.catalogue_name.as_str())
        .unwrap();

    let phase = right_cat_model.get_phase(1).unwrap();

    let pool = crate::tools::create_pool(Some(
        phase
            .n_workers
            .unwrap_or_else(|| available_parallelism().unwrap().get() / 2),
    ))?;

    pool.install(|| {
        use rayon::iter::ParallelIterator;
        use rayon::prelude::*;

        xmatch_partition_info
            .into_iter()
            .par_bridge()
            .try_for_each(|(xmatch_order, xmatch_pixel)| {
                xmatch_phase1_join_type_table_pixel(
                    Some((xmatch_order, xmatch_pixel)),
                    xmatch_data,
                    join_path_model,
                    &datasets,
                )
                .unwrap();

                // Update the progress tracker.
                if let Some(tracker) = &mut *progress_tracker.lock().unwrap() {
                    tracker.increment().unwrap();
                }

                Ok::<(), anyhow::Error>(())
            })
            .unwrap();
    });

    if let Some(tracker) = &mut *progress_tracker.lock().unwrap() {
        tracker.terminate().unwrap();
    }

    Ok(())
}

/// Returns the column used to rank the candidates of a target and whether higher values are
/// better, or `None` if the join table type does not need ranking.
///
/// For Gaia neighbourhood and best neighbour join tables, each target in the new catalogue may
/// be matched to several catalogids. The best candidate is the one with the highest score for
/// neighbourhood tables and the one with the smallest angular distance for best neighbour tables.
fn join_table_ranking(join_path_model: &JoinPathModel) -> Option<(&'static str, bool)> {
    match join_path_model.join_table_type {
        Some(JoinPathTableType::GaiaNeighbourhood) => Some(("score", true)),
        Some(JoinPathTableType::GaiaBestNeighbour) => Some(("angular_distance", false)),
        _ => None,
    }
}

/// Scans the join table. We assume that the join table may not be in HATS format, so we scan
/// all the parquet files.
fn scan_join_table(join_path_model: &JoinPathModel) -> Result<LazyFrame> {
    let join_table_path = join_path_model.join_table_path.clone().unwrap();
    Ok(LazyFrame::scan_parquet(
        to_pl_path(&join_table_path.join("**/*.parquet"))?,
        Default::default(),
    )?)
}

/// Checks that the join table has the column used to rank the candidates, if needed.
fn check_join_table_ranking(join_path_model: &JoinPathModel) -> Result<()> {
    let Some((ranking_column, _)) = join_table_ranking(join_path_model) else {
        return Ok(());
    };

    if !scan_join_table(join_path_model)?
        .collect_schema()?
        .contains(ranking_column)
    {
        anyhow::bail!(
            "Join table {} does not have a '{}' column.",
            join_path_model.join_table_path.as_ref().unwrap().display(),
            ranking_column
        );
    }

    Ok(())
}

// Process one pixel of the from catalogue.
fn xmatch_phase1_join_type_table_pixel(
    xmatch_order_pixel: Option<(u8, u64)>,
    xmatch_data: &XMatchData,
    join_path_model: &JoinPathModel,
    datasets: &Phase1Datasets,
) -> Result<()> {
    let log_prefix = format!(
        "[{}] - Phase 1 -",
        xmatch_data.catalogue_name.to_uppercase()
    );

    let use_partitioning = xmatch_order_pixel.is_some();
    let (xmatch_order, xmatch_pixel) = xmatch_order_pixel.unwrap_or((0, 0));

    if use_partitioning {
        log::trace!(
            "{} Processing pixel {} (order {}).",
            log_prefix,
            xmatch_pixel,
            xmatch_order
        );
    }

    // Eventually we want to implement joining right to left, but for now we only support
    // joining left to right.
    if join_path_model.join_direction == JoinDirection::RightToLeft {
        anyhow::bail!("Right-to-left joins are not yet supported.");
    }

    let catalog_to_dataset = &datasets.catalog_to_left;
    let ranking = join_table_ranking(join_path_model);

    // The left catalogue may not have any entries in this xmatch pixel.
    if use_partitioning
        && !catalog_to_dataset
            .partition()
            .contains(xmatch_order, xmatch_pixel)?
    {
        log::trace!(
            "{} ({}, {}): no catalog_to_{} entries in pixel, skipping.",
            log_prefix,
            xmatch_order,
            xmatch_pixel,
            join_path_model.left_catalogue
        );
        return Ok(());
    }

    let right_cat_model = xmatch_data
        .get_catalogue_model(xmatch_data.catalogue_name.as_str())
        .unwrap();
    let left_cat_model = xmatch_data
        .get_catalogue_model(&join_path_model.left_catalogue)
        .unwrap();

    let left_on = join_path_model.left_on.clone();
    let join_table_left_on = join_path_model.join_table_left_on.clone().unwrap();
    let join_table_right_on = join_path_model.join_table_right_on.clone().unwrap();

    // If the left join column is not the primary key in the catalog_to_<left> table, we need to
    // get it from the left catalogue before joining with the join table.
    let left_cat_lf = if left_on != left_cat_model.primary_key {
        let left_cat_dataset = datasets
            .left_cat
            .as_ref()
            .expect("left catalogue dataset must exist if the join column is not the pk");

        let left_cat_lf = if use_partitioning {
            let left_cat_pixels =
                left_cat_dataset.get_neighbours(xmatch_order, xmatch_pixel, true)?;
            if left_cat_pixels.is_empty() {
                return Ok(());
            }
            left_cat_dataset.scan_pixels(left_cat_pixels)?
        } else {
            left_cat_dataset.get_frame()?
        };

        Some(left_cat_lf.select([col(&left_cat_model.primary_key), col(&left_on)]))
    } else {
        None
    };

    // Scans the best associations in catalog_to_<left> (as in the column join) for some pixels,
    // or for all of them if no pixels are given, with the left join column.
    let scan_catalog_to_left = |pixels: Option<Vec<(u8, u64)>>| -> Result<LazyFrame> {
        let lf = match pixels {
            Some(pixels) => catalog_to_dataset.scan_pixels(pixels)?,
            None => catalog_to_dataset.get_frame()?,
        }
        .filter(col("best").eq(lit(true)));

        Ok(match &left_cat_lf {
            Some(left_cat_lf) => lf.join(
                left_cat_lf.clone(),
                [col(&left_cat_model.primary_key)],
                [col(&left_cat_model.primary_key)],
                JoinArgs {
                    how: JoinType::Inner,
                    ..Default::default()
                },
            ),
            None => lf,
        })
    };

    let join_table_lf = scan_join_table(join_path_model)?;

    // Read the right catalogue. When using partitioning, we select all the neighbouring pixels to
    // make sure we get all the rows that may match the input pixel.
    let right_cat_dataset = &datasets.right_cat;

    let right_cat_lf = if use_partitioning {
        let right_cat_pixels =
            right_cat_dataset.get_neighbours(xmatch_order, xmatch_pixel, true)?;
        right_cat_dataset.scan_pixels(right_cat_pixels)?
    } else {
        right_cat_dataset.get_frame()?
    };

    // Keep only the primary key and join column of the right catalogue, with reserved names.
    // Otherwise they could collide with the columns from catalog_to_<left> or the join table
    // (e.g., if both primary keys are called source_id).
    let right_cat_lf = right_cat_lf.select([
        col(&right_cat_model.primary_key).alias(RIGHT_PK_COLUMN),
        col(&join_path_model.right_on).alias(RIGHT_ON_COLUMN),
    ]);

    // Let's do all the joins now, for the catalogids in this pixel.
    let catalog_to_lf =
        scan_catalog_to_left(use_partitioning.then(|| vec![(xmatch_order, xmatch_pixel)]))?;

    let joined_lf = catalog_to_lf
        .join(
            join_table_lf.clone(),
            [col(&left_on)],
            [col(&join_table_left_on)],
            JoinArgs {
                how: JoinType::Inner,
                ..Default::default()
            },
        )
        .join(
            right_cat_lf,
            [col(&join_table_right_on)],
            [col(RIGHT_ON_COLUMN)],
            JoinArgs {
                how: JoinType::Inner,
                ..Default::default()
            },
        );

    // Keep only the columns we need. For ranking we also keep the ranking value and the join
    // table right column, which is used to find the candidates in the neighbouring pixels.
    let mut columns = vec![col("catalogid"), col("_healpix_29"), col(RIGHT_PK_COLUMN)];
    if let Some((ranking_column, _)) = ranking {
        columns.push(col(ranking_column));
        columns.push(col(&join_table_right_on).alias(JOIN_TABLE_RIGHT_ON_COLUMN));
    }
    let joined_lf = joined_lf.select(columns);

    // Finally, we remove entries that were already cross-matched using previous join paths.
    let joined_lf = remove_already_xmatched(
        joined_lf,
        RIGHT_PK_COLUMN,
        xmatch_order_pixel,
        xmatch_data,
        datasets,
    )?;

    // Collect the associations once, instead of counting the rows and then writing them, which
    // would run the joins twice.
    let joined_df = joined_lf
        .collect_with_engine(Engine::Streaming)?
        .unwrap_single();

    let n_rows = joined_df.height();

    if n_rows == 0 {
        log::trace!(
            "{} ({}, {}): no rows to write, skipping.",
            log_prefix,
            xmatch_order,
            xmatch_pixel,
        );
        return Ok(());
    }

    log::trace!(
        "{} ({}, {}): {} rows successfully joined.",
        log_prefix,
        xmatch_order,
        xmatch_pixel,
        n_rows
    );

    // Each target in the new catalogue may be matched to multiple catalogids, which can fall in
    // different xmatch pixels. For each target we mark as best the candidate with the best
    // ranking value (ties are broken by the lowest catalogid) but we keep all of them.
    let lf = if let Some((ranking_column, descending)) = ranking {
        // The columns of the candidates, in the same order for this pixel and its neighbours.
        let candidate_columns = |central: bool| {
            vec![
                col("catalogid"),
                col("_healpix_29"),
                col(RIGHT_PK_COLUMN),
                col(ranking_column),
                col(JOIN_TABLE_RIGHT_ON_COLUMN),
                lit(central).alias(CENTRAL_COLUMN),
            ]
        };

        let candidates_lf = joined_df.clone().lazy().select(candidate_columns(true));

        // When using partitioning, add the candidates for catalogids in the neighbouring pixels,
        // but only for the targets matched in this pixel. Those are few, so we restrict the join
        // table to them before joining it with catalog_to_<left>.
        let neighbour_pixels = if use_partitioning {
            catalog_to_dataset.get_neighbours(xmatch_order, xmatch_pixel, false)?
        } else {
            Vec::new()
        };

        let candidates_lf = if neighbour_pixels.is_empty() {
            candidates_lf
        } else {
            // Restrict the join table with a semi-join. Filtering with is_in() and a literal list
            // of values is much slower when there are many values (dense pixels).
            let right_on_values_lf = joined_df
                .select([JOIN_TABLE_RIGHT_ON_COLUMN])?
                .lazy()
                .unique(None, UniqueKeepStrategy::Any);

            let join_table_lf = join_table_lf
                .select([
                    col(&join_table_left_on),
                    col(&join_table_right_on).alias(JOIN_TABLE_RIGHT_ON_COLUMN),
                    col(ranking_column),
                ])
                .join(
                    right_on_values_lf,
                    [col(JOIN_TABLE_RIGHT_ON_COLUMN)],
                    [col(JOIN_TABLE_RIGHT_ON_COLUMN)],
                    JoinArgs {
                        how: JoinType::Semi,
                        ..Default::default()
                    },
                );

            // The right primary keys of the targets matched in this pixel.
            let right_pk_lf = joined_df
                .select([JOIN_TABLE_RIGHT_ON_COLUMN, RIGHT_PK_COLUMN])?
                .lazy()
                .unique(None, UniqueKeepStrategy::Any);

            let neighbours_lf = scan_catalog_to_left(Some(neighbour_pixels))?
                .select([col("catalogid"), col("_healpix_29"), col(&left_on)])
                .join(
                    join_table_lf,
                    [col(&left_on)],
                    [col(&join_table_left_on)],
                    JoinArgs {
                        how: JoinType::Inner,
                        ..Default::default()
                    },
                )
                .join(
                    right_pk_lf,
                    [col(JOIN_TABLE_RIGHT_ON_COLUMN)],
                    [col(JOIN_TABLE_RIGHT_ON_COLUMN)],
                    JoinArgs {
                        how: JoinType::Inner,
                        ..Default::default()
                    },
                );

            let neighbours_lf = remove_already_xmatched(
                neighbours_lf,
                RIGHT_PK_COLUMN,
                xmatch_order_pixel,
                xmatch_data,
                datasets,
            )?;

            concat(
                [
                    candidates_lf,
                    neighbours_lf.select(candidate_columns(false)),
                ],
                UnionArgs::default(),
            )?
        };

        // Flag the candidate with the best ranking value for each target, breaking ties by the
        // lowest catalogid. We sort instead of using window expressions, which are extremely
        // slow for dense pixels.
        let candidates_lf = xmatch::tools::flag_best_per_group(
            candidates_lf,
            RIGHT_PK_COLUMN,
            &[(ranking_column, descending), ("catalogid", false)],
            "best",
        );

        // Keep only the catalogids in this pixel. Rows for catalogids in the neighbouring
        // pixels are written when those pixels are processed.
        candidates_lf.filter(col(CENTRAL_COLUMN))
    } else {
        joined_df.lazy().with_columns([lit(true).alias("best")])
    };

    let added_on = chrono::Utc::now().timestamp_millis();
    let lf = lf.select([
        col("catalogid"),
        col("_healpix_29"),
        col(RIGHT_PK_COLUMN).alias("pk"),
        lit(xmatch_data.model.version.clone())
            .cast(DataType::String)
            .alias("version"),
        lit(Null {}).cast(DataType::Float64).alias("distance"),
        col("best"),
        lit(1_u8).alias("added_by_phase"),
        lit(added_on).alias("added_on"),
    ]);

    xmatch::tools::write_catalog_to_table(
        right_cat_model.name.clone(),
        &xmatch_data.path,
        lf,
        right_cat_model.primary_key.clone(),
        xmatch_data.model.pixelation_order,
        xmatch_data.model.version.clone(),
        right_cat_model.run_id,
        true,
    )?;

    Ok(())
}
