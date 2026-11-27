/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-10-26
 *  @Filename: phase_1_column.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::{Result, bail};
use polars::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::thread::available_parallelism;

use crate::hats::dataset::{AsDataset, Dataset};
use crate::tools::{self, IsEmpty};
use crate::xmatch;
use crate::xmatch::models::{JoinPathModel, XMatchData};

/// Reserved name for the primary key of the catalogue being processed during phase 1 joins.
pub(crate) const RIGHT_PK_COLUMN: &str = "__xmatch_right_pk";

/// Reserved name for the join column of the catalogue being processed during phase 1 joins.
pub(crate) const RIGHT_ON_COLUMN: &str = "__xmatch_right_on";

/// Datasets used to process each pixel of a phase 1 join path.
///
/// They are created once per join path and shared between the pixel workers, so that their
/// partition info is only read once. None of them is modified while the join path is processed.
pub(crate) struct Phase1Datasets {
    /// The HATS dataset of the catalogue being processed ("right").
    pub(crate) right_cat: Dataset,
    /// The catalog_to_<left> dataset.
    pub(crate) catalog_to_left: Dataset,
    /// The HATS dataset of the left catalogue. Only needed (and created) if the left join column
    /// is not the left primary key.
    pub(crate) left_cat: Option<Dataset>,
    /// Snapshot of the associations written by previous join paths, used to remove targets
    /// that were already cross-matched. `None` if there are no previous associations.
    pub(crate) already_xmatched: Option<AlreadyXMatched>,
}

impl Phase1Datasets {
    /// Creates the datasets for a join path.
    ///
    /// Returns `None` if the catalog_to_<left> dataset does not exist, in which case there is
    /// nothing to join.
    pub(crate) fn new(
        xmatch_data: &XMatchData,
        join_path_model: &JoinPathModel,
    ) -> Result<Option<Self>> {
        let right_cat_model = xmatch_data
            .get_catalogue_model(xmatch_data.catalogue_name.as_str())
            .unwrap();
        let left_cat_model = xmatch_data
            .get_catalogue_model(&join_path_model.left_catalogue)
            .unwrap();

        let catalog_to_left_path = xmatch::tools::get_xmatch_path(
            xmatch_data.path.clone(),
            Some(left_cat_model.name.clone()),
        )?;

        if !catalog_to_left_path.exists() {
            log::warn!(
                "[{}] - Phase 1 - catalog_to_{} does not exist. Skipping join path.",
                xmatch_data.catalogue_name.to_uppercase(),
                left_cat_model.name
            );
            return Ok(None);
        }

        let left_cat = if join_path_model.left_on != left_cat_model.primary_key {
            Some(Dataset::new(left_cat_model.hats_path.clone(), true))
        } else {
            None
        };

        Ok(Some(Self {
            right_cat: Dataset::new(right_cat_model.hats_path.clone(), true),
            catalog_to_left: Dataset::new(catalog_to_left_path, false),
            left_cat,
            already_xmatched: AlreadyXMatched::new(xmatch_data)?,
        }))
    }
}

/// Snapshot of the temporary `catalog_to_<right>` dataset taken before a join path is processed.
///
/// The pixel workers of a join path write their associations to the same temporary dataset that
/// is used to remove the targets already cross-matched by previous join paths. Listing that
/// dataset while it is being written is racy (a pixel directory may exist before any parquet
/// file has been written to it), so we record the partition and the list of parquet files once,
/// before the workers start, and only read those files.
pub(crate) struct AlreadyXMatched {
    dataset: Dataset,
    files: HashMap<(u8, u64), Vec<PathBuf>>,
}

impl AlreadyXMatched {
    /// Takes the snapshot. Returns `None` if no previous join path wrote any associations.
    fn new(xmatch_data: &XMatchData) -> Result<Option<Self>> {
        let right_cat_model = xmatch_data
            .get_catalogue_model(xmatch_data.catalogue_name.as_str())
            .unwrap();

        // This must match the path where write_catalog_to_table() writes the temporary files.
        let catalog_to_right_path = xmatch::tools::get_temp_dir_path(
            xmatch_data.path.clone(),
            &right_cat_model.name,
            &xmatch_data.model.version,
            right_cat_model.run_id as usize,
            false,
        )?
        .join(format!("catalog_to_{}", right_cat_model.name))
        .join("dataset");

        if !catalog_to_right_path.exists() {
            return Ok(None);
        }

        // The partition info is cached in the dataset, so later calls to get_neighbours()
        // won't see pixels added by the current join path.
        let dataset = Dataset::new(catalog_to_right_path, false);

        let mut files: HashMap<(u8, u64), Vec<PathBuf>> = HashMap::new();
        for (order, pixel) in dataset.partition() {
            let pattern = dataset.get_pixel_path(order, pixel)?.join("*.parquet");
            let mut pixel_files = glob::glob(pattern.to_str().unwrap())?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if pixel_files.is_empty() {
                continue;
            }
            pixel_files.sort();
            files.insert((order, pixel), pixel_files);
        }

        if files.is_empty() {
            return Ok(None);
        }

        Ok(Some(Self { dataset, files }))
    }

    /// Scans the snapshot files for a pixel and its neighbours, or all the files if no pixel is
    /// given. Returns `None` if there are no files to scan.
    fn scan(&self, xmatch_order_pixel: Option<(u8, u64)>) -> Result<Option<LazyFrame>> {
        let pixels = if let Some((xmatch_order, xmatch_pixel)) = xmatch_order_pixel {
            // Include the neighbours since the same target may have been associated with a
            // catalogid in an adjacent pixel.
            self.dataset
                .get_neighbours(xmatch_order, xmatch_pixel, true)?
        } else {
            self.files.keys().copied().collect()
        };

        let frames = pixels
            .iter()
            .filter_map(|pixel| self.files.get(pixel))
            .flatten()
            .map(|file| {
                Ok(LazyFrame::scan_parquet(
                    tools::to_pl_path(file)?,
                    Default::default(),
                )?)
            })
            .collect::<Result<Vec<_>>>()?;

        if frames.is_empty() {
            return Ok(None);
        }

        Ok(Some(concat(frames, UnionArgs::default())?))
    }
}

// Process one join path on the primary key.
// Here "left" is the catalogue already cross-matched and "right" is the catalogue being processed.
pub(crate) fn xmatch_phase1_join_type_column(
    xmatch_data: &XMatchData,
    join_path_model: &JoinPathModel,
) -> Result<()> {
    let log_prefix = format!(
        "[{}] - Phase 1 -",
        xmatch_data.catalogue_name.to_uppercase()
    );

    // "Left" tablename and column for the join.
    let left_table = join_path_model.left_catalogue.clone();
    let left_join_column = join_path_model.left_on.clone();

    log::info!(
        "{} Processing join path of type 'column': {}.{} -> {}.{}",
        log_prefix,
        left_table,
        left_join_column,
        xmatch_data.catalogue_name,
        join_path_model.right_on
    );

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
        xmatch_phase1_join_type_column_pixel(None, xmatch_data, join_path_model, &datasets)
            .unwrap();
        return Ok(());
    }

    // Determine the number of workers. Use half of the CPUs if n_workers is not specified.
    let right_cat_model = xmatch_data
        .get_catalogue_model(xmatch_data.catalogue_name.as_str())
        .unwrap();
    let phase_model = right_cat_model.get_phase(1).unwrap();
    let n_workers = phase_model
        .n_workers
        .unwrap_or_else(|| available_parallelism().unwrap().get() / 2);

    // We are going to parallelise based on the pixels in the xmatch dataset.
    let xmatch_dataset = Dataset::new(
        xmatch::tools::get_xmatch_path(xmatch_data.path.clone(), None)?,
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

    // Parallelise the processing of each pixel.
    let pool = crate::tools::create_pool(Some(n_workers))?;
    pool.install(|| {
        use rayon::iter::ParallelIterator;
        use rayon::prelude::*;

        xmatch_partition_info
            .into_iter()
            .par_bridge()
            .try_for_each(|(xmatch_order, xmatch_pixel)| {
                xmatch_phase1_join_type_column_pixel(
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

// Process one pixel of the new catalogue.
fn xmatch_phase1_join_type_column_pixel(
    xmatch_order_pixel: Option<(u8, u64)>,
    xmatch_data: &XMatchData,
    join_path_model: &JoinPathModel,
    datasets: &Phase1Datasets,
) -> Result<()> {
    let log_prefix = format!(
        "[{}] - Phase 1 -",
        xmatch_data.catalogue_name.to_uppercase()
    );
    let use_partitioning = join_path_model.use_partitioning;

    let (xmatch_order, xmatch_pixel) = if use_partitioning {
        let (xmatch_order, xmatch_pixel) = xmatch_order_pixel.unwrap();
        log::trace!(
            "{} Processing xmatch pixel {} (order {}).",
            log_prefix,
            xmatch_pixel,
            xmatch_order
        );

        (xmatch_order, xmatch_pixel)
    } else {
        (0, 0)
    };

    let right_cat_model = xmatch_data
        .get_catalogue_model(xmatch_data.catalogue_name.as_str())
        .unwrap();
    let left_cat_model = xmatch_data
        .get_catalogue_model(&join_path_model.left_catalogue)
        .unwrap();

    // Get columns for the left and right catalogues.
    let right_primary_key = right_cat_model.primary_key.clone();
    let right_join_column = join_path_model.right_on.clone();

    let left_primary_key = left_cat_model.primary_key.clone();
    let left_join_column = join_path_model.left_on.clone();

    // Scan the right catalogue data. We get the matching pixels for the xmatch-pixels and
    // their neighbours to ensure we include all matching rows.
    let right_cat_dataset = &datasets.right_cat;

    let right_lf = if use_partitioning {
        let xmatch_right_neighbours =
            right_cat_dataset.get_neighbours(xmatch_order, xmatch_pixel, true)?;

        if xmatch_right_neighbours.is_empty() {
            log::trace!(
                "{} ({}, {}): no neighbouring pixels, skipping.",
                log_prefix,
                xmatch_order,
                xmatch_pixel,
            );
            return Ok(());
        }

        right_cat_dataset.scan_pixels(xmatch_right_neighbours)?
    } else {
        right_cat_dataset.get_frame()?
    };

    // Filter the right_lf dataset. Remove any rows with null values in the join column.
    // We use reserved names for the right primary key and join column. Otherwise they could
    // collide with the columns from the left catalogue (e.g., if both primary keys are called
    // source_id), or the primary key could be coalesced away if it is also the join column.
    let right_lf = right_lf
        .select([
            col(&right_primary_key).alias(RIGHT_PK_COLUMN),
            col(&right_join_column).alias(RIGHT_ON_COLUMN),
        ])
        .filter(col(RIGHT_ON_COLUMN).is_not_null());

    // Exit early if there are no rows with non-null values in the join column.
    if right_lf.is_empty() {
        log::trace!(
            "{} ({}, {}): no rows with non-null join column, skipping.",
            log_prefix,
            xmatch_order,
            xmatch_pixel,
        );
        return Ok(());
    }

    // Get the lazy frame for the catalog_to table for the "left" catalogue.
    let catalog_to_dataset = &datasets.catalog_to_left;

    let catalog_to_lf = if use_partitioning {
        // The left catalogue may not have any entries in this xmatch pixel.
        if !catalog_to_dataset
            .partition()
            .contains(xmatch_order, xmatch_pixel)?
        {
            log::trace!(
                "{} ({}, {}): no catalog_to_{} entries in pixel, skipping.",
                log_prefix,
                xmatch_order,
                xmatch_pixel,
                left_cat_model.name
            );
            return Ok(());
        }

        catalog_to_dataset
            .scan_pixel(xmatch_order, xmatch_pixel, false)?
            .filter(col("best").eq(lit(true)))
    } else {
        catalog_to_dataset
            .get_frame()?
            .filter(col("best").eq(lit(true)))
    };

    // If the join column is the primary key, that will be present in the catalog_to table.
    // Otherwise, we need to join with the actual catalogue.
    let catalog_to_lf = if left_join_column != left_primary_key {
        let left_cat_dataset = datasets
            .left_cat
            .as_ref()
            .expect("left catalogue dataset must exist if the join column is not the pk");

        let mut left_cat_lf = if use_partitioning {
            let left_cat_neighbours =
                left_cat_dataset.get_neighbours(xmatch_order, xmatch_pixel, true)?;
            left_cat_dataset.scan_pixels(left_cat_neighbours)?
        } else {
            left_cat_dataset.get_frame()?
        };

        let left_cat_schema = left_cat_lf.collect_schema()?;
        if !left_cat_schema.contains(&left_join_column) {
            bail!(
                "{} ({}, {}): join column {} not found in catalogue {}.",
                log_prefix,
                xmatch_order,
                xmatch_pixel,
                left_join_column,
                left_cat_model.name
            );
        }

        let left_cat_lf = left_cat_lf.select([col(&left_primary_key), col(&left_join_column)]);

        // Join the catalog_to table with the to catalogue to get the join column.
        catalog_to_lf.join(
            left_cat_lf,
            [col(&left_primary_key)],
            [col(&left_primary_key)],
            JoinArgs {
                how: JoinType::Inner,
                ..Default::default()
            },
        )
    } else {
        catalog_to_lf
    };

    // Keep only the catalogid and join columns.
    let catalog_to_lf =
        catalog_to_lf.select([col("catalogid"), col(&left_join_column), col("_healpix_29")]);

    // Cast the right join column to the data type of the left join column.
    let left_join_dtype = xmatch::models::get_column_dtype(&catalog_to_lf, &left_join_column)?;

    // Perform the join.
    let join_lf = catalog_to_lf.join(
        right_lf,
        [col(&left_join_column)],
        [col(RIGHT_ON_COLUMN).cast(left_join_dtype)],
        JoinArgs {
            how: JoinType::Inner,
            ..Default::default()
        },
    );

    // Create the catalog_to_table frame.
    let added_on = chrono::Utc::now().timestamp_millis();
    let lf = join_lf.select([
        col("catalogid"),
        col("_healpix_29"),
        col(RIGHT_PK_COLUMN).alias("pk"),
        lit(xmatch_data.model.version.clone())
            .cast(DataType::String)
            .alias("version"),
        lit(Null {}).cast(DataType::Float64).alias("distance"),
        lit(true).alias("best"),
        lit(1_u8).alias("added_by_phase"),
        lit(added_on).alias("added_on"),
    ]);

    // We do not check whether the join is empty here. The row count below covers that case
    // and evaluating the join an extra time is expensive for large frames.

    // This lazy frame may include rows with catalogids that are already in catalog_to_<right>
    // from a different join path. We remove those if they exist.
    // TODO: Better than removing them, it would be good to check that they are the same
    // associations and if they are not, do something nuanced.
    let lf = remove_already_xmatched(lf, "pk", xmatch_order_pixel, xmatch_data, datasets)?;

    let lf_nrows = tools::lf_row_count(&lf)?;

    if lf_nrows == 0 {
        log::trace!(
            "{} ({}, {}): no rows after removing duplicates, skipping.",
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
        lf_nrows
    );

    xmatch::tools::write_catalog_to_table(
        right_cat_model.name.clone(),
        &xmatch_data.path,
        lf,
        right_primary_key.clone(),
        xmatch_data.model.pixelation_order,
        xmatch_data.model.version.clone(),
        right_cat_model.run_id,
        true,
    )?;

    Ok(())
}

/// Removes rows that were already cross-matched by a previous join path.
///
/// `lf` is the frame of new associations, which must contain a `catalogid` column and the
/// primary key of the catalogue being processed in a column named `pk_column`. Rows are removed
/// if their catalogid or their primary key are already present in the temporary
/// `catalog_to_<right>` dataset written by the previous join paths (as recorded in
/// `datasets.already_xmatched`). Earlier join paths therefore take priority.
pub(crate) fn remove_already_xmatched(
    lf: LazyFrame,
    pk_column: &str,
    xmatch_order_pixel: Option<(u8, u64)>,
    xmatch_data: &XMatchData,
    datasets: &Phase1Datasets,
) -> Result<LazyFrame> {
    let right_cat_model = xmatch_data
        .get_catalogue_model(xmatch_data.catalogue_name.as_str())
        .unwrap();

    let Some(already_xmatched) = &datasets.already_xmatched else {
        return Ok(lf);
    };

    let Some(catalog_to_right_lf) = already_xmatched.scan(xmatch_order_pixel)? else {
        return Ok(lf);
    };

    let right_pk = right_cat_model.primary_key.clone();

    // Anti-join the new associations with the existing ones to remove any rows with
    // catalogids or targets that are already present.
    let lf = lf
        .join(
            catalog_to_right_lf.clone().select([col("catalogid")]),
            [col("catalogid")],
            [col("catalogid")],
            JoinArgs {
                how: JoinType::Anti,
                ..Default::default()
            },
        )
        .join(
            catalog_to_right_lf.select([col(&right_pk)]),
            [col(pk_column)],
            [col(&right_pk)],
            JoinArgs {
                how: JoinType::Anti,
                ..Default::default()
            },
        );

    Ok(lf)
}
