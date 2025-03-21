"""
Core implementation of the IBOSS algorithm using Polars for efficient data processing.
"""


import numpy as np
import polars as pl
from tqdm import tqdm


def entiny(
    data: pl.LazyFrame | pl.DataFrame | str,
    n: int,
    seed: int | None = None,
    scan_kwargs: dict | None = None,
    show_progress: bool = True
) -> pl.LazyFrame:
    """
    Memory-efficient implementation of Information-Based Optimal Subdata Selection (iBoss)
    designed to work with datasets larger than memory, with support for stratification.
    
    This version avoids selecting the same rows multiple times when processing different columns.
    
    Parameters:
    -----------
    data : Union[pl.LazyFrame, pl.DataFrame, str]
        Input data as a LazyFrame, DataFrame, or path to a file.
        For large datasets, a LazyFrame or file path is recommended.
    n : int
        Number of extreme values to select from each end (top and bottom)
        of each variable within each stratum.
    seed : int, optional
        Random seed for reproducibility.
    scan_kwargs : dict, optional
        Additional arguments to pass to scan_csv/scan_parquet if data is a file path.
    show_progress : bool, optional
        Whether to show progress bars. Default is True.
        
    Returns:
    --------
    pl.LazyFrame
        A lazy subset of the original data containing the selected samples.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Convert input to LazyFrame if it's not already
    if isinstance(data, str):
        # Determine file type from extension
        if data.endswith('.csv'):
            scan_kwargs = scan_kwargs or {}
            print("Loading CSV file...")
            lf = pl.scan_csv(data, **scan_kwargs)
        elif data.endswith('.parquet'):
            scan_kwargs = scan_kwargs or {}
            print("Loading Parquet file...")
            lf = pl.scan_parquet(data, **scan_kwargs)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet for large datasets.")
    elif isinstance(data, pl.DataFrame):
        lf = data.lazy()
    elif isinstance(data, pl.LazyFrame):
        lf = data

    # Get schema to identify numeric and string columns

    schema = lf.collect_schema()
    
    # If variables not specified, use all numeric columns
    variables = [
        name for name, dtype in schema.items() 
        if dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    ]
    if not variables:
        raise ValueError("No numeric columns found in the dataset.")
    
    # If strata not specified, check for string columns
    strata = [
        name for name, dtype in schema.items() 
        if dtype in [pl.Utf8, pl.Categorical]
    ]
    
    # Add row index
    lf_with_idx = lf.with_row_index("__tinying_index__")
    
    # Keep track of all selected indices to avoid duplicates
    selected_indices: set[int] = set()
    
    # Create an empty list to collect all filtered LazyFrames
    selected_lfs = []
    
    # Create a progress bar for variables
    var_iter = tqdm(variables, desc="Processing variables", disable=not show_progress)
    
    # Process each variable to find extreme values
    for var in var_iter:
        if show_progress:
            var_iter.set_description(f"Processing variable: {var}")
        
        # Create a filtered LazyFrame that excludes already selected indices
        if selected_indices:
            # Convert selected_indices to a list for filtering
            filtered_lf = lf_with_idx.filter(
                ~pl.col("__tinying_index__").is_in(list(selected_indices))
            )
        else:
            filtered_lf = lf_with_idx
        
        # If we have strata columns, perform stratified sampling
        if strata and len(strata) > 0:
            # Top n values within each stratum
            top_indices_lf = (
                filtered_lf
                .select([*strata, var, "__tinying_index__"])
                .group_by(strata)
                .agg(
                    pl.col("__tinying_index__")
                    .sort_by(pl.col(var), descending=True)
                    .limit(n)
                    .alias("__top_indices__")
                )
                .explode("__top_indices__")
                .select(pl.col("__top_indices__").alias("__tinying_index__"))
            )
            
            # Collect top indices and add to our set of selected indices
            top_indices = top_indices_lf.collect().to_series().to_list()
            selected_indices.update(top_indices)
            
            # Bottom n values within each stratum
            # Use the filtered LazyFrame again to exclude already selected indices
            if top_indices:
                filtered_lf = lf_with_idx.filter(
                    ~pl.col("__tinying_index__").is_in(list(selected_indices))
                )
            
            bottom_indices_lf = (
                filtered_lf
                .select([*strata, var, "__tinying_index__"])
                .group_by(strata)
                .agg(
                    pl.col("__tinying_index__")
                    .sort_by(pl.col(var), descending=False)
                    .limit(n)
                    .alias("__bottom_indices__")
                )
                .explode("__bottom_indices__")
                .select(pl.col("__bottom_indices__").alias("__tinying_index__"))
            )
            
            # Collect bottom indices and add to our set of selected indices
            bottom_indices = bottom_indices_lf.collect().to_series().to_list()
            selected_indices.update(bottom_indices)
            
            # Add to our collection of LazyFrames
            selected_lfs.append(top_indices_lf)
            selected_lfs.append(bottom_indices_lf)
        else:
            # No stratification - simple sampling
            
            # Top n values
            top_indices_lf = (
                filtered_lf
                .select([var, "__tinying_index__"])
                .sort(by=var, descending=True)
                .limit(n)
                .select("__tinying_index__")
            )
            
            # Collect top indices and add to our set of selected indices
            top_indices = top_indices_lf.collect().to_series().to_list()
            selected_indices.update(top_indices)
            
            # Bottom n values
            # Use the filtered LazyFrame again to exclude already selected indices
            if top_indices:
                filtered_lf = lf_with_idx.filter(
                    ~pl.col("__tinying_index__").is_in(list(selected_indices))
                )
            
            bottom_indices_lf = (
                filtered_lf
                .select([var, "__tinying_index__"])
                .sort(by=var)
                .limit(n)
                .select("__tinying_index__")
            )
            
            # Collect bottom indices and add to our set of selected indices
            bottom_indices = bottom_indices_lf.collect().to_series().to_list()
            selected_indices.update(bottom_indices)
            
            # Add to our collection
            selected_lfs.append(top_indices_lf)
            selected_lfs.append(bottom_indices_lf)
    
    # Convert the set of selected indices to a list
    print(f"Selected {len(selected_indices)} unique rows")
    
    # Filter the original LazyFrame to include only the selected rows

    result_lf = lf_with_idx.filter(
        pl.col("__tinying_index__").is_in(list(selected_indices))
    ).drop("__tinying_index__")
    
    return result_lf