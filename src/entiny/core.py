"""
Core implementation of the IBOSS algorithm using Polars for efficient data processing.
"""

from typing import Union, Optional
import polars as pl
import numpy as np
from tqdm import tqdm


def entiny(
    data: Union[pl.LazyFrame, pl.DataFrame, str],
    n: int,
    seed: Optional[int] = None,
    scan_kwargs: Optional[dict] = None,
    show_progress: bool = True
) -> pl.LazyFrame:
    """
    Memory-efficient implementation of Information-Based Optimal Subdata Selection (iBoss)
    designed to work with datasets larger than memory, with support for stratification.
    
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
    print("Collecting schema information...")
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
    print("Adding row indices...")
    lf_with_idx = lf.with_row_index("__tinying_index__")
    
    # Create an empty list to collect all filtered LazyFrames
    selected_lfs = []
    
    # Process each variable to find extreme values
    # If we have strata columns, perform stratified sampling
    if strata and len(strata) > 0:
        print(f"Found {len(strata)} strata columns: {', '.join(strata)}")
        print(f"Will perform stratified sampling for {len(variables)} numeric variables")
        
        # Create a progress bar for variables
        var_iter = tqdm(variables, desc="Processing variables", disable=not show_progress)
        
        for var in var_iter:
            if show_progress:
                var_iter.set_description(f"Processing variable: {var}")
            
            # Top n values within each stratum
            top_indices_lf = (
                lf_with_idx
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
            
            # Bottom n values within each stratum
            bottom_indices_lf = (
                lf_with_idx
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
            
            # Add to our collection
            selected_lfs.append(top_indices_lf)
            selected_lfs.append(bottom_indices_lf)
    else:
        print(f"No strata columns found. Will perform regular sampling for {len(variables)} numeric variables")
        
        # Create a progress bar for variables
        var_iter = tqdm(variables, desc="Processing variables", disable=not show_progress)
        
        for var in var_iter:
            if show_progress:
                var_iter.set_description(f"Processing variable: {var}")
            
            # Top n values
            top_indices_lf = (
                lf_with_idx
                .select([var, "__tinying_index__"])
                .sort(by=var, descending=True)
                .limit(n)
                .select("__tinying_index__")
            )
            
            # Bottom n values
            bottom_indices_lf = (
                lf_with_idx
                .select([var, "__tinying_index__"])
                .sort(by=var)
                .limit(n)
                .select("__tinying_index__")
            )
            
            # Add to our collection
            selected_lfs.append(top_indices_lf)
            selected_lfs.append(bottom_indices_lf)
    
    # Union all the index LazyFrames and get unique indices
    print("Combining selected indices...")
    if len(selected_lfs) > 1:
        all_indices_lf = pl.concat(selected_lfs).unique()
    else:
        all_indices_lf = selected_lfs[0].unique()
    
    # Collect only the indices (this is a small amount of data)
    # We need to collect here to get the actual indices for final filtering
    print("Collecting unique indices...")
    unique_indices = all_indices_lf.collect().to_series().to_list()
    print(f"Selected {len(unique_indices)} unique rows")
    
    # Filter the original LazyFrame to include only the selected rows
    print("Creating final filtered dataset...")
    result_lf = lf_with_idx.filter(
        pl.col("__tinying_index__").is_in(unique_indices)
    ).drop("__tinying_index__")
    
    return result_lf
