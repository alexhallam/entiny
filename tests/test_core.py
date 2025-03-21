"""
Test cases for the core functionality.
"""

import numpy as np
import polars as pl

from entiny import entiny


def test_tinying_sampling_numeric_df() -> None:
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a test DataFrame with numeric columns
    df = pl.DataFrame({
        "X1": np.random.normal(0, 1, 1000),
        "X2": np.random.uniform(-5, 5, 1000),
        "X3": np.random.exponential(2, 1000),
        "y": np.random.normal(0, 1, 1000)
    })
    
    # Test with n=10 (selecting 10 extreme values from each end)
    n = 10
    result = entiny(df, n=n).collect()
    
    # Basic checks
    assert isinstance(result, pl.DataFrame)
    assert len(result) <= n * 2 * 4  # Maximum possible rows (n*2 for each variable)
    assert set(result.columns) == {"X1", "X2", "X3", "y"}
    
    # Check that we got extreme values for each column
    for col in ["X1", "X2", "X3", "y"]:
        col_values = result[col].to_numpy()
        original_values = df[col].to_numpy()
        
        # Check that we got some of the highest values
        assert any(
            val >= np.percentile(original_values, 90) for val in col_values
        ), f"Missing high values for {col}"
        # Check that we got some of the lowest values
        assert any(
            val <= np.percentile(original_values, 10) for val in col_values
        ), f"Missing low values for {col}"

def test_tinying_sampling_with_auto_strata() -> None:
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create categories
    categories = ["A", "B", "C"]
    n_per_category = 500
    
    # Create a test DataFrame with numeric columns and a categorical column
    df = pl.DataFrame({
        "category": np.repeat(categories, n_per_category),  # This will be automatically detected as stratum
        "X1": np.concatenate([
            np.random.normal(0, 1, n_per_category),  # Category A
            np.random.normal(2, 1, n_per_category),  # Category B
            np.random.normal(4, 1, n_per_category)   # Category C
        ]),
        "X2": np.concatenate([
            np.random.uniform(-5, -2, n_per_category),  # Category A
            np.random.uniform(-2, 2, n_per_category),   # Category B
            np.random.uniform(2, 5, n_per_category)     # Category C
        ])
    })
    
    # Test with n=5 (selecting 5 extreme values from each end for each stratum)
    n = 5
    result = entiny(df, n=n).collect()
    
    # Basic checks
    assert isinstance(result, pl.DataFrame)
    assert len(result) <= n * 2 * 2 * len(categories)  # Maximum possible rows (n*2 for each variable * number of categories)
    assert set(result.columns) == {"category", "X1", "X2"}
    
    # Check that we got extreme values for each column within each category
    for category in categories:
        category_mask = result["category"] == category
        original_category_mask = df["category"] == category
        
        for col in ["X1", "X2"]:
            col_values = result.filter(category_mask)[col].to_numpy()
            original_values = df.filter(original_category_mask)[col].to_numpy()
            
            # Check that we got some of the highest values within this category
            assert any(val >= np.percentile(original_values, 90) for val in col_values), \
                f"Missing high values for {col} in category {category}"
            # Check that we got some of the lowest values within this category
            assert any(val <= np.percentile(original_values, 10) for val in col_values), \
                f"Missing low values for {col} in category {category}"
            
            # Verify that values come from the correct category
            assert all(df.filter(pl.col("category") == category)[col].to_numpy().min() <= val <= 
                      df.filter(pl.col("category") == category)[col].to_numpy().max() 
                      for val in col_values), \
                f"Values for {col} in category {category} are outside the category's range"

def test_tinying_sampling_multiple_strata() -> None:
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create categories
    groups = ["X", "Y"]
    categories = ["A", "B"]
    n_per_combo = 250  # 250 * (2 groups * 2 categories) = 1000 total rows
    
    # Create all combinations of groups and categories
    all_groups = []
    all_categories = []
    for group in groups:
        for category in categories:
            all_groups.extend([group] * n_per_combo)
            all_categories.extend([category] * n_per_combo)
    
    # Create a test DataFrame with numeric columns and multiple categorical columns
    df = pl.DataFrame({
        "group": all_groups,        # First stratum
        "category": all_categories, # Second stratum
        "X1": np.concatenate([
            np.random.normal(0, 1, n_per_combo),    # X-A
            np.random.normal(2, 1, n_per_combo),    # X-B
            np.random.normal(4, 1, n_per_combo),    # Y-A
            np.random.normal(6, 1, n_per_combo)     # Y-B
        ]),
        "X2": np.concatenate([
            np.random.uniform(-5, -2, n_per_combo), # X-A
            np.random.uniform(-2, 2, n_per_combo),  # X-B
            np.random.uniform(2, 5, n_per_combo),   # Y-A
            np.random.uniform(5, 8, n_per_combo)    # Y-B
        ])
    })
    
    # Test with n=3 (selecting 3 extreme values from each end for each stratum combination)
    n = 3
    result = entiny(df, n=n).collect()
    
    # Basic checks
    assert isinstance(result, pl.DataFrame)
    assert len(result) <= n * 2 * 2 * len(groups) * len(categories)  # Maximum possible rows
    assert set(result.columns) == {"group", "category", "X1", "X2"}
    
    # Check that we got extreme values for each column within each stratum combination
    for group in groups:
        for category in categories:
            # Create mask for this combination of strata
            combo_mask = (result["group"] == group) & (result["category"] == category)
            original_combo_mask = (df["group"] == group) & (df["category"] == category)
            
            for col in ["X1", "X2"]:
                col_values = result.filter(combo_mask)[col].to_numpy()
                original_values = df.filter(original_combo_mask)[col].to_numpy()
                
                if len(col_values) > 0:  # Only test if we have values for this combination
                    # Check that we got some of the highest values within this stratum combination
                    assert any(val >= np.percentile(original_values, 90) for val in col_values), \
                        f"Missing high values for {col} in group {group}, category {category}"
                    # Check that we got some of the lowest values within this stratum combination
                    assert any(val <= np.percentile(original_values, 10) for val in col_values), \
                        f"Missing low values for {col} in group {group}, category {category}"
                    
                    # Verify that values come from the correct stratum combination
                    assert all(df.filter(original_combo_mask)[col].to_numpy().min() <= val <= 
                             df.filter(original_combo_mask)[col].to_numpy().max() 
                             for val in col_values), (
                        f"Values for {col} in group {group}, category {category} "
                        "are outside the stratum's range"
                    )
