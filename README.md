# Entiny

[![Test](https://github.com/alexhallam/entiny/actions/workflows/test.yml/badge.svg)](https://github.com/alexhallam/entiny/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/entiny.svg)](https://badge.fury.io/py/entiny)

A high-performance Python package for Information-Based Optimal Subdata Selection (IBOSS) using Polars for efficient data processing.

## Features

- Larger than memory implementation suitable for large datasets
- Automatic detection and handling of stratification variables
- Progress tracking with tqdm
- Support for both CSV and Parquet file formats
- Command-line interface for easy usage

## Installation

```bash
# Install the package with pip
pip install entiny

# The CLI command 'entiny' will be automatically installed
# Verify the installation
entiny --help
```

The installation will automatically add the `entiny` command to your system. You can verify the installation by running `entiny --help` to see the available options.

## Quick Start

```python
import polars as pl
import numpy as np
from entiny import entiny

# Create or load your data
df = pl.DataFrame({
    "category": ["A", "A", "B", "B"] * 250,
    "value1": np.random.normal(0, 1, 1000),
    "value2": np.random.uniform(-5, 5, 1000)
})

# Sample extreme values
# This will automatically detect "category" as a stratum
# and sample extreme values within each category
result = entiny(df, n=10).collect()
```

## Usage

### Python API

```python
from entiny import entiny

# From a DataFrame
result = entiny(df, n=10).collect()

# From a CSV file
result = entiny("data.csv", n=10).collect()

# From a Parquet file
result = entiny("data.parquet", n=10).collect()

# With custom options
result = entiny(
    data=df,
    n=10,                    # Number of extreme values to select from each end
    seed=42,                 # For reproducibility
    show_progress=True       # Show progress bars
).collect()
```

### Command Line Interface

```bash
# Basic usage
entiny -i input.csv -o output.csv -n 10

# With all options
entiny \
    --input data.csv \
    --output sampled.csv \
    --n 10 \
    --seed 42 \
    --no-progress  # Optional: disable progress bars
```

## How It Works

1. **Automatic Feature Detection**:
   - Numeric columns are used for sampling extreme values
   - String/categorical columns are automatically detected as strata

2. **Stratified Sampling**:
   - If categorical columns are present, sampling is performed within each stratum
   - For each numeric variable in each stratum:
     - Selects n highest values
     - Selects n lowest values

3. **Memory Efficiency**:
   - Uses Polars' lazy evaluation
   - Processes data in chunks
   - Minimizes memory usage for large datasets

## Example with Stratification

```python
import polars as pl
import numpy as np
from entiny import entiny

# Create a dataset with multiple strata
df = pl.DataFrame({
    "region": ["North", "South"] * 500,
    "category": ["A", "B", "A", "B"] * 250,
    "sales": np.random.lognormal(0, 1, 1000),
    "quantity": np.random.poisson(5, 1000)
})

# Sample extreme values
# Will automatically detect "region" and "category" as strata
result = entiny(df, n=5).collect()
```

## Performance Considerations

- Uses Polars for high-performance data operations
- Lazy evaluation minimizes memory usage
- Progress bars show operation status
- Efficient handling of large datasets through streaming

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
