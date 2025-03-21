"""
Command-line interface for the Tinying package.
"""

import click
from .core import entiny

HELP_TEXT = """
Extreme value sampling tool using Information-Based Optimal Subdata Selection (IBOSS).

This tool performs intelligent subsampling by selecting extreme values from numeric 
variables while preserving the structure of categorical variables through automatic 
stratification.

Features:
  - Automatic detection of numeric and categorical columns
  - Stratified sampling within categorical groups
  - Support for both CSV and Parquet files
  - Memory-efficient processing for large datasets
  - Progress tracking with customizable display

Examples:
  # Basic usage with CSV files:
  entiny -i data.csv -o sampled.csv -n 10

  # Using Parquet files with seed for reproducibility:
  entiny -i data.parquet -o sampled.parquet -n 20 --seed 42

  # Disable progress bars for scripting:
  entiny -i data.csv -o sampled.csv -n 10 --no-progress

Input Data Requirements:
  - Must be CSV or Parquet format
  - Must contain at least one numeric column
  - String/categorical columns are automatically used for stratification
"""

@click.command(help=HELP_TEXT)
@click.option(
    '--input',
    '-i',
    required=True,
    help='Input file path (CSV or Parquet format)'
)
@click.option(
    '--output',
    '-o',
    required=True,
    help='Output file path (CSV or Parquet format)'
)
@click.option(
    '--n',
    '-n',
    default=10,
    type=int,
    help='Number of extreme values to select from each end of each numeric variable (default: 10)'
)
@click.option(
    '--seed',
    '-s',
    type=int,
    help='Random seed for reproducibility'
)
@click.option(
    '--no-progress',
    is_flag=True,
    help='Disable progress bars (useful for scripting)'
)
def cli(
    input: str,
    output: str,
    n: int,
    seed: int | None,
    no_progress: bool
) -> None:
    """
    Perform IBOSS subsampling on a dataset using command-line interface.
    
    This tool will:
    1. Automatically detect numeric columns for sampling extreme values
    2. Automatically detect categorical columns for stratification
    3. Sample n extreme values from each end of each numeric variable
    4. If categorical columns are present, perform sampling within each stratum
    """
    try:
        # Validate n
        if n <= 0:
            raise ValueError("n must be a positive integer")

        # Determine output format from file extension
        output_format = output.split('.')[-1].lower()
        if output_format not in ['csv', 'parquet']:
            raise ValueError("Output file must be CSV or Parquet format")

        # Perform subsampling
        result = entiny(
            data=input,
            n=n,
            seed=seed,
            show_progress=not no_progress
        )

        # Write to output file
        if output_format == 'csv':
            result.collect().write_csv(output)
        else:
            result.collect().write_parquet(output)

        click.echo(f"Successfully subsampled data and saved to {output}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    cli() 