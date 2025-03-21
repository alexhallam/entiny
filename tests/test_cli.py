"""
Test cases for the CLI functionality.
"""

import pytest
from click.testing import CliRunner
import polars as pl
import numpy as np
import os
from entiny.cli import cli
import tempfile

@pytest.fixture
def sample_csv():
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        # Create sample data
        df = pl.DataFrame({
            "category": ["A", "A", "B", "B"] * 250,
            "value1": np.random.normal(0, 1, 1000),
            "value2": np.random.uniform(-5, 5, 1000)
        })
        df.write_csv(f.name)
        yield f.name
        os.unlink(f.name)

@pytest.fixture
def sample_parquet():
    """Create a temporary Parquet file with sample data."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        # Create sample data
        df = pl.DataFrame({
            "category": ["A", "A", "B", "B"] * 250,
            "value1": np.random.normal(0, 1, 1000),
            "value2": np.random.uniform(-5, 5, 1000)
        })
        df.write_parquet(f.name)
        yield f.name
        os.unlink(f.name)

def test_cli_help():
    """Test the help command output."""
    runner = CliRunner()
    
    # Test --help
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    
    # Check that key sections are present
    assert "Usage:" in result.output
    assert "Options:" in result.output
    
    # Check that all options are documented
    assert "--input" in result.output
    assert "--output" in result.output
    assert "--n" in result.output
    assert "--seed" in result.output
    assert "--no-progress" in result.output
    
    # Check that help contains important information
    assert "Features:" in result.output
    assert "Examples:" in result.output
    assert "Input Data Requirements:" in result.output
    
    # Check that examples are present
    help_text = result.output
    # Basic CSV example
    assert "data.csv" in help_text
    assert "sampled.csv" in help_text
    assert "-n 10" in help_text
    
    # Parquet example with seed
    assert "data.parquet" in help_text
    assert "sampled.parquet" in help_text
    assert "-n 20" in help_text
    assert "--seed 42" in help_text
    
    # No progress example
    assert "--no-progress" in help_text
    
    # Check for command name
    assert "entiny" in help_text

def test_cli_basic_csv(sample_csv):
    """Test basic CLI functionality with CSV files."""
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix='.csv') as output:
        result = runner.invoke(cli, [
            '--input', sample_csv,
            '--output', output.name,
            '--n', '10'
        ])
        assert result.exit_code == 0
        assert "Successfully subsampled data" in result.output
        
        # Verify output file exists and has correct format
        df = pl.read_csv(output.name)
        assert isinstance(df, pl.DataFrame)
        assert "category" in df.columns
        assert "value1" in df.columns
        assert "value2" in df.columns

def test_cli_basic_parquet(sample_parquet):
    """Test basic CLI functionality with Parquet files."""
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix='.parquet') as output:
        result = runner.invoke(cli, [
            '--input', sample_parquet,
            '--output', output.name,
            '--n', '10'
        ])
        assert result.exit_code == 0
        assert "Successfully subsampled data" in result.output
        
        # Verify output file exists and has correct format
        df = pl.read_parquet(output.name)
        assert isinstance(df, pl.DataFrame)
        assert "category" in df.columns
        assert "value1" in df.columns
        assert "value2" in df.columns

def test_cli_with_seed(sample_csv):
    """Test CLI with seed parameter for reproducibility."""
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix='.csv') as output1, \
         tempfile.NamedTemporaryFile(suffix='.csv') as output2:
        # Run twice with same seed
        result1 = runner.invoke(cli, [
            '--input', sample_csv,
            '--output', output1.name,
            '--n', '10',
            '--seed', '42'
        ])
        result2 = runner.invoke(cli, [
            '--input', sample_csv,
            '--output', output2.name,
            '--n', '10',
            '--seed', '42'
        ])
        
        assert result1.exit_code == 0
        assert result2.exit_code == 0
        
        # Results should be identical
        df1 = pl.read_csv(output1.name)
        df2 = pl.read_csv(output2.name)
        assert df1.equals(df2)

def test_cli_no_progress(sample_csv):
    """Test CLI with progress bars disabled."""
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix='.csv') as output:
        result = runner.invoke(cli, [
            '--input', sample_csv,
            '--output', output.name,
            '--n', '10',
            '--no-progress'
        ])
        assert result.exit_code == 0
        assert "Successfully subsampled data" in result.output

def test_cli_invalid_input():
    """Test CLI with invalid input file."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        '--input', 'nonexistent.csv',
        '--output', 'output.csv',
        '--n', '10'
    ])
    assert result.exit_code != 0
    assert "Error" in result.output

def test_cli_invalid_output_format(sample_csv):
    """Test CLI with invalid output format."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        '--input', sample_csv,
        '--output', 'output.txt',  # Invalid format
        '--n', '10'
    ])
    assert result.exit_code != 0
    assert "Error" in result.output
    assert "must be CSV or Parquet format" in result.output

def test_cli_invalid_n(sample_csv):
    """Test CLI with invalid n parameter."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        '--input', sample_csv,
        '--output', 'output.csv',
        '--n', '-1'  # Invalid n value
    ])
    assert result.exit_code != 0
    assert "Error" in result.output
    assert "n must be a positive integer" in result.output 