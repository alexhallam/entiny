#!/bin/bash

# Create a sample dataset
python3 -c '
import polars as pl
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 10000

df = pl.DataFrame({
    "age": np.random.normal(35, 15, n_samples),
    "income": np.random.lognormal(10, 1, n_samples),
    "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], n_samples),
    "region": np.random.choice(["North", "South", "East", "West"], n_samples)
})

# Save as CSV
df.write_csv("sample_data.csv")
'

echo "Created sample dataset: sample_data.csv"

# Basic usage
echo -e "\nBasic usage:"
tinying --input sample_data.csv --output basic_subsample.parquet --height-by-width-factor 10

# With categorical handling strategy
echo -e "\nWith stratified categorical handling:"
tinying -i sample_data.csv -o stratified_subsample.parquet -n 20 -c stratified

# With random categorical handling
echo -e "\nWith random categorical handling:"
tinying -i sample_data.csv -o random_subsample.parquet -n 20 -c random

# With dummy encoding
echo -e "\nWith dummy encoding:"
tinying -i sample_data.csv -o dummy_subsample.parquet -n 20 -c dummy

echo -e "\nAll subsamples have been created successfully!" 