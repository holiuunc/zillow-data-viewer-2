#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from data_processor import load_json_data, extract_property_features, clean_dataframe

def create_sample_2023(data_dir, sample_size=1000, output_dir="sample_data"):
    """
    Create a sample dataset from 2023 Zillow data
    
    Parameters:
    -----------
    data_dir : str
        Directory containing 2023 dataset files
    sample_size : int
        Number of records to include in sample
    output_dir : str
        Directory to save sample dataset
    
    Returns:
    --------
    str or None
        Path to sample file if created, None otherwise
    """
    print(f"Creating 2023 sample dataset (max {sample_size} records)...")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of JSON files in directory
    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return None
    
    # Check if sample already exists
    sample_path = os.path.join(output_dir, "zillow_data_2023_sample.csv")
    if os.path.exists(sample_path):
        print(f"Sample file already exists at {sample_path}")
        
        # Ask if we should overwrite
        overwrite = input("Overwrite existing sample? (y/n): ").lower().strip() == 'y'
        if not overwrite:
            print("Using existing sample file")
            return sample_path
    
    # Load data from the first file (sufficient for sample)
    print(f"Loading data from {os.path.basename(json_files[0])}...")
    data = load_json_data(json_files[0], max_records=sample_size * 2)  # Load extra for cleaning
    
    if not data:
        print("No data loaded from file")
        return None
    
    print(f"Loaded {len(data)} records")
    
    # Extract property features
    print("Extracting property features...")
    properties = extract_property_features(data, dataset_year=2023)
    
    if not properties:
        print("Failed to extract property features")
        return None
    
    # Convert to DataFrame and clean
    df = pd.DataFrame(properties)
    df = clean_dataframe(df)
    
    # Ensure we have at least some data
    if len(df) == 0:
        print("No valid properties found after cleaning")
        return None
    
    # Create sample by taking random subset
    df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    
    # Save to CSV
    df_sample.to_csv(sample_path, index=False)
    
    print(f"Saved 2023 sample dataset with {len(df_sample)} records to {sample_path}")
    return sample_path

def create_sample_2025(data_dir, sample_size=1000, output_dir="sample_data"):
    """
    Create a sample dataset from 2025 Zillow data
    
    Parameters:
    -----------
    data_dir : str
        Directory containing 2025 dataset files
    sample_size : int
        Number of records to include in sample
    output_dir : str
        Directory to save sample dataset
    
    Returns:
    --------
    str or None
        Path to sample file if created, None otherwise
    """
    print(f"Creating 2025 sample dataset (max {sample_size} records)...")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of JSON files in directory
    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return None
    
    # Check if sample already exists
    sample_path = os.path.join(output_dir, "zillow_data_2025_sample.csv")
    if os.path.exists(sample_path):
        print(f"Sample file already exists at {sample_path}")
        
        # Ask if we should overwrite
        overwrite = input("Overwrite existing sample? (y/n): ").lower().strip() == 'y'
        if not overwrite:
            print("Using existing sample file")
            return sample_path
    
    # Calculate how many records to take from each file
    records_per_file = sample_size // len(json_files) + 1
    
    all_properties = []
    
    # Process each file to get a balanced sample
    for json_file in json_files:
        print(f"Processing {os.path.basename(json_file)}...")
        
        # Load a portion of this file
        data = load_json_data(json_file, max_records=records_per_file * 2)  # Load extra for filtering
        
        if data:
            # Extract property features
            properties = extract_property_features(data, dataset_year=2025)
            
            if properties:
                print(f"Extracted {len(properties)} properties from {os.path.basename(json_file)}")
                all_properties.extend(properties)
            else:
                print(f"No properties extracted from {os.path.basename(json_file)}")
        else:
            print(f"No data loaded from {os.path.basename(json_file)}")
    
    if not all_properties:
        print("No properties extracted from any file")
        return None
    
    # Convert to DataFrame and clean
    df = pd.DataFrame(all_properties)
    df = clean_dataframe(df)
    
    # Ensure we have at least some data
    if len(df) == 0:
        print("No valid properties found after cleaning")
        return None
    
    # Create sample by taking random subset
    df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    
    # Save to CSV
    df_sample.to_csv(sample_path, index=False)
    
    print(f"Saved 2025 sample dataset with {len(df_sample)} records to {sample_path}")
    return sample_path

def create_combined_sample(sample_path_2023, sample_path_2025, output_dir="sample_data"):
    """
    Create a combined sample dataset from 2023 and 2025 samples
    
    Parameters:
    -----------
    sample_path_2023 : str
        Path to 2023 sample CSV file
    sample_path_2025 : str
        Path to 2025 sample CSV file
    output_dir : str
        Directory to save combined sample dataset
    
    Returns:
    --------
    str or None
        Path to combined sample file if created, None otherwise
    """
    print("Creating combined sample dataset...")
    
    # Check if both input files exist
    if not os.path.exists(sample_path_2023) or not os.path.exists(sample_path_2025):
        print("Both 2023 and 2025 sample files must exist")
        return None
    
    # Load sample datasets
    df_2023 = pd.read_csv(sample_path_2023)
    df_2025 = pd.read_csv(sample_path_2025)
    
    # Ensure dataset_year column exists
    if 'dataset_year' not in df_2023.columns:
        df_2023['dataset_year'] = 2023
    if 'dataset_year' not in df_2025.columns:
        df_2025['dataset_year'] = 2025
    
    # Find common columns
    common_cols = list(set(df_2023.columns) & set(df_2025.columns))
    print(f"Found {len(common_cols)} common columns between datasets")
    
    # Create combined dataset
    df_combined = pd.concat([
        df_2023[common_cols],
        df_2025[common_cols]
    ], ignore_index=True)
    
    # Save combined sample
    combined_path = os.path.join(output_dir, "zillow_data_combined_sample.csv")
    df_combined.to_csv(combined_path, index=False)
    
    print(f"Saved combined sample with {len(df_combined)} records to {combined_path}")
    return combined_path

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Create sample datasets from Zillow data')
    parser.add_argument('--data-dir-2023', type=str, default='../Zillow_Data_2023', help='Directory containing 2023 dataset files')
    parser.add_argument('--data-dir-2025', type=str, default='../Zillow_Data_2025', help='Directory containing 2025 dataset files')
    parser.add_argument('--output-dir', type=str, default='sample_data', help='Directory to save sample datasets')
    parser.add_argument('--sample-size', type=int, default=1000, help='Number of records to include in each sample')
    parser.add_argument('--year', type=str, choices=['2023', '2025', 'both'], default='both', help='Which year(s) to process')
    
    args = parser.parse_args()
    
    # Create sample datasets
    sample_path_2023 = None
    sample_path_2025 = None
    
    if args.year in ['2023', 'both']:
        sample_path_2023 = create_sample_2023(args.data_dir_2023, args.sample_size, args.output_dir)
    
    if args.year in ['2025', 'both']:
        sample_path_2025 = create_sample_2025(args.data_dir_2025, args.sample_size, args.output_dir)
    
    # Create combined sample if both individual samples were created
    if sample_path_2023 and sample_path_2025:
        create_combined_sample(sample_path_2023, sample_path_2025, args.output_dir)
    
    # Print instructions for using samples
    print("\nTo use these sample datasets in the Zillow Data Viewer app:")
    print(f"1. Copy the sample files from {args.output_dir}/ to the app's sample_data/ directory")
    print("2. Select 'Use Sample Data' option in the app sidebar")
    print("3. Click 'Load Sample Data' button")

if __name__ == "__main__":
    main() 