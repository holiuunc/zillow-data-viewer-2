#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import io
import streamlit as st

@st.cache_data
def load_json_data(file_path, max_records=None):
    """
    Load JSON data from file, handling large files and different formats
    
    Parameters:
    -----------
    file_path : str
        Path to the JSON file
    max_records : int, optional
        Maximum number of records to load
        
    Returns:
    --------
    list
        List of dictionaries containing the data
    """
    print(f"Loading data from {file_path}...")
    
    try:
        # Special handling for 2023 dataset which has a different structure
        if "2023" in file_path and os.path.basename(file_path).startswith("Zillow-ChapelHill"):
            print(f"Detected 2023 dataset format, using specialized loading for {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # For 2023 format, convert the dictionary structure to a list of objects
                if isinstance(data, dict) and 'zpid' in data:
                    print("Converting 2023 dictionary format to list format...")
                    # Get all property IDs
                    zpids = list(data['zpid'].keys())
                    
                    # Limit the number of properties if max_records is specified
                    if max_records is not None:
                        zpids = zpids[:max_records]
                    
                    # Convert to list format
                    result = []
                    for zpid in tqdm(zpids, desc="Converting properties"):
                        property_data = {}
                        # Collect all features for this property
                        for feature in data.keys():
                            if zpid in data[feature]:
                                property_data[feature] = data[feature][zpid]
                        result.append(property_data)
                    
                    print(f"Converted {len(result)} properties from 2023 format")
                    return result
                else:
                    # Handle other potential formats in 2023 file
                    if max_records is not None:
                        data = data[:max_records]
                    return data
        
        # Standard handling for 2025 and other formats - read in chunks to reduce memory usage
        data = []
        chunk_size = 100  # Process 100 records at a time
        counter = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Check if the file starts with an array
            first_char = f.read(1).strip()
            f.seek(0)
            
            if first_char == '[':
                # JSON array format - read line by line
                bracket_level = 0
                current_object = ""
                for line in f:
                    current_object += line
                    bracket_level += line.count('[') - line.count(']')
                    bracket_level += line.count('{') - line.count('}')
                    
                    if bracket_level == 0 and current_object.strip():
                        try:
                            obj = json.loads(current_object)
                            if isinstance(obj, list):
                                # If we got a list, add all items
                                data.extend(obj[:max_records-len(data)] if max_records else obj)
                            else:
                                # Single object
                                data.append(obj)
                            current_object = ""
                            
                            counter += 1
                            if max_records and len(data) >= max_records:
                                break
                        except json.JSONDecodeError:
                            # Continue collecting more lines
                            pass
            else:
                # JSONL or line-by-line format
                for line in f:
                    line = line.strip()
                    if not line or line in ['[', ']']:
                        continue
                    
                    # Remove trailing comma if exists
                    if line.endswith(','):
                        line = line[:-1]
                    
                    try:
                        obj = json.loads(line)
                        data.append(obj)
                        
                        counter += 1
                        if max_records and counter >= max_records:
                            break
                    except json.JSONDecodeError:
                        continue
            
        print(f"Loaded {len(data)} records from {os.path.basename(file_path)}")
        return data
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

@st.cache_data
def load_multiple_json_files(file_paths, max_records=None):
    """
    Load data from multiple JSON files
    
    Parameters:
    -----------
    file_paths : list
        List of file paths to load
    max_records : int, optional
        Maximum number of records to load in total
        
    Returns:
    --------
    list
        Combined list of dictionaries containing the data
    """
    all_data = []
    remaining = max_records
    
    for file_path in file_paths:
        if os.path.isfile(file_path):
            if max_records is not None:
                part_data = load_json_data(file_path, remaining)
            else:
                part_data = load_json_data(file_path)
                
            all_data.extend(part_data)
            
            if max_records is not None:
                remaining -= len(part_data)
                if remaining <= 0:
                    break
        else:
            print(f"File {file_path} does not exist, skipping...")
    
    return all_data

@st.cache_data
def extract_property_features(data_list, dataset_year=None):
    """
    Extract key attributes from the property data
    
    Parameters:
    -----------
    data_list : list
        List of dictionaries containing property data
    dataset_year : int, optional
        Year of the dataset (2023 or 2025)
        
    Returns:
    --------
    list
        List of dictionaries containing extracted property features
    """
    properties = []
    
    for item in tqdm(data_list, desc="Processing properties"):
        # Skip items that aren't dictionaries
        if not isinstance(item, dict):
            continue
        
        property_data = {}
        
        # Add dataset year if provided
        if dataset_year:
            property_data['dataset_year'] = dataset_year
        
        # Direct access for simple properties
        for field in ['zpid', 'id', 'price', 'bedrooms', 'bathrooms', 'livingArea', 'lotSize', 
                     'yearBuilt', 'homeType', 'streetAddress', 'city', 'state', 'zipcode',
                     'latitude', 'longitude', 'description']:
            if field in item:
                property_data[field] = item.get(field)
        
        # Handle 2023 dataset structure with nested props
        if 'props' in item:
            props = item.get('props', {})
            property_info = props.get('pageProps', {}).get('initialData', {}).get('building', {})
            
            # Basic property info
            for field in ['zpid', 'price', 'bedrooms', 'bathrooms', 'livingArea', 'lotSize', 
                         'yearBuilt', 'homeType', 'latitude', 'longitude', 'description']:
                if field in property_info and field not in property_data:
                    property_data[field] = property_info.get(field)
            
            # Location info
            address = property_info.get('address', {})
            for addr_field, prop_field in [
                ('streetAddress', 'streetAddress'),
                ('city', 'city'),
                ('state', 'state'),
                ('zipcode', 'zipcode')
            ]:
                if addr_field in address and prop_field not in property_data:
                    property_data[prop_field] = address.get(addr_field)
        
        # Handle alternate 2025 structure
        if 'hdpData' in item:
            hdp_data = item.get('hdpData', {})
            home_info = hdp_data.get('homeInfo', {})
            
            # Basic property info
            for field in ['zpid', 'price', 'bedrooms', 'bathrooms', 'livingArea', 'lotSize', 
                         'yearBuilt', 'homeType', 'latitude', 'longitude']:
                if field in home_info and field not in property_data:
                    property_data[field] = home_info.get(field)
            
            # Address info
            address = home_info.get('address', {})
            for addr_field, prop_field in [
                ('streetAddress', 'streetAddress'),
                ('city', 'city'),
                ('state', 'state'),
                ('zipcode', 'zipcode')
            ]:
                if addr_field in address and prop_field not in property_data:
                    property_data[prop_field] = address.get(addr_field)
        
        # Add property to list if we have at least some key information
        if property_data.get('zpid') or property_data.get('id'):
            properties.append(property_data)
    
    return properties

@st.cache_data
def clean_dataframe(df):
    """
    Clean and preprocess property dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing property data
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure price is numeric and handle currency formatting
    if 'price' in df.columns:
        # Convert price to numeric, handling strings with currency symbols
        df['price'] = df['price'].astype(str).str.replace('[\$,]', '', regex=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Remove properties with missing or zero prices
        df = df[df['price'] > 0]
    
    # Convert area fields to numeric
    for field in ['livingArea', 'lotSize']:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce')
    
    # Ensure bedrooms and bathrooms are numeric
    for field in ['bedrooms', 'bathrooms']:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce')
            
            # Cap unrealistic values
            max_value = 10 if field == 'bedrooms' else 8
            df.loc[df[field] > max_value, field] = np.nan
    
    # Ensure year is numeric and within reasonable range
    if 'yearBuilt' in df.columns:
        df['yearBuilt'] = pd.to_numeric(df['yearBuilt'], errors='coerce')
        df.loc[(df['yearBuilt'] < 1800) | (df['yearBuilt'] > 2025), 'yearBuilt'] = np.nan
    
    # Handle duplicate zpids by keeping the most complete record
    if 'zpid' in df.columns:
        # Count non-null values for each duplicate
        if df['zpid'].duplicated().any():
            # Create completeness score based on non-null values
            completeness = df.notnull().sum(axis=1)
            
            # Sort by completeness (descending) and keep first of each duplicate
            df = df.sort_values(by=completeness.name, ascending=False)
            df = df.drop_duplicates(subset='zpid', keep='first')
    
    # Ensure all string columns are actually strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].replace('nan', np.nan)
    
    # Create full address field
    address_parts = []
    for field in ['streetAddress', 'city', 'state', 'zipcode']:
        if field in df.columns:
            address_parts.append(df[field].fillna('').astype(str))
    
    if address_parts:
        df['full_address'] = address_parts[0]
        for part in address_parts[1:]:
            df['full_address'] = df['full_address'] + ', ' + part
        
        # Clean up addresses
        df['full_address'] = df['full_address'].str.replace(' , ', ', ', regex=False)
        df['full_address'] = df['full_address'].str.replace(', , ', ', ', regex=False)
        df['full_address'] = df['full_address'].str.replace('^, ', '', regex=True)
        df['full_address'] = df['full_address'].str.replace(', $', '', regex=True)
        
        # Remove rows with empty addresses
        df.loc[df['full_address'].str.strip() == '', 'full_address'] = np.nan
    
    # Drop rows that are missing critical data (price, location)
    if 'price' in df.columns:
        df = df.dropna(subset=['price'])
    
    return df

@st.cache_data
def process_zillow_data(data_paths_2023, data_paths_2025, max_records=5000, output_dir=None):
    """
    Process Zillow data from 2023 and 2025 datasets
    
    Parameters:
    -----------
    data_paths_2023 : list or str
        File path(s) for 2023 dataset
    data_paths_2025 : list or str
        File path(s) for 2025 dataset
    max_records : int, optional
        Maximum number of records to process per dataset
    output_dir : str, optional
        Directory to save processed data
        
    Returns:
    --------
    tuple
        (df_2023, df_2025, df_combined) - Processed DataFrames
    """
    # Convert single paths to lists
    if isinstance(data_paths_2023, str):
        data_paths_2023 = [data_paths_2023]
    if isinstance(data_paths_2025, str):
        data_paths_2025 = [data_paths_2025]
    
    # Process 2023 data
    print("\nProcessing 2023 dataset...")
    data_2023 = load_multiple_json_files(data_paths_2023, max_records)
    df_2023 = None
    if data_2023:
        properties_2023 = extract_property_features(data_2023, dataset_year=2023)
        if properties_2023:
            df_2023 = pd.DataFrame(properties_2023)
            df_2023 = clean_dataframe(df_2023)
            print(f"2023 dataset processed: {len(df_2023)} properties")
    
    # Process 2025 data
    print("\nProcessing 2025 dataset...")
    data_2025 = load_multiple_json_files(data_paths_2025, max_records)
    df_2025 = None
    if data_2025:
        properties_2025 = extract_property_features(data_2025, dataset_year=2025)
        if properties_2025:
            df_2025 = pd.DataFrame(properties_2025)
            df_2025 = clean_dataframe(df_2025)
            print(f"2025 dataset processed: {len(df_2025)} properties")
    
    # Create combined dataset if both exist
    df_combined = None
    if df_2023 is not None and df_2025 is not None:
        print("\nCreating combined dataset...")
        
        # Find common columns between datasets
        common_cols = list(set(df_2023.columns) & set(df_2025.columns))
        
        # Combine datasets
        df_combined = pd.concat([
            df_2023[common_cols], 
            df_2025[common_cols]
        ], ignore_index=True)
        
        print(f"Combined dataset created: {len(df_combined)} properties")
    
    # Save processed data if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        if df_2023 is not None:
            df_2023.to_csv(os.path.join(output_dir, 'zillow_data_2023_processed.csv'), index=False)
            print(f"2023 data saved to {os.path.join(output_dir, 'zillow_data_2023_processed.csv')}")
        
        if df_2025 is not None:
            df_2025.to_csv(os.path.join(output_dir, 'zillow_data_2025_processed.csv'), index=False)
            print(f"2025 data saved to {os.path.join(output_dir, 'zillow_data_2025_processed.csv')}")
        
        if df_combined is not None:
            df_combined.to_csv(os.path.join(output_dir, 'zillow_data_combined_processed.csv'), index=False)
            print(f"Combined data saved to {os.path.join(output_dir, 'zillow_data_combined_processed.csv')}")
    
    return df_2023, df_2025, df_combined

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Zillow data for the Zillow Data Viewer application')
    parser.add_argument('--data-dir-2023', type=str, help='Directory containing 2023 Zillow data')
    parser.add_argument('--data-dir-2025', type=str, help='Directory containing 2025 Zillow data')
    parser.add_argument('--output-dir', type=str, default='processed_data', help='Directory to save processed data')
    parser.add_argument('--max-records', type=int, default=5000, help='Maximum number of records to process per dataset')
    
    args = parser.parse_args()
    
    # Set up paths
    if args.data_dir_2023:
        data_dir_2023 = Path(args.data_dir_2023)
        data_paths_2023 = [str(data_dir_2023 / f) for f in os.listdir(data_dir_2023) if f.endswith('.json')]
    else:
        data_paths_2023 = []
    
    if args.data_dir_2025:
        data_dir_2025 = Path(args.data_dir_2025)
        data_paths_2025 = [str(data_dir_2025 / f) for f in os.listdir(data_dir_2025) if f.endswith('.json')]
    else:
        data_paths_2025 = []
    
    # Process data
    process_zillow_data(data_paths_2023, data_paths_2025, args.max_records, args.output_dir) 