#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

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
        
        # Standard handling for 2025 and other formats
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to load the entire file first
            try:
                content = f.read()
                # Check if the content starts with an array
                if content.strip().startswith('['):
                    data = json.loads(content)
                    if max_records:
                        data = data[:max_records]
                else:
                    # Try to parse each line as a separate JSON object
                    f.seek(0)
                    for i, line in enumerate(f):
                        if max_records and i >= max_records:
                            break
                        try:
                            line = line.strip()
                            if line and not line in ['[', ']']:
                                # Remove trailing comma if exists
                                if line.endswith(','):
                                    line = line[:-1]
                                obj = json.loads(line)
                                data.append(obj)
                        except json.JSONDecodeError:
                            continue
            except json.JSONDecodeError:
                # If loading as a whole fails, try line-by-line approach
                f.seek(0)
                for i, line in enumerate(f):
                    if max_records and i >= max_records:
                        break
                    try:
                        line = line.strip()
                        if line and not line in ['[', ']']:
                            # Remove trailing comma if exists
                            if line.endswith(','):
                                line = line[:-1]
                            obj = json.loads(line)
                            data.append(obj)
                    except json.JSONDecodeError:
                        continue
            
        print(f"Loaded {len(data)} records from {os.path.basename(file_path)}")
        return data
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

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
                ('zipcode', 'zipcode'),
                ('neighborhood', 'neighborhood')
            ]:
                if addr_field in address and prop_field not in property_data:
                    property_data[prop_field] = address.get(addr_field)
            
            # Tax info
            tax_history = property_info.get('taxHistory', [])
            if tax_history and isinstance(tax_history, list) and len(tax_history) > 0:
                latest_tax = tax_history[0]
                if isinstance(latest_tax, dict):
                    property_data['taxPaid'] = latest_tax.get('taxPaid')
                    property_data['taxValue'] = latest_tax.get('value')
                    property_data['taxYear'] = latest_tax.get('year')
            
            # Price history
            price_history = property_info.get('priceHistory', [])
            if price_history and isinstance(price_history, list) and len(price_history) > 0:
                latest_price = price_history[0]
                if isinstance(latest_price, dict):
                    property_data['lastSoldPrice'] = latest_price.get('price')
                    property_data['lastSoldDate'] = latest_price.get('date')
        
        # Handle nested address structure
        elif 'address' in item and isinstance(item['address'], dict):
            address = item.get('address', {})
            for addr_field, prop_field in [
                ('streetAddress', 'streetAddress'),
                ('city', 'city'),
                ('state', 'state'),
                ('zipcode', 'zipcode'),
                ('neighborhood', 'neighborhood')
            ]:
                if addr_field in address and prop_field not in property_data:
                    property_data[prop_field] = address.get(addr_field)
            
        # Tax info
        tax_history = item.get('taxHistory', [])
        if tax_history and isinstance(tax_history, list) and len(tax_history) > 0:
            latest_tax = tax_history[0]
            if isinstance(latest_tax, dict):
                property_data['taxPaid'] = latest_tax.get('taxPaid')
                property_data['taxValue'] = latest_tax.get('value')
                property_data['taxYear'] = latest_tax.get('year')
        
        # Price history
        price_history = item.get('priceHistory', [])
        if price_history and isinstance(price_history, list) and len(price_history) > 0:
            latest_price = price_history[0]
            if isinstance(latest_price, dict):
                property_data['lastSoldPrice'] = latest_price.get('price')
                property_data['lastSoldDate'] = latest_price.get('date')
        
        # Ensure we have at least an ID field
        if 'zpid' not in property_data and 'id' in property_data:
            property_data['zpid'] = property_data['id']
        
        # Extract features from description if available
        desc = str(property_data.get('description', '')).lower()
        property_data['hasGarage'] = 'garage' in desc
        property_data['hasParkingSpot'] = 'parking' in desc
        property_data['hasPool'] = 'pool' in desc
        property_data['hasAC'] = 'air conditioning' in desc or 'ac' in desc or ' ac ' in desc
        
        # Only append if we have some meaningful data
        if property_data and ('zpid' in property_data or 'id' in property_data):
            properties.append(property_data)
    
    return properties

def clean_dataframe(df):
    """
    Clean and prepare the dataframe for analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing property data
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame
    """
    # Ensure we have a zpid column, even if it's from the id column
    if 'zpid' not in df.columns and 'id' in df.columns:
        df['zpid'] = df['id']
    
    # Convert columns to appropriate types
    numeric_cols = ['price', 'bedrooms', 'bathrooms', 'livingArea', 'lotSize', 
                   'yearBuilt', 'taxPaid', 'taxValue', 'lastSoldPrice']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing critical values
    critical_cols = [col for col in ['zpid', 'price'] if col in df.columns]
    if critical_cols:
        df = df.dropna(subset=critical_cols)
    
    # Clean up boolean columns
    boolean_cols = ['hasGarage', 'hasParkingSpot', 'hasPool', 'hasAC']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False)
    
    # Generate derived features
    if 'price' in df.columns and 'livingArea' in df.columns:
        mask = (df['livingArea'] > 0) & df['livingArea'].notna() & df['price'].notna()
        df.loc[mask, 'pricePerSqFt'] = df.loc[mask, 'price'] / df.loc[mask, 'livingArea']
    
    if 'lastSoldDate' in df.columns:
        df['lastSoldDate'] = pd.to_datetime(df['lastSoldDate'], errors='coerce')
    
    # Remove extreme outliers (properties with prices more than 3 standard deviations from mean)
    if 'price' in df.columns:
        mean_price = df['price'].mean()
        std_price = df['price'].std()
        df = df[(df['price'] > mean_price - 3*std_price) & (df['price'] < mean_price + 3*std_price)]
    
    return df

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