#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from data_processor import load_json_data, load_multiple_json_files, extract_property_features, clean_dataframe

def process_zillow_data_to_csv(data_dir_2023=None, data_dir_2025=None, output_dir="processed_data", max_records=None):
    """
    Process Zillow data from 2023 and 2025 datasets and save as CSV files
    
    Parameters:
    -----------
    data_dir_2023 : str
        Directory containing 2023 dataset files
    data_dir_2025 : str
        Directory containing 2025 dataset files
    output_dir : str
        Directory to save processed CSV files
    max_records : int
        Maximum number of records to process per dataset (None for all)
    """
    print("Starting Zillow data processing...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process 2023 dataset if directory provided
    df_2023 = None
    if data_dir_2023 and os.path.exists(data_dir_2023):
        print(f"\nProcessing 2023 dataset from {data_dir_2023}")
        
        # Get list of JSON files in 2023 directory
        json_files_2023 = [os.path.join(data_dir_2023, f) for f in os.listdir(data_dir_2023) if f.endswith('.json')]
        
        if json_files_2023:
            print(f"Found {len(json_files_2023)} JSON files in 2023 dataset")
            
            # Load data from all JSON files
            data_2023 = load_multiple_json_files(json_files_2023, max_records=max_records)
            
            if data_2023:
                print(f"Loaded {len(data_2023)} records from 2023 dataset")
                
                # Special handling for 2023 dataset structure
                # Check if we need special handling for the structure
                if any('props' in item for item in data_2023[:10] if isinstance(item, dict)):
                    print("Detected 2023 nested structure with 'props' field")
                    
                    # Extract property features with special handling
                    print("Extracting property features from 2023 data...")
                    properties_2023 = []
                    
                    for item in tqdm(data_2023, desc="Processing 2023 properties"):
                        # Skip items that aren't dictionaries
                        if not isinstance(item, dict):
                            continue
                        
                        property_data = {'dataset_year': 2023}
                        
                        # Handle 2023 dataset structure with nested props
                        if 'props' in item:
                            props = item.get('props', {})
                            property_info = props.get('pageProps', {}).get('initialData', {}).get('building', {})
                            
                            # Basic property info
                            for field in ['zpid', 'price', 'bedrooms', 'bathrooms', 'livingArea', 'lotSize', 
                                          'yearBuilt', 'homeType', 'latitude', 'longitude', 'description']:
                                if field in property_info:
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
                                if addr_field in address:
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
                        
                        # Extract features from description if available
                        desc = str(property_data.get('description', '')).lower()
                        property_data['hasGarage'] = 'garage' in desc
                        property_data['hasParkingSpot'] = 'parking' in desc
                        property_data['hasPool'] = 'pool' in desc
                        property_data['hasAC'] = 'air conditioning' in desc or 'ac' in desc or ' ac ' in desc
                        
                        # Only append if we have some meaningful data
                        if property_data and 'zpid' in property_data and 'price' in property_data:
                            properties_2023.append(property_data)
                else:
                    # Use standard extraction for non-nested structure
                    print("Using standard property feature extraction for 2023 data")
                    properties_2023 = extract_property_features(data_2023, dataset_year=2023)
                
                if properties_2023:
                    # Convert to DataFrame and clean
                    df_2023 = pd.DataFrame(properties_2023)
                    df_2023 = clean_dataframe(df_2023)
                    
                    # Save to CSV
                    csv_path_2023 = os.path.join(output_dir, "zillow_data_2023_processed.csv")
                    df_2023.to_csv(csv_path_2023, index=False)
                    
                    print(f"Saved 2023 dataset with {len(df_2023)} records to {csv_path_2023}")
                else:
                    print("Failed to extract property features from 2023 data")
            else:
                print("No data loaded from 2023 dataset files")
        else:
            print(f"No JSON files found in 2023 dataset directory: {data_dir_2023}")
    
    # Process 2025 dataset if directory provided
    df_2025 = None
    if data_dir_2025 and os.path.exists(data_dir_2025):
        print(f"\nProcessing 2025 dataset from {data_dir_2025}")
        
        # Get list of JSON files in 2025 directory
        json_files_2025 = [os.path.join(data_dir_2025, f) for f in os.listdir(data_dir_2025) if f.endswith('.json')]
        
        if json_files_2025:
            print(f"Found {len(json_files_2025)} JSON files in 2025 dataset")
            
            # Process each file individually and combine later to avoid memory issues
            all_properties_2025 = []
            
            for json_file in json_files_2025:
                file_size_mb = os.path.getsize(json_file) / (1024 * 1024)
                print(f"Processing {os.path.basename(json_file)} ({file_size_mb:.2f} MB)...")
                
                # Calculate records to load from each file based on total max_records
                file_max_records = None
                if max_records is not None:
                    file_max_records = max_records // len(json_files_2025)
                    # Ensure at least 1000 records per file
                    file_max_records = max(1000, file_max_records)
                
                # Load data from this file
                try:
                    data_file = load_json_data(json_file, max_records=file_max_records)
                    
                    if data_file:
                        print(f"Loaded {len(data_file)} records from {os.path.basename(json_file)}")
                        
                        # Extract property features
                        properties_file = extract_property_features(data_file, dataset_year=2025)
                        
                        if properties_file:
                            print(f"Extracted {len(properties_file)} properties from {os.path.basename(json_file)}")
                            all_properties_2025.extend(properties_file)
                        else:
                            print(f"Failed to extract property features from {os.path.basename(json_file)}")
                    else:
                        print(f"No data loaded from {os.path.basename(json_file)}")
                except Exception as e:
                    print(f"Error processing {os.path.basename(json_file)}: {e}")
            
            if all_properties_2025:
                # Convert combined properties to DataFrame and clean
                print(f"Converting {len(all_properties_2025)} properties to DataFrame...")
                df_2025 = pd.DataFrame(all_properties_2025)
                df_2025 = clean_dataframe(df_2025)
                
                # Save to CSV
                csv_path_2025 = os.path.join(output_dir, "zillow_data_2025_processed.csv")
                df_2025.to_csv(csv_path_2025, index=False)
                
                print(f"Saved 2025 dataset with {len(df_2025)} records to {csv_path_2025}")
            else:
                print("No properties extracted from 2025 dataset")
        else:
            print(f"No JSON files found in 2025 dataset directory: {data_dir_2025}")
    
    # Create combined dataset if both datasets were processed
    if df_2023 is not None and df_2025 is not None:
        print("\nCreating combined dataset...")
        
        # Find common columns between datasets
        common_cols = list(set(df_2023.columns) & set(df_2025.columns))
        print(f"Found {len(common_cols)} common columns between 2023 and 2025 datasets")
        
        # Combine datasets
        df_combined = pd.concat([
            df_2023[common_cols], 
            df_2025[common_cols]
        ], ignore_index=True)
        
        # Save combined dataset to CSV
        csv_path_combined = os.path.join(output_dir, "zillow_data_combined_processed.csv")
        df_combined.to_csv(csv_path_combined, index=False)
        
        print(f"Saved combined dataset with {len(df_combined)} records to {csv_path_combined}")
    
    # Sample datasets creation has been removed to process only full datasets
    
    print("\nData processing complete!")
    
    # Return paths to created CSV files
    return {
        "2023": os.path.join(output_dir, "zillow_data_2023_processed.csv") if df_2023 is not None else None,
        "2025": os.path.join(output_dir, "zillow_data_2025_processed.csv") if df_2025 is not None else None,
        "combined": os.path.join(output_dir, "zillow_data_combined_processed.csv") if df_2023 is not None and df_2025 is not None else None
    }

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Process Zillow data from 2023 and 2025 datasets and save as CSV files')
    parser.add_argument('--data-dir-2023', type=str, default='Zillow_Data_2023', help='Directory containing 2023 dataset files')
    parser.add_argument('--data-dir-2025', type=str, default='Zillow_Data_2025', help='Directory containing 2025 dataset files')
    parser.add_argument('--output-dir', type=str, default='processed_data', help='Directory to save processed CSV files')
    parser.add_argument('--max-records', type=int, default=None, help='Maximum number of records to process per dataset (default: all)')
    parser.add_argument('--year', type=str, choices=['2023', '2025', 'both'], default='both', help='Which year(s) to process')
    
    args = parser.parse_args()
    
    # Set directories based on selected years
    data_dir_2023 = args.data_dir_2023 if args.year in ['2023', 'both'] else None
    data_dir_2025 = args.data_dir_2025 if args.year in ['2025', 'both'] else None
    
    # Process data
    csv_paths = process_zillow_data_to_csv(data_dir_2023, data_dir_2025, args.output_dir, args.max_records)
    
    # Print summary of CSV files created
    print("\nSummary of CSV files created:")
    for dataset, path in csv_paths.items():
        if path and os.path.exists(path):
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"{dataset}: {path} ({file_size_mb:.2f} MB)")
    
    print("\nTo load these datasets into the Zillow Data Viewer app:")
    print("1. Use the 'Upload CSV Files' option in the sidebar")
    print("2. The full processed datasets will be used for all analysis")

if __name__ == "__main__":
    main() 