#!/usr/bin/env python3
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# Import visualization modules
from property_map import display_property_map
from price_trends import display_price_trends
from feature_impact import display_feature_impact
from market_segments import display_market_segments
from neighborhood_comparison import display_neighborhood_comparison
from property_time_comparison import display_property_time_comparison
from property_comparison import display_property_comparison

# Set page configuration
st.set_page_config(
    page_title="Zillow Data Viewer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    h1 {color: #0066cc;}
    h2 {color: #0080ff;}
    h3 {color: #3399ff;}
    .stButton>button {background-color: #0066cc; color: white;}
    .stButton>button:hover {background-color: #004d99;}
    .sidebar .sidebar-content {background-color: #f5f5f5;}
</style>
""", unsafe_allow_html=True)

# App title
st.title("🏠 Zillow Data Viewer")
st.markdown("Explore and discover insights from Zillow property data (2023-2025)")

# Sidebar for data loading and options
st.sidebar.header("Data Settings")

# Function to load data
@st.cache_data
def load_data(file_path):
    """Load data from CSV file with caching"""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Convert date columns
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Data upload options
data_source = st.sidebar.radio(
    "Select Data Source:",
    ["Upload CSV Files", "Use Sample Data", "Process Raw Data"]
)

# Initialize dataframes
df_2023 = None
df_2025 = None

if data_source == "Upload CSV Files":
    st.sidebar.markdown("#### Upload processed Zillow data files")
    
    uploaded_file_2023 = st.sidebar.file_uploader("Upload 2023 Dataset (CSV)", type=["csv"])
    uploaded_file_2025 = st.sidebar.file_uploader("Upload 2025 Dataset (CSV)", type=["csv"])
    
    if uploaded_file_2023:
        try:
            df_2023 = pd.read_csv(uploaded_file_2023)
            st.sidebar.success(f"Loaded 2023 dataset: {len(df_2023)} properties")
        except Exception as e:
            st.sidebar.error(f"Error loading 2023 dataset: {e}")
    
    if uploaded_file_2025:
        try:
            df_2025 = pd.read_csv(uploaded_file_2025)
            st.sidebar.success(f"Loaded 2025 dataset: {len(df_2025)} properties")
        except Exception as e:
            st.sidebar.error(f"Error loading 2025 dataset: {e}")

elif data_source == "Use Sample Data":
    # Path to sample data directory - try multiple possible locations
    possible_paths = [
        Path("sample_data"),
        Path("./sample_data"),
        Path("../sample_data"),
        Path("/app/sample_data")  # for Streamlit Cloud
    ]
    
    sample_data_dir = None
    for path in possible_paths:
        if path.exists():
            sample_data_dir = path
            st.sidebar.success(f"Found sample data at {path}")
            break
    
    if sample_data_dir is None:
        st.sidebar.warning("Sample data directory not found.")
        st.sidebar.info("Please upload CSV files or process raw data instead.")
    else:
        sample_2023 = sample_data_dir / "zillow_data_2023_sample.csv"
        sample_2025 = sample_data_dir / "zillow_data_2025_sample.csv"
        
        has_2023_sample = sample_2023.exists()
        has_2025_sample = sample_2025.exists()
        
        if not has_2023_sample and not has_2025_sample:
            st.sidebar.warning("No sample files found in the sample data directory.")
            st.sidebar.info("Please upload CSV files or process raw data instead.")
        else:
            load_sample = st.sidebar.button("Load Sample Data")
            
            if 'df_2023' not in st.session_state:
                st.session_state.df_2023 = None
            if 'df_2025' not in st.session_state:
                st.session_state.df_2025 = None
                
            if load_sample:
                if has_2023_sample:
                    st.session_state.df_2023 = load_data(sample_2023)
                    if st.session_state.df_2023 is not None:
                        st.sidebar.success(f"Loaded 2023 sample: {len(st.session_state.df_2023)} properties")
                else:
                    st.sidebar.warning(f"Sample file not found: {sample_2023}")
                    
                if has_2025_sample:
                    st.session_state.df_2025 = load_data(sample_2025)
                    if st.session_state.df_2025 is not None:
                        st.sidebar.success(f"Loaded 2025 sample: {len(st.session_state.df_2025)} properties")
                else:
                    st.sidebar.warning(f"Sample file not found: {sample_2025}")
            
            # Use session state data if available
            df_2023 = st.session_state.df_2023
            df_2025 = st.session_state.df_2025

elif data_source == "Process Raw Data":
    st.sidebar.markdown("#### Process Raw Zillow Data")
    st.sidebar.info("Note: Processing large JSON files may not work on Streamlit Cloud due to memory constraints. Consider using this locally or uploading pre-processed CSV files.")
    
    # Allow users to upload JSON files
    st.sidebar.markdown("##### Upload or specify JSON files")
    
    # 2023 dataset
    data_files_2023 = st.sidebar.file_uploader("Upload 2023 Dataset JSON file(s)", 
                                              type=["json"], accept_multiple_files=True)
    
    # 2025 dataset
    data_files_2025 = st.sidebar.file_uploader("Upload 2025 Dataset JSON file(s)", 
                                              type=["json"], accept_multiple_files=True)
    
    # Processing options
    max_records = st.sidebar.slider("Max Records per Dataset", 500, 5000, 1000, 500)
    
    # Only show process button if files are uploaded
    if data_files_2023 or data_files_2025:
        process_data = st.sidebar.button("Process Data")
        
        if process_data:
            with st.spinner('Processing data...'):
                try:
                    # Import data processor here to avoid importing it when not needed
                    from data_processor import process_zillow_data
                    
                    # Save temporary uploaded files
                    tmp_dir = Path("temp_uploads")
                    os.makedirs(tmp_dir, exist_ok=True)
                    
                    # Process uploaded 2023 files
                    data_paths_2023 = []
                    for uploaded_file in data_files_2023:
                        tmp_path = tmp_dir / uploaded_file.name
                        with open(tmp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        data_paths_2023.append(str(tmp_path))
                    
                    # Process uploaded 2025 files
                    data_paths_2025 = []
                    for uploaded_file in data_files_2025:
                        tmp_path = tmp_dir / uploaded_file.name
                        with open(tmp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        data_paths_2025.append(str(tmp_path))
                    
                    # Process data
                    if data_paths_2023 or data_paths_2025:
                        df_2023, df_2025, df_combined = process_zillow_data(
                            data_paths_2023, data_paths_2025, max_records, None
                        )
                        
                        if df_2023 is not None:
                            st.sidebar.success(f"Processed 2023 data: {len(df_2023)} properties")
                        
                        if df_2025 is not None:
                            st.sidebar.success(f"Processed 2025 data: {len(df_2025)} properties")
                        
                        if df_combined is not None:
                            st.sidebar.success(f"Created combined dataset: {len(df_combined)} properties")
                    else:
                        st.sidebar.error("No data files provided. Please upload JSON files.")
                    
                    # Clean up temporary files
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    
                except Exception as e:
                    st.sidebar.error(f"Error processing data: {e}")
    else:
        st.sidebar.info("Please upload JSON files to process.")

# Create combined dataset if both are loaded and not already combined
if 'df_combined' not in st.session_state:
    st.session_state.df_combined = None

if df_2023 is not None and df_2025 is not None and st.session_state.df_combined is None:
    try:
        # Make sure both have a dataset_year column
        df_2023_copy = df_2023.copy()
        df_2025_copy = df_2025.copy()
        
        if 'dataset_year' not in df_2023_copy.columns:
            df_2023_copy['dataset_year'] = 2023
        if 'dataset_year' not in df_2025_copy.columns:
            df_2025_copy['dataset_year'] = 2025
        
        # Find common columns
        common_cols = list(set(df_2023_copy.columns) & set(df_2025_copy.columns))
        
        # Combine datasets
        st.session_state.df_combined = pd.concat([
            df_2023_copy[common_cols], 
            df_2025_copy[common_cols]
        ], ignore_index=True)
        
        st.sidebar.success(f"Created combined dataset with {len(st.session_state.df_combined)} properties")
    except Exception as e:
        st.sidebar.error(f"Error creating combined dataset: {e}")

df_combined = st.session_state.df_combined

# Visualization selection
st.sidebar.header("Select Visualization")
viz_options = [
    "Interactive Property Map",
    "Price Trends Dashboard",
    "Property Time Comparison",
    "Property Feature Impact",
    "Market Segment Explorer",
    "Neighborhood Comparison",
    "Property Comparison"
]
selected_viz = st.sidebar.selectbox("Choose a visualization", viz_options)

# Display the selected visualization
try:
    if selected_viz == "Interactive Property Map":
        display_property_map(df_2023, df_2025, df_combined)
    elif selected_viz == "Price Trends Dashboard":
        display_price_trends(df_2023, df_2025, df_combined)
    elif selected_viz == "Property Time Comparison":
        display_property_time_comparison(df_2023, df_2025, df_combined)
    elif selected_viz == "Property Feature Impact":
        display_feature_impact(df_2023, df_2025, df_combined)
    elif selected_viz == "Market Segment Explorer":
        display_market_segments(df_2023, df_2025, df_combined)
    elif selected_viz == "Neighborhood Comparison":
        display_neighborhood_comparison(df_2023, df_2025, df_combined)
    elif selected_viz == "Property Comparison":
        display_property_comparison(df_2023, df_2025, df_combined)
except Exception as e:
    st.error(f"Error displaying visualization: {e}")
    st.info("Please try selecting a different visualization or reload the data.")

# Show data overview if no visualization is selected or no data is loaded
if df_2023 is None and df_2025 is None:
    st.info("👈 Please load data using the sidebar options")
    
    st.markdown("""
    ## Welcome to the Zillow Data Viewer!
    
    This application helps you analyze and visualize Zillow property data from 2023 and 2025.
    
    ### Getting Started:
    
    1. **Load Data**: Use the sidebar to upload CSV files, load sample data, or process raw JSON data.
    2. **Select Visualization**: Once data is loaded, choose a visualization type from the sidebar.
    3. **Explore Insights**: Interact with the visualizations to discover trends and patterns.
    
    ### Visualization Types:
    
    - **Interactive Property Map**: Explore properties geographically with filters
    - **Price Trends Dashboard**: Analyze price changes by location and property type
    - **Property Time Comparison**: Compare properties that appear in both 2023 and 2025 datasets
    - **Property Feature Impact**: See how different features affect property values
    - **Market Segment Explorer**: Explore different segments of the housing market
    - **Neighborhood Comparison**: Compare statistics across different neighborhoods
    - **Property Comparison**: Analyze property attributes and their relation to price
    
    ### Data Requirements:
    
    The application works with Zillow property data in CSV format. If you have raw JSON data,
    you can use the "Process Raw Data" option to convert it to the required format.
    
    ### Cloud Deployment Note:
    
    If you're using this app on Streamlit Cloud, we recommend uploading pre-processed CSV files
    instead of processing raw JSON data, as the cloud environment has memory limitations.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This application was created to analyze Zillow property data from 2023 and 2025.
    
    The tool is designed to help identify trends, patterns, and insights in the housing market.
    """
) 