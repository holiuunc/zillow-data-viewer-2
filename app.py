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

# Data upload options
data_source = st.sidebar.radio(
    "Select Data Source:",
    ["Upload CSV Files", "Use Sample Data", "Process Raw Data"]
)

# Initialize dataframes
df_2023 = None
df_2025 = None
df_combined = None

if data_source == "Upload CSV Files":
    st.sidebar.markdown("#### Upload processed Zillow data files")
    
    uploaded_file_2023 = st.sidebar.file_uploader("Upload 2023 Dataset (CSV)", type=["csv"])
    uploaded_file_2025 = st.sidebar.file_uploader("Upload 2025 Dataset (CSV)", type=["csv"])
    
    if uploaded_file_2023:
        df_2023 = pd.read_csv(uploaded_file_2023)
        st.sidebar.success(f"Loaded 2023 dataset: {len(df_2023)} properties")
    
    if uploaded_file_2025:
        df_2025 = pd.read_csv(uploaded_file_2025)
        st.sidebar.success(f"Loaded 2025 dataset: {len(df_2025)} properties")

elif data_source == "Use Sample Data":
    # Path to sample data directory
    sample_data_dir = Path("sample_data")
    
    if not os.path.exists(sample_data_dir):
        st.sidebar.warning("Sample data directory not found. Creating it.")
        os.makedirs(sample_data_dir, exist_ok=True)
        
        # Check if we can find sample data in parent directory
        parent_sample_dir = Path("../sample_data")
        if os.path.exists(parent_sample_dir):
            st.sidebar.info("Found sample data in parent directory. Using it.")
            sample_data_dir = parent_sample_dir
    
    sample_2023 = sample_data_dir / "zillow_data_2023_sample.csv"
    sample_2025 = sample_data_dir / "zillow_data_2025_sample.csv"
    
    load_sample = st.sidebar.button("Load Sample Data")
    
    if load_sample:
        if os.path.exists(sample_2023):
            df_2023 = load_data(sample_2023)
            if df_2023 is not None:
                st.sidebar.success(f"Loaded 2023 sample: {len(df_2023)} properties")
        else:
            st.sidebar.error(f"Sample file not found: {sample_2023}")
            
        if os.path.exists(sample_2025):
            df_2025 = load_data(sample_2025)
            if df_2025 is not None:
                st.sidebar.success(f"Loaded 2025 sample: {len(df_2025)} properties")
        else:
            st.sidebar.error(f"Sample file not found: {sample_2025}")

elif data_source == "Process Raw Data":
    from data_processor import process_zillow_data
    
    st.sidebar.markdown("#### Process Raw Zillow Data")
    
    # 2023 dataset
    data_dir_2023 = st.sidebar.text_input("2023 Data Directory", "Zillow_Data_2023")
    
    # 2025 dataset
    data_dir_2025 = st.sidebar.text_input("2025 Data Directory", "Zillow_Data_2025")
    
    # Processing options
    max_records = st.sidebar.slider("Max Records per Dataset", 1000, 20000, 5000, 1000)
    
    output_dir = st.sidebar.text_input("Output Directory", "processed_data")
    
    process_data = st.sidebar.button("Process Data")
    
    if process_data:
        with st.spinner('Processing data...'):
            # Convert paths
            if data_dir_2023:
                data_dir_2023 = Path(data_dir_2023)
                if os.path.exists(data_dir_2023):
                    data_paths_2023 = [str(data_dir_2023 / f) for f in os.listdir(data_dir_2023) if f.endswith('.json')]
                else:
                    data_paths_2023 = []
                    st.sidebar.error(f"Directory not found: {data_dir_2023}")
            else:
                data_paths_2023 = []
            
            if data_dir_2025:
                data_dir_2025 = Path(data_dir_2025)
                if os.path.exists(data_dir_2025):
                    data_paths_2025 = [str(data_dir_2025 / f) for f in os.listdir(data_dir_2025) if f.endswith('.json')]
                else:
                    data_paths_2025 = []
                    st.sidebar.error(f"Directory not found: {data_dir_2025}")
            else:
                data_paths_2025 = []
            
            # Process data if files were found
            if data_paths_2023 or data_paths_2025:
                try:
                    df_2023, df_2025, df_combined = process_zillow_data(
                        data_paths_2023, data_paths_2025, max_records, output_dir
                    )
                    
                    if df_2023 is not None:
                        st.sidebar.success(f"Processed 2023 data: {len(df_2023)} properties")
                    
                    if df_2025 is not None:
                        st.sidebar.success(f"Processed 2025 data: {len(df_2025)} properties")
                    
                    if df_combined is not None:
                        st.sidebar.success(f"Created combined dataset: {len(df_combined)} properties")
                except Exception as e:
                    st.sidebar.error(f"Error processing data: {e}")
            else:
                st.sidebar.error("No data files found. Please check the directories.")

# Create combined dataset if both are loaded and not already combined
if df_2023 is not None and df_2025 is not None and df_combined is None:
    # Make sure both have a dataset_year column
    if 'dataset_year' not in df_2023.columns:
        df_2023['dataset_year'] = 2023
    if 'dataset_year' not in df_2025.columns:
        df_2025['dataset_year'] = 2025
    
    # Find common columns
    common_cols = list(set(df_2023.columns) & set(df_2025.columns))
    
    # Combine datasets
    df_combined = pd.concat([
        df_2023[common_cols], 
        df_2025[common_cols]
    ], ignore_index=True)
    
    st.sidebar.success(f"Created combined dataset with {len(df_combined)} properties")

# Visualization selection
st.sidebar.header("Select Visualization")
viz_options = [
    "Interactive Property Map",
    "Price Trends Dashboard",
    "Property Time Comparison",
    "Property Feature Impact",
    "Market Segment Explorer",
    "Neighborhood Comparison"
]
selected_viz = st.sidebar.selectbox("Choose a visualization", viz_options)

# Display the selected visualization
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
    
    ### Data Requirements:
    
    The application works with Zillow property data in CSV format. If you have raw JSON data,
    you can use the "Process Raw Data" option to convert it to the required format.
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