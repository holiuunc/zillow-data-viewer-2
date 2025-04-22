# Zillow Data Viewer

A comprehensive interactive dashboard for exploring and analyzing Zillow property data from 2023 and 2025.

## Overview

The Zillow Data Viewer is a Streamlit-based application that allows users to visualize and analyze property data across different years. The application supports various visualizations including interactive maps, price trends, property comparisons, and market segment analysis.

## Features

- **Interactive Property Map**: Explore properties geographically with customizable filters and various map styles
- **Price Trends Dashboard**: Analyze price changes by location and property type
- **Property Time Comparison**: Compare properties that appear in both 2023 and 2025 datasets to track changes
- **Property Feature Impact**: See how different features affect property values
- **Market Segment Explorer**: Explore different segments of the housing market
- **Neighborhood Comparison**: Compare statistics across different neighborhoods

## Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

1. Unzip the project file or clone the repository
2. Open a terminal/command prompt in the project directory
3. Run the setup script:

```bash
# On macOS/Linux
bash run_app.sh

# On Windows
run_app.bat
```

This will install the required dependencies and start the application.

Alternatively, you can manually install dependencies and run the app:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Using Your Own Data

The application accepts CSV files containing Zillow property data. To use your own data:

1. Launch the application
2. In the sidebar, select "Upload CSV Files"
3. Upload your 2023 and/or 2025 dataset files
4. The application will automatically process and display the data

#### Required Data Format

Your CSV files should contain the following columns:
- `zpid`: Zillow property ID (required for property matching)
- `price`: Property price
- `latitude` and `longitude`: Geographic coordinates
- `bedrooms`, `bathrooms`, `livingArea`: Property characteristics

Additional columns like `homeType`, `yearBuilt`, `city`, `state` will enhance available visualizations.

## Navigation

Once data is loaded:
1. Use the sidebar to select a visualization type
2. Apply filters to customize the view
3. Interact with the visualizations to explore insights

## Troubleshooting

- If you encounter memory issues, try using smaller datasets or apply more restrictive filters
- For CSV formatting issues, ensure your data includes the required columns and proper data types
- For any other issues, check the error messages in the terminal where you launched the app 