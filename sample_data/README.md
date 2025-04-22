# Sample Data Directory

This directory is used to store sample datasets for the Zillow Data Viewer.

## Required Files

For the "Use Sample Data" option to work, you need to place the following files in this directory:

1. `zillow_data_2023_sample.csv` - Sample data from the 2023 Zillow dataset
2. `zillow_data_2025_sample.csv` - Sample data from the 2025 Zillow dataset
3. `zillow_data_combined_sample.csv` - Combined sample data from both datasets (optional)

## Creating Sample Files

You can create these sample files in two ways:

1. **Using the app**: Upload JSON files using the "Process Raw Data" option, then download the processed CSV files and place them in this directory.

2. **Using the sample creation script**: Run `python create_sample_data.py` with appropriate arguments to create sample datasets from raw JSON files.

## For Streamlit Cloud Deployment

When deploying to Streamlit Cloud, you should:

1. Create sample files locally
2. Place them in this directory
3. Commit them to your Git repository
4. Deploy the application to Streamlit Cloud

This ensures that sample data is available when deployed, as Streamlit Cloud has limited memory and may not be able to process large JSON files.

## File Size Recommendations

For optimal performance on Streamlit Cloud, keep sample files under these sizes:
- Individual sample files: < 25 MB
- Combined total: < 50 MB 