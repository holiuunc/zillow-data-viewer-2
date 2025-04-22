# Zillow Data Viewer

A Streamlit application for analyzing and visualizing Zillow property data from 2023 and 2025.

## Features

- Interactive property map visualization
- Price trends dashboard
- Property time comparison (properties appearing in both datasets)
- Property feature impact analysis
- Market segment explorer
- Neighborhood comparison
- Property attribute comparison

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd zillow-data-viewer-2
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

You can run the application using the provided scripts:

- On Linux/Mac:
  ```
  ./run_app.sh
  ```

- On Windows:
  ```
  run_app.bat
  ```

Alternatively, you can run it directly with:
```
streamlit run app.py
```

## Data Sources

The application can work with three types of data sources:

1. **Upload CSV Files**: Upload pre-processed CSV files with Zillow property data.
2. **Use Sample Data**: Use pre-created sample datasets for quick testing.
3. **Process Raw Data**: Process raw JSON data files from Zillow API.

### Creating Sample Data

You can create sample datasets using the provided script:

```
python create_sample_data.py --data-dir-2023 <path-to-2023-data> --data-dir-2025 <path-to-2025-data> --output-dir sample_data --sample-size 1000
```

## Deploying to Streamlit Cloud

To deploy this application to Streamlit Cloud, follow these steps:

1. **Prepare Sample Data**: Create sample data files locally following the instructions in the [sample_data/README.md](sample_data/README.md) file.

2. **Commit Sample Data to Repository**: Make sure to include the sample data files in your Git repository.

3. **Create a GitHub Repository**: Push your code to a GitHub repository.

4. **Deploy on Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with GitHub
   - Select your repository
   - Set the main file to `app.py`
   - Click "Deploy"

### Streamlit Cloud Limitations

When running on Streamlit Cloud, be aware of these limitations:

1. **Memory Limits**: Streamlit Cloud has memory limitations (~1GB), so processing large JSON files may fail.
   - Use the "Upload CSV Files" option or pre-created sample data instead.

2. **File Storage**: Files created during a session are not persistent between runs.
   - Include your sample data files in the Git repository.

3. **Processing Power**: Limited CPU resources mean complex operations may be slower.
   - Keep sample sizes reasonable (1000-5000 properties recommended).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 