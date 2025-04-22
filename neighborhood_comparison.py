import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_neighborhood_comparison(df_2023, df_2025, df_combined):
    """
    Display neighborhood comparison dashboard
    
    Parameters:
    -----------
    df_2023 : DataFrame or None
        2023 property data
    df_2025 : DataFrame or None
        2025 property data
    df_combined : DataFrame or None
        Combined property data
    """
    st.header("🏘️ Neighborhood Comparison")
    st.markdown("Compare statistics across different neighborhoods")
    
    # Check if we have any dataset loaded
    if df_2023 is None and df_2025 is None and df_combined is None:
        st.warning("Please load at least one dataset to use this visualization.")
        return
    
    # Create dataset selector
    available_dfs = []
    if df_2023 is not None:
        available_dfs.append("2023 Dataset")
    if df_2025 is not None:
        available_dfs.append("2025 Dataset")
    if df_combined is not None:
        available_dfs.append("Combined Dataset")
    
    selected_dataset = st.selectbox(
        "Select Dataset", 
        options=available_dfs,
        index=len(available_dfs)-1  # Default to the last (most recent) dataset
    )
    
    # Get the selected dataframe
    if selected_dataset == "2023 Dataset":
        df = df_2023
        year_filter = None
    elif selected_dataset == "2025 Dataset":
        df = df_2025
        year_filter = None
    else:  # Combined Dataset
        df = df_combined
        year_filter = st.radio(
            "Filter by Year",
            options=["All Years", "2023 Only", "2025 Only"],
            index=0
        )
        
        if year_filter == "2023 Only" and "dataset_year" in df.columns:
            df = df[df["dataset_year"] == 2023]
        elif year_filter == "2025 Only" and "dataset_year" in df.columns:
            df = df[df["dataset_year"] == 2025]
    
    if df is None or len(df) == 0:
        st.warning("The selected dataset is empty.")
        return
    
    # Ensure price column is numeric
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        
        if len(df) == 0:
            st.error("No valid price data available after cleaning. Cannot proceed.")
            return
    else:
        st.error("Price column not found in the dataset. Cannot proceed with neighborhood comparison.")
        return
    
    # Check for location columns
    location_columns = [col for col in ['neighborhood', 'city', 'zipcode'] if col in df.columns]
    
    if not location_columns:
        st.error("No location columns (neighborhood, city, zipcode) found in the dataset.")
        return
    
    # Select location level
    location_level = st.selectbox(
        "Select Location Level",
        options=location_columns,
        index=0 if 'neighborhood' in location_columns else 0
    )
    
    # Get list of unique locations
    locations = df[location_level].dropna().unique()
    
    if len(locations) == 0:
        st.error(f"No valid {location_level} values found in the dataset.")
        return
    
    # Limit to locations with enough data
    location_counts = df[location_level].value_counts()
    min_properties = 5  # Minimum number of properties for analysis
    valid_locations = location_counts[location_counts >= min_properties].index.tolist()
    
    if len(valid_locations) == 0:
        st.error(f"No {location_level}s found with at least {min_properties} properties.")
        return
    
    # Select locations to compare
    num_locations = min(10, len(valid_locations))
    default_locations = valid_locations[:num_locations]
    
    selected_locations = st.multiselect(
        f"Select {location_level.title()}s to Compare",
        options=valid_locations,
        default=default_locations
    )
    
    if not selected_locations:
        st.warning(f"Please select at least one {location_level} for comparison.")
        return
    
    # Filter data to selected locations
    df_filtered = df[df[location_level].isin(selected_locations)]
    
    # Main comparison dashboard
    st.markdown(f"## {location_level.title()} Comparison")
    
    # Price statistics by location
    st.markdown("### Price Comparison")
    
    # Calculate price statistics by location
    price_stats = df_filtered.groupby(location_level)['price'].agg([
        'mean', 'median', 'min', 'max', 'count', 'std'
    ]).reset_index()
    
    price_stats.columns = [
        location_level, 'Mean Price', 'Median Price', 'Min Price', 
        'Max Price', 'Property Count', 'Price Std Dev'
    ]
    
    # Sort by median price
    price_stats = price_stats.sort_values('Median Price', ascending=False)
    
    # Display bar chart of median prices
    fig1 = px.bar(
        price_stats,
        x=location_level,
        y='Median Price',
        title=f"Median Price by {location_level.title()}",
        color='Median Price',
        text_auto=True,
        color_continuous_scale=px.colors.sequential.Blugrn,
        labels={location_level: location_level.title(), 'Median Price': 'Median Price ($)'}
    )
    fig1.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
    fig1.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig1, use_container_width=True)
    
    # Price distribution by location
    fig2 = px.box(
        df_filtered,
        x=location_level,
        y='price',
        title=f"Price Distribution by {location_level.title()}",
        color=location_level,
        labels={location_level: location_level.title(), 'price': 'Price ($)'}
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Property characteristics comparison
    st.markdown("### Property Characteristics Comparison")
    
    # Convert potential numeric columns to ensure proper data type
    for col in df_filtered.columns:
        if col not in ['zpid', 'id', 'streetAddress', 'description', 'city', 'state', 'zipcode', 'homeType', 'neighborhood']:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    
    # Select characteristic for comparison
    characteristic_options = [col for col in ['bedrooms', 'bathrooms', 'livingArea', 'yearBuilt'] 
                             if col in df_filtered.columns and df_filtered[col].notna().sum() > len(df_filtered) * 0.1]
    
    if not characteristic_options:
        st.warning("No property characteristic columns found with sufficient data for comparison.")
    else:
        selected_characteristic = st.selectbox(
            "Select Characteristic",
            options=characteristic_options
        )
        
        if selected_characteristic:
            # Filter out rows with NaN values for the selected characteristic
            valid_data = df_filtered.dropna(subset=[selected_characteristic])
            
            if len(valid_data) < min_properties:
                st.warning(f"Not enough valid data points for {selected_characteristic}. Please select another characteristic.")
            else:
                try:
                    # Calculate characteristic statistics by location
                    char_stats = valid_data.groupby(location_level)[selected_characteristic].agg([
                        'mean', 'median', 'min', 'max', 'count'
                    ]).reset_index()
                    
                    char_stats.columns = [
                        location_level, 'Mean', 'Median', 'Min', 'Max', 'Count'
                    ]
                    
                    # Sort by median value
                    char_stats = char_stats.sort_values('Median', ascending=False)
                    
                    # Display bar chart of median characteristic
                    fig3 = px.bar(
                        char_stats,
                        x=location_level,
                        y='Median',
                        title=f"Median {selected_characteristic.replace('_', ' ').title()} by {location_level.title()}",
                        color='Median',
                        text_auto=True,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        labels={
                            location_level: location_level.title(), 
                            'Median': f"Median {selected_characteristic.replace('_', ' ').title()}"
                        }
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Distribution of characteristic by location
                    fig4 = px.box(
                        valid_data,
                        x=location_level,
                        y=selected_characteristic,
                        title=f"{selected_characteristic.replace('_', ' ').title()} Distribution by {location_level.title()}",
                        color=location_level,
                        labels={
                            location_level: location_level.title(), 
                            selected_characteristic: selected_characteristic.replace('_', ' ').title()
                        }
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                    
                    # Show relationship between price and selected characteristic
                    st.markdown(f"### Price vs {selected_characteristic.replace('_', ' ').title()} by {location_level.title()}")
                    
                    # Create scatter plot of price vs characteristic by location
                    fig5 = px.scatter(
                        valid_data,
                        x=selected_characteristic,
                        y='price',
                        color=location_level,
                        title=f"Price vs {selected_characteristic.replace('_', ' ').title()} by {location_level.title()}",
                        labels={
                            selected_characteristic: selected_characteristic.replace('_', ' ').title(),
                            'price': 'Price ($)'
                        },
                        opacity=0.7,
                        trendline="ols"
                    )
                    st.plotly_chart(fig5, use_container_width=True)
                except Exception as e:
                    st.error(f"Error analyzing characteristic data: {e}")
                    st.warning("The selected characteristic may contain invalid values. Please select another.")
    
    # Price per square foot comparison (if livingArea available)
    if 'livingArea' in df_filtered.columns:
        st.markdown("### Price per Square Foot Comparison")
        
        # Calculate price per square foot
        df_filtered['pricePerSqFt'] = df_filtered.apply(
            lambda x: x['price'] / x['livingArea'] if x['livingArea'] > 0 else None, 
            axis=1
        )
        
        # Remove outliers (more than 3 std dev from mean)
        mean_ppsf = df_filtered['pricePerSqFt'].mean()
        std_ppsf = df_filtered['pricePerSqFt'].std()
        df_filtered = df_filtered[
            (df_filtered['pricePerSqFt'] >= mean_ppsf - 3*std_ppsf) & 
            (df_filtered['pricePerSqFt'] <= mean_ppsf + 3*std_ppsf)
        ]
        
        # Calculate price per square foot statistics by location
        ppsf_stats = df_filtered.groupby(location_level)['pricePerSqFt'].agg([
            'mean', 'median', 'min', 'max', 'count'
        ]).reset_index()
        
        ppsf_stats.columns = [
            location_level, 'Mean Price/sqft', 'Median Price/sqft', 
            'Min Price/sqft', 'Max Price/sqft', 'Count'
        ]
        
        # Sort by median price per square foot
        ppsf_stats = ppsf_stats.sort_values('Median Price/sqft', ascending=False)
        
        # Display bar chart of median price per square foot
        fig6 = px.bar(
            ppsf_stats,
            x=location_level,
            y='Median Price/sqft',
            title=f"Median Price per Square Foot by {location_level.title()}",
            color='Median Price/sqft',
            text_auto=True,
            color_continuous_scale=px.colors.sequential.Reds,
            labels={
                location_level: location_level.title(), 
                'Median Price/sqft': 'Median Price per Square Foot ($)'
            }
        )
        fig6.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
        st.plotly_chart(fig6, use_container_width=True)
    
    # Property type distribution by location (if homeType available)
    if 'homeType' in df_filtered.columns:
        st.markdown("### Property Type Distribution")
        
        # Create a crosstab of location by property type
        type_distribution = pd.crosstab(
            df_filtered[location_level], 
            df_filtered['homeType'],
            normalize='index'
        ) * 100
        
        # Plot stacked bar chart
        fig7 = px.bar(
            type_distribution.reset_index(),
            x=location_level,
            y=type_distribution.columns.tolist(),
            title=f"Property Type Distribution by {location_level.title()}",
            labels={
                'value': 'Percentage (%)',
                'variable': 'Property Type'
            },
            barmode='stack'
        )
        st.plotly_chart(fig7, use_container_width=True)
    
    # Year built comparison (if yearBuilt available)
    if 'yearBuilt' in df_filtered.columns:
        st.markdown("### Property Age Comparison")
        
        # Calculate year built statistics by location
        year_stats = df_filtered.groupby(location_level)['yearBuilt'].agg([
            'mean', 'median', 'min', 'max', 'count'
        ]).reset_index()
        
        year_stats.columns = [
            location_level, 'Mean Year', 'Median Year', 
            'Oldest Property', 'Newest Property', 'Count'
        ]
        
        # Sort by median year
        year_stats = year_stats.sort_values('Median Year', ascending=True)
        
        # Calculate average age
        current_year = 2025
        year_stats['Median Age'] = current_year - year_stats['Median Year']
        
        # Display bar chart of median property age
        fig8 = px.bar(
            year_stats,
            x=location_level,
            y='Median Age',
            title=f"Median Property Age by {location_level.title()}",
            color='Median Age',
            text_auto=True,
            color_continuous_scale=px.colors.sequential.Oranges,
            labels={
                location_level: location_level.title(), 
                'Median Age': 'Median Property Age (years)'
            }
        )
        st.plotly_chart(fig8, use_container_width=True)
        
        # Year built distribution by location
        fig9 = px.box(
            df_filtered,
            x=location_level,
            y='yearBuilt',
            title=f"Year Built Distribution by {location_level.title()}",
            color=location_level,
            labels={
                location_level: location_level.title(), 
                'yearBuilt': 'Year Built'
            }
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    # Combined comparison metrics
    st.markdown("### Combined Metrics Comparison")
    
    # Radar chart for locations (select up to 5 locations for clarity)
    display_locations = selected_locations[:5]
    df_radar = df_filtered[df_filtered[location_level].isin(display_locations)]
    
    if len(display_locations) > 0:
        # Metrics to include in radar chart
        radar_metrics = []
        
        if 'price' in df_radar.columns:
            radar_metrics.append('price')
        
        if 'livingArea' in df_radar.columns:
            radar_metrics.append('livingArea')
        
        if 'bedrooms' in df_radar.columns:
            radar_metrics.append('bedrooms')
        
        if 'bathrooms' in df_radar.columns:
            radar_metrics.append('bathrooms')
        
        if 'yearBuilt' in df_radar.columns:
            radar_metrics.append('yearBuilt')
        
        if radar_metrics:
            # Calculate median values for each metric by location
            radar_data = df_radar.groupby(location_level)[radar_metrics].median().reset_index()
            
            # Normalize data for radar chart (0-1 scale)
            for metric in radar_metrics:
                min_val = radar_data[metric].min()
                max_val = radar_data[metric].max()
                if max_val > min_val:
                    radar_data[f"{metric}_norm"] = (radar_data[metric] - min_val) / (max_val - min_val)
                else:
                    radar_data[f"{metric}_norm"] = 0.5
            
            # Create radar chart
            fig10 = go.Figure()
            
            for i, location in enumerate(radar_data[location_level]):
                fig10.add_trace(go.Scatterpolar(
                    r=[radar_data.loc[radar_data[location_level] == location, f"{m}_norm"].values[0] for m in radar_metrics],
                    theta=[m.replace('_', ' ').title() for m in radar_metrics],
                    fill='toself',
                    name=location
                ))
            
            fig10.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title=f"Comparison of {location_level.title()}s (Normalized Values)",
                showlegend=True
            )
            
            st.plotly_chart(fig10, use_container_width=True)
    
    # Price trends by location if combined dataset with both years
    if selected_dataset == "Combined Dataset" and "dataset_year" in df.columns:
        if len(df_filtered[df_filtered['dataset_year'] == 2023]) > 0 and len(df_filtered[df_filtered['dataset_year'] == 2025]) > 0:
            st.markdown("### Price Trends (2023 vs 2025)")
            
            # Calculate median price by location and year
            price_trends = df_filtered.groupby([location_level, 'dataset_year'])['price'].median().reset_index()
            
            # Pivot to get years as columns
            price_trends_pivot = price_trends.pivot(index=location_level, columns='dataset_year', values='price')
            
            # Calculate price change
            if 2023 in price_trends_pivot.columns and 2025 in price_trends_pivot.columns:
                price_trends_pivot['change'] = price_trends_pivot[2025] - price_trends_pivot[2023]
                price_trends_pivot['pct_change'] = (price_trends_pivot['change'] / price_trends_pivot[2023]) * 100
                
                # Reset index for plotting
                price_change_data = price_trends_pivot.reset_index()
                
                # Filter to locations with data for both years
                price_change_data = price_change_data.dropna(subset=[2023, 2025])
                
                if not price_change_data.empty:
                    # Sort by percentage change
                    price_change_data = price_change_data.sort_values('pct_change', ascending=False)
                    
                    # Plot price change
                    fig11 = px.bar(
                        price_change_data,
                        x=location_level,
                        y='pct_change',
                        title=f"Price Change by {location_level.title()} (2023-2025)",
                        color='pct_change',
                        text_auto=True,
                        color_continuous_scale=px.colors.diverging.RdBu,
                        labels={
                            location_level: location_level.title(), 
                            'pct_change': 'Price Change (%)'
                        }
                    )
                    fig11.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                    st.plotly_chart(fig11, use_container_width=True)
    
    # Summary table
    st.markdown("### Summary Table")
    
    # Compile complete statistics for comparison
    summary_cols = [location_level]
    
    # Add price statistics
    if 'price' in df_filtered.columns:
        summary_cols.extend(['Median Price', 'Mean Price', 'Property Count'])
    
    # Add living area if available
    if 'livingArea' in df_filtered.columns:
        summary_cols.append('Median Living Area')
    
    # Add bedrooms if available
    if 'bedrooms' in df_filtered.columns:
        summary_cols.append('Median Bedrooms')
    
    # Add bathrooms if available
    if 'bathrooms' in df_filtered.columns:
        summary_cols.append('Median Bathrooms')
    
    # Add year built if available
    if 'yearBuilt' in df_filtered.columns:
        summary_cols.append('Median Year Built')
    
    # Add price per square foot if calculated
    if 'pricePerSqFt' in df_filtered.columns:
        summary_cols.append('Median Price/sqft')
    
    # Create summary dataframe
    summary_df = price_stats[[location_level, 'Median Price', 'Mean Price', 'Property Count']]
    
    # Add living area
    if 'livingArea' in df_filtered.columns:
        living_area_stats = df_filtered.groupby(location_level)['livingArea'].median().reset_index()
        living_area_stats.columns = [location_level, 'Median Living Area']
        summary_df = summary_df.merge(living_area_stats, on=location_level)
    
    # Add bedrooms
    if 'bedrooms' in df_filtered.columns:
        bedroom_stats = df_filtered.groupby(location_level)['bedrooms'].median().reset_index()
        bedroom_stats.columns = [location_level, 'Median Bedrooms']
        summary_df = summary_df.merge(bedroom_stats, on=location_level)
    
    # Add bathrooms
    if 'bathrooms' in df_filtered.columns:
        bathroom_stats = df_filtered.groupby(location_level)['bathrooms'].median().reset_index()
        bathroom_stats.columns = [location_level, 'Median Bathrooms']
        summary_df = summary_df.merge(bathroom_stats, on=location_level)
    
    # Add year built
    if 'yearBuilt' in df_filtered.columns:
        year_built_stats = df_filtered.groupby(location_level)['yearBuilt'].median().reset_index()
        year_built_stats.columns = [location_level, 'Median Year Built']
        summary_df = summary_df.merge(year_built_stats, on=location_level)
    
    # Add price per square foot
    if 'pricePerSqFt' in df_filtered.columns:
        ppsf_median_stats = df_filtered.groupby(location_level)['pricePerSqFt'].median().reset_index()
        ppsf_median_stats.columns = [location_level, 'Median Price/sqft']
        summary_df = summary_df.merge(ppsf_median_stats, on=location_level)
    
    # Format numeric columns
    if 'Median Price' in summary_df.columns:
        summary_df['Median Price'] = summary_df['Median Price'].map('${:,.0f}'.format)
    if 'Mean Price' in summary_df.columns:
        summary_df['Mean Price'] = summary_df['Mean Price'].map('${:,.0f}'.format)
    if 'Median Living Area' in summary_df.columns:
        summary_df['Median Living Area'] = summary_df['Median Living Area'].map('{:,.0f} sqft'.format)
    if 'Median Price/sqft' in summary_df.columns:
        summary_df['Median Price/sqft'] = summary_df['Median Price/sqft'].map('${:.2f}'.format)
    
    # Display summary table
    st.dataframe(summary_df, use_container_width=True) 