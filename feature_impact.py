import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

def display_feature_impact(df_2023, df_2025, df_combined):
    """
    Display feature impact analysis dashboard
    
    Parameters:
    -----------
    df_2023 : DataFrame or None
        2023 property data
    df_2025 : DataFrame or None
        2025 property data
    df_combined : DataFrame or None
        Combined property data
    """
    st.header("🔍 Property Feature Impact Analysis")
    st.markdown("Analyze how different property features impact price and value")
    
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
    elif selected_dataset == "2025 Dataset":
        df = df_2025
    else:  # Combined Dataset
        df = df_combined
        if "dataset_year" in df.columns:
            year_filter = st.radio(
                "Filter by Year",
                options=["All Years", "2023 Only", "2025 Only"],
                index=0
            )
            
            if year_filter == "2023 Only":
                df = df[df["dataset_year"] == 2023]
            elif year_filter == "2025 Only":
                df = df[df["dataset_year"] == 2025]
    
    if df is None or len(df) == 0:
        st.warning("The selected dataset is empty.")
        return
    
    # Check for price column as it's required for impact analysis
    if 'price' not in df.columns:
        st.error("Price column not found in the dataset. Cannot proceed with price impact analysis.")
        return
    
    # Ensure price column is numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])
    
    if len(df) == 0:
        st.error("No valid price data available after cleaning. Cannot proceed with analysis.")
        return
    
    # Convert all potential numeric columns to numeric type
    for col in df.columns:
        if col not in ['zpid', 'id', 'streetAddress', 'description', 'city', 'state', 'zipcode', 'homeType', 'neighborhood']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Get numeric features for analysis (after conversion)
    numeric_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) 
                       and col != 'price' 
                       and col != 'zpid'
                       and not col.startswith('dataset_')
                       and df[col].notna().sum() > len(df) * 0.1  # At least 10% non-null values
                       and df[col].nunique() > 1]
    
    if not numeric_features:
        st.error("No valid numeric features found for analysis after data cleaning.")
        return
    
    # Get categorical features for analysis
    categorical_features = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
                           and col != 'zpid'
                           and df[col].nunique() > 1
                           and df[col].nunique() < 20]  # Limit to reasonable number of categories
    
    # Filter section
    with st.expander("Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Price range filter
            price_min = int(df['price'].min())
            price_max = int(df['price'].max())
            price_range = st.slider(
                "Price Range ($)",
                min_value=price_min,
                max_value=price_max,
                value=(price_min, min(price_max, price_min + 1000000)),
                step=10000
            )
            
        with col2:
            # Home type filter if available
            if 'homeType' in df.columns:
                home_types = df['homeType'].dropna().unique()
                selected_home_types = st.multiselect(
                    "Home Types",
                    options=home_types,
                    default=home_types
                )
                if selected_home_types:
                    df = df[df['homeType'].isin(selected_home_types)]
    
    # Apply price filter
    df = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])]
    
    if len(df) == 0:
        st.warning("No properties match your filters. Please adjust the criteria.")
        return
    
    # Main dashboard tabs
    selected_tabs = st.tabs(["Price Correlations", "Feature Analysis", "Geographic Analysis"])
    
    # Tab 1: Price Correlations
    with selected_tabs[0]:
        st.markdown("### Feature Correlations with Price")
        
        if not numeric_features:
            st.warning("No numeric features available for correlation analysis.")
        else:
            # Correlation with price
            # Choose features with sufficient non-null values
            valid_features = [col for col in numeric_features if df[col].notna().sum() > len(df) * 0.3]
            valid_features = valid_features + ['price']
            
            if len(valid_features) <= 1:
                st.warning("Not enough valid numeric features for correlation analysis.")
            else:
                try:
                    # Calculate correlations, handling potential errors
                    correlation_df = df[valid_features].copy()
                    corr_df = correlation_df.corr()['price'].sort_values(ascending=False).reset_index()
                    corr_df.columns = ['Feature', 'Correlation with Price']
                    
                    # Remove price itself
                    corr_df = corr_df[corr_df['Feature'] != 'price']
                    
                    if len(corr_df) > 0:
                        # Plot correlations
                        fig = px.bar(
                            corr_df,
                            x='Feature',
                            y='Correlation with Price',
                            title="Features Correlation with Price",
                            labels={
                                'Feature': 'Property Feature',
                                'Correlation with Price': 'Correlation Coefficient'
                            },
                            color='Correlation with Price',
                            color_continuous_scale=px.colors.diverging.RdBu_r,
                            range_color=[-1, 1]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Correlation heatmap for top features
                        st.markdown("### Correlation Heatmap")
                        top_features = corr_df.head(min(8, len(corr_df)))['Feature'].tolist() + ['price']
                        
                        if len(top_features) > 2:  # Need at least 2 features for a meaningful heatmap
                            corr_matrix = df[top_features].corr()
                            
                            fig = px.imshow(
                                corr_matrix,
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale=px.colors.diverging.RdBu_r,
                                title="Correlation Matrix of Top Features"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No significant correlations found.")
                except Exception as e:
                    st.error(f"Error calculating correlations: {str(e)}")
    
    # Tab 2: Feature Analysis
    with selected_tabs[1]:
        st.markdown("### Feature-Price Relationships")
        
        # Feature selector
        feature_options = numeric_features.copy()
        if len(feature_options) == 0:
            st.warning("No numeric features available for analysis.")
        else:
            selected_feature = st.selectbox(
                "Select Feature to Analyze",
                options=feature_options
            )
            
            try:
                # Scatter plot with trend line
                fig = px.scatter(
                    df,
                    x=selected_feature,
                    y='price',
                    title=f"{selected_feature} vs. Price",
                    labels={selected_feature: selected_feature.replace('_', ' ').title(), 'price': 'Price ($)'},
                    opacity=0.7,
                    trendline="ols",
                    trendline_color_override="red"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Basic statistics
                stats_df = df[[selected_feature, 'price']].dropna().describe().T
                st.markdown("### Feature Statistics")
                st.dataframe(stats_df.style.format("{:.2f}"))
                
                # Calculate simple price impact using quartiles
                try:
                    q1 = df[selected_feature].quantile(0.25)
                    q3 = df[selected_feature].quantile(0.75)
                    
                    low_group = df[(df[selected_feature] <= q1) & df[selected_feature].notna()]
                    high_group = df[(df[selected_feature] >= q3) & df[selected_feature].notna()]
                    
                    if len(low_group) > 5 and len(high_group) > 5:
                        low_price = low_group['price'].mean()
                        high_price = high_group['price'].mean()
                        price_diff = high_price - low_price
                        pct_diff = (price_diff / low_price) * 100 if low_price > 0 else 0
                        
                        st.markdown("### Price Impact Analysis")
                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"Bottom 25% {selected_feature}", f"${low_price:,.0f}")
                        col2.metric(f"Top 25% {selected_feature}", f"${high_price:,.0f}")
                        col3.metric("Price Difference", f"${price_diff:,.0f}", f"{pct_diff:.1f}%")
                except Exception as e:
                    pass  # Silently skip if this analysis fails
                
                # Bin analysis - group by feature bins and show average price
                if len(df) > 20:  # Only if we have enough data
                    st.markdown("### Average Price by Feature Bins")
                    
                    # Determine number of bins based on data size and variability
                    nunique = df[selected_feature].nunique()
                    if nunique <= 10:  # If feature has few unique values, use them as is
                        bin_df = df.groupby(selected_feature).agg(
                            avg_price=('price', 'mean'),
                            count=('price', 'count')
                        ).reset_index()
                        bin_df = bin_df.sort_values(selected_feature)
                    else:  # Otherwise create bins
                        # Create 5-10 bins based on data size
                        num_bins = min(max(5, len(df) // 100), 10)
                        
                        df['bins'] = pd.cut(
                            df[selected_feature], 
                            bins=num_bins,
                            precision=0
                        )
                        
                        bin_df = df.groupby('bins').agg(
                            avg_price=('price', 'mean'),
                            count=('price', 'count')
                        ).reset_index()
                        
                        # Convert bin ranges to strings for better display
                        bin_df['bins'] = bin_df['bins'].astype(str)
                    
                    # Plot the binned data
                    fig = px.bar(
                        bin_df,
                        x='bins' if 'bins' in bin_df.columns else selected_feature,
                        y='avg_price',
                        title=f"Average Price by {selected_feature.replace('_', ' ').title()} Bins",
                        labels={
                            'bins': selected_feature.replace('_', ' ').title() if 'bins' in bin_df.columns else selected_feature,
                            'avg_price': 'Average Price ($)',
                        },
                        text=bin_df['count'],
                        color='avg_price',
                        height=400
                    )
                    
                    # Update hover template to show count
                    fig.update_traces(
                        hovertemplate='%{x}<br>Average Price: $%{y:,.0f}<br>Count: %{text}'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error analyzing feature: {str(e)}")
    
    # Tab 3: Geographic Analysis
    with selected_tabs[2]:
        st.markdown("### Geographic Impact on Prices")
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            st.warning("Location data (latitude/longitude) not found in the dataset.")
        else:
            # Check if we have sufficient location data
            location_df = df.dropna(subset=['latitude', 'longitude', 'price'])
            
            if len(location_df) < 20:
                st.warning("Not enough properties with location data for geographic analysis.")
            else:
                # Scatter plot on map
                fig = px.scatter_mapbox(
                    location_df,
                    lat='latitude',
                    lon='longitude',
                    color='price',
                    size='price',
                    color_continuous_scale=px.colors.sequential.Plasma,
                    size_max=15,
                    zoom=10,
                    mapbox_style="open-street-map",
                    title="Property Prices by Location",
                    opacity=0.7
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Group by location (using city if available)
                if 'city' in df.columns and df['city'].notna().sum() > 0.5 * len(df):
                    st.markdown("### Price by City")
                    
                    city_df = df.groupby('city').agg(
                        avg_price=('price', 'mean'),
                        median_price=('price', 'median'),
                        count=('price', 'count')
                    ).reset_index()
                    
                    # Filter to cities with enough properties
                    city_df = city_df[city_df['count'] >= 5].sort_values('avg_price', ascending=False)
                    
                    if len(city_df) > 0:
                        fig = px.bar(
                            city_df,
                            x='city',
                            y='avg_price',
                            title="Average Price by City",
                            labels={
                                'city': 'City',
                                'avg_price': 'Average Price ($)'
                            },
                            text='count',
                            color='avg_price'
                        )
                        
                        fig.update_traces(
                            hovertemplate='%{x}<br>Average Price: $%{y:,.0f}<br>Count: %{text}'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough cities with sufficient data for comparison.")
                
                # Try to use zip code if available and city isn't
                elif 'zipcode' in df.columns and df['zipcode'].notna().sum() > 0.5 * len(df):
                    st.markdown("### Price by ZIP Code")
                    
                    zip_df = df.groupby('zipcode').agg(
                        avg_price=('price', 'mean'),
                        median_price=('price', 'median'),
                        count=('price', 'count')
                    ).reset_index()
                    
                    # Filter to zip codes with enough properties
                    zip_df = zip_df[zip_df['count'] >= 5].sort_values('avg_price', ascending=False)
                    
                    if len(zip_df) > 0:
                        fig = px.bar(
                            zip_df,
                            x='zipcode',
                            y='avg_price',
                            title="Average Price by ZIP Code",
                            labels={
                                'zipcode': 'ZIP Code',
                                'avg_price': 'Average Price ($)'
                            },
                            text='count',
                            color='avg_price'
                        )
                        
                        fig.update_traces(
                            hovertemplate='%{x}<br>Average Price: $%{y:,.0f}<br>Count: %{text}'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough ZIP codes with sufficient data for comparison.")
                else:
                    st.info("No city or ZIP code information found for location-based comparison.") 