import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def display_market_segments(df_2023, df_2025, df_combined):
    """
    Display market segment explorer dashboard
    
    Parameters:
    -----------
    df_2023 : DataFrame or None
        2023 property data
    df_2025 : DataFrame or None
        2025 property data
    df_combined : DataFrame or None
        Combined property data
    """
    st.header("🔍 Market Segment Explorer")
    st.markdown("Explore different segments of the housing market")
    
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
    if 'price' not in df.columns:
        st.error("Price column not found in the dataset. Cannot proceed with market segmentation.")
        return
    
    # Convert price to numeric, coercing errors to NaN
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Drop rows with NaN price values
    df = df.dropna(subset=['price'])
    
    if len(df) == 0:
        st.error("No valid price data available after cleaning. Cannot proceed.")
        return
    
    # Main segmentation section
    st.markdown("## Define Market Segments")
    
    # Option to select segmentation method
    segment_method = st.radio(
        "Segmentation Method",
        options=["By Price Range", "By Property Type", "By Price & Size", "Custom Segments"],
        index=0
    )
    
    # By Price Range
    if segment_method == "By Price Range":
        # Define price segments
        price_min = float(df['price'].min())
        price_max = float(df['price'].max())
        
        # Allow custom price range segmentation
        num_segments = st.slider("Number of Price Segments", 2, 10, 5)
        
        # Create price ranges
        if num_segments > 1:
            step = (price_max - price_min) / num_segments
            price_ranges = [(price_min + i * step, price_min + (i + 1) * step) for i in range(num_segments)]
            
            # Create segment labels
            segment_labels = [f"${int(low):,} - ${int(high):,}" for low, high in price_ranges]
            
            # Create a new column for price segments
            df['price_segment'] = pd.cut(
                df['price'], 
                bins=[price_min] + [pr[1] for pr in price_ranges], 
                labels=segment_labels,
                include_lowest=True
            )
        else:
            st.warning("Please select at least 2 segments.")
            return
        
        segment_col = 'price_segment'
    
    # By Property Type
    elif segment_method == "By Property Type":
        if 'homeType' not in df.columns:
            st.error("The 'homeType' column is not present in the dataset. Please select a different segmentation method.")
            return
        
        home_types = df['homeType'].dropna().unique()
        
        if len(home_types) == 0:
            st.error("No property types found in the dataset.")
            return
        
        segment_col = 'homeType'
        
        # Show the available segments
        st.markdown("### Property Type Segments")
        type_counts = df[segment_col].value_counts().reset_index()
        type_counts.columns = ['Property Type', 'Count']
        
        # Display as a bar chart
        fig = px.bar(
            type_counts,
            x='Property Type',
            y='Count',
            title="Property Types in Dataset",
            color='Property Type'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # By Price & Size
    elif segment_method == "By Price & Size":
        if 'livingArea' not in df.columns:
            st.error("The 'livingArea' column is not present in the dataset. Please select a different segmentation method.")
            return
        
        # Ensure livingArea is numeric
        df['livingArea'] = pd.to_numeric(df['livingArea'], errors='coerce')
        df = df.dropna(subset=['livingArea'])
        
        if len(df) == 0:
            st.error("No valid livingArea data available after cleaning. Cannot proceed.")
            return
        
        # Define price thresholds
        price_threshold = st.slider(
            "Price Threshold ($)",
            min_value=int(df['price'].min()),
            max_value=int(df['price'].max()),
            value=int(df['price'].median()),
            step=10000
        )
        
        # Define size thresholds
        size_threshold = st.slider(
            "Size Threshold (sq ft)",
            min_value=int(df['livingArea'].min()),
            max_value=int(df['livingArea'].max()),
            value=int(df['livingArea'].median()),
            step=100
        )
        
        # Create segment column
        def get_price_size_segment(row):
            if pd.isna(row['price']) or pd.isna(row['livingArea']):
                return "Unknown"
            elif row['price'] > price_threshold and row['livingArea'] > size_threshold:
                return "Luxury (High Price, Large Size)"
            elif row['price'] > price_threshold and row['livingArea'] <= size_threshold:
                return "Premium (High Price, Smaller Size)"
            elif row['price'] <= price_threshold and row['livingArea'] > size_threshold:
                return "Value (Lower Price, Large Size)"
            else:
                return "Economy (Lower Price, Smaller Size)"
        
        df['price_size_segment'] = df.apply(get_price_size_segment, axis=1)
        segment_col = 'price_size_segment'
        
        # Show the segmentation on a scatter plot
        fig = px.scatter(
            df,
            x='livingArea',
            y='price',
            color=segment_col,
            title="Property Segments by Price and Size",
            labels={
                'livingArea': 'Living Area (sq ft)',
                'price': 'Price ($)'
            },
            opacity=0.6
        )
        
        # Add threshold lines
        fig.add_hline(y=price_threshold, line_dash="dash", line_color="gray")
        fig.add_vline(x=size_threshold, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Custom Segments
    elif segment_method == "Custom Segments":
        st.markdown("### Define Custom Segments")
        
        # Get available numeric columns for segmentation
        numeric_cols = []
        for col in df.columns:
            try:
                # Only check the first few values to avoid computational overhead
                df[col].head().astype(float)
                # If it succeeds, it's potentially numeric
                if col not in ['zpid', 'id', 'dataset_year'] and df[col].nunique() > 1:
                    numeric_cols.append(col)
            except:
                continue
        
        if not numeric_cols:
            st.error("No numeric columns found for custom segmentation.")
            return
        
        # Select primary and secondary segmentation features
        primary_feature = st.selectbox(
            "Primary Segmentation Feature",
            options=numeric_cols,
            index=numeric_cols.index('price') if 'price' in numeric_cols else 0
        )
        
        # Ensure selected column is numeric
        df[primary_feature] = pd.to_numeric(df[primary_feature], errors='coerce')
        
        # Define threshold for primary feature
        primary_min = float(df[primary_feature].min())
        primary_max = float(df[primary_feature].max())
        primary_threshold = st.slider(
            f"{primary_feature.replace('_', ' ').title()} Threshold",
            min_value=primary_min,
            max_value=primary_max,
            value=(primary_min + primary_max) / 2
        )
        
        # Option for secondary feature
        use_secondary = st.checkbox("Add Secondary Segmentation Feature")
        
        if use_secondary:
            # Filter out the primary feature from options
            secondary_options = [col for col in numeric_cols if col != primary_feature]
            
            if not secondary_options:
                st.warning("No other numeric columns available for secondary segmentation.")
                use_secondary = False
            else:
                secondary_feature = st.selectbox(
                    "Secondary Segmentation Feature",
                    options=secondary_options
                )
                
                # Ensure selected column is numeric
                df[secondary_feature] = pd.to_numeric(df[secondary_feature], errors='coerce')
                
                # Define threshold for secondary feature
                secondary_min = float(df[secondary_feature].min())
                secondary_max = float(df[secondary_feature].max())
                secondary_threshold = st.slider(
                    f"{secondary_feature.replace('_', ' ').title()} Threshold",
                    min_value=secondary_min,
                    max_value=secondary_max,
                    value=(secondary_min + secondary_max) / 2
                )
        
        # Create segment definitions
        segment_definitions = []
        
        for i in range(num_custom_segments):
            st.markdown(f"#### Segment {i+1}")
            
            segment_name = st.text_input(f"Segment {i+1} Name", value=f"Segment {i+1}")
            
            conditions = []
            
            for col in selected_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # For numeric columns, create range filters
                    col_min = float(df[col].min())
                    col_max = float(df[col].max())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        min_val = st.number_input(
                            f"Min {col}",
                            min_value=col_min,
                            max_value=col_max,
                            value=col_min,
                            key=f"min_{col}_{i}"
                        )
                    
                    with col2:
                        max_val = st.number_input(
                            f"Max {col}",
                            min_value=col_min,
                            max_value=col_max,
                            value=col_max,
                            key=f"max_{col}_{i}"
                        )
                    
                    conditions.append(f"(df['{col}'] >= {min_val} and df['{col}'] <= {max_val})")
                else:
                    # For categorical columns, create multi-select
                    cat_values = df[col].dropna().unique()
                    selected_values = st.multiselect(
                        f"Select {col} values",
                        options=cat_values,
                        default=cat_values,
                        key=f"cat_{col}_{i}"
                    )
                    
                    if selected_values:
                        conditions.append(f"df['{col}'].isin({selected_values})")
            
            segment_definitions.append({
                'name': segment_name,
                'condition': ' and '.join(conditions)
            })
        
        # Create segments based on conditions
        df['custom_segment'] = "Other"
        
        for segment in segment_definitions:
            # Create each segment with its condition
            mask = eval(segment['condition'])
            df.loc[mask, 'custom_segment'] = segment['name']
        
        segment_col = 'custom_segment'
    
    # Display segment statistics
    st.markdown("## Market Segment Analysis")
    
    # General segment overview
    segment_counts = df[segment_col].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    segment_counts['Percentage'] = segment_counts['Count'] / len(df) * 100
    
    # Calculate price statistics by segment
    segment_stats = df.groupby(segment_col)['price'].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
    segment_stats.columns = ['Segment', 'Mean Price', 'Median Price', 'Min Price', 'Max Price', 'Count']
    
    # Display segment statistics
    st.markdown("### Segment Overview")
    
    # Display as a bar chart
    fig = px.bar(
        segment_counts,
        x='Segment',
        y='Count',
        title="Property Counts by Segment",
        color='Segment',
        text=segment_counts['Percentage'].round(1).astype(str) + '%'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display segment price statistics
    st.markdown("### Price Statistics by Segment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Median price by segment
        fig = px.bar(
            segment_stats,
            x='Segment',
            y='Median Price',
            title="Median Price by Segment",
            color='Segment',
            labels={'Median Price': 'Median Price ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price distribution by segment
        fig = px.box(
            df,
            x=segment_col,
            y='price',
            title="Price Distribution by Segment",
            color=segment_col,
            labels={segment_col: 'Segment', 'price': 'Price ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional segment characteristics
    st.markdown("### Segment Characteristics")
    
    # Property characteristics by segment
    property_features = ['bedrooms', 'bathrooms', 'livingArea', 'yearBuilt']
    available_features = [f for f in property_features if f in df.columns]
    
    if available_features:
        selected_feature = st.selectbox(
            "Select Feature to Compare",
            options=available_features
        )
        
        if selected_feature:
            # Calculate average feature value by segment
            feature_stats = df.groupby(segment_col)[selected_feature].median().reset_index()
            
            fig = px.bar(
                feature_stats,
                x='Segment',
                y=selected_feature,
                title=f"Median {selected_feature.replace('_', ' ').title()} by Segment",
                color='Segment',
                labels={
                    selected_feature: selected_feature.replace('_', ' ').title()
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution by segment
            fig = px.box(
                df,
                x=segment_col,
                y=selected_feature,
                title=f"{selected_feature.replace('_', ' ').title()} Distribution by Segment",
                color=segment_col,
                labels={
                    segment_col: 'Segment',
                    selected_feature: selected_feature.replace('_', ' ').title()
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Geographic distribution if we have location data
    location_cols = ['city', 'zipcode', 'neighborhood']
    available_loc_cols = [col for col in location_cols if col in df.columns]
    
    if available_loc_cols:
        st.markdown("### Geographic Distribution of Segments")
        
        selected_loc = st.selectbox(
            "Select Location Level",
            options=available_loc_cols
        )
        
        if selected_loc:
            # Create a crosstab of segment by location
            loc_segment = pd.crosstab(
                df[selected_loc], 
                df[segment_col],
                normalize='index'
            ).reset_index()
            
            # Select top locations by count
            top_locs = df[selected_loc].value_counts().head(10).index.tolist()
            loc_segment_filtered = loc_segment[loc_segment[selected_loc].isin(top_locs)]
            
            # Plot stacked bar chart
            fig = px.bar(
                loc_segment_filtered,
                x=selected_loc,
                y=loc_segment_filtered.columns[1:].tolist(),
                title=f"Segment Distribution by {selected_loc.title()} (Top 10)",
                labels={
                    'value': 'Percentage',
                    'variable': 'Segment'
                },
                barmode='stack'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Time trends for combined dataset
    if selected_dataset == "Combined Dataset" and "dataset_year" in df.columns:
        st.markdown("### Segment Trends (2023 vs 2025)")
        
        # Calculate segment distribution by year
        year_segment = pd.crosstab(
            df["dataset_year"], 
            df[segment_col],
            normalize='index'
        ).reset_index()
        
        # Plot comparison
        fig = px.bar(
            year_segment,
            x="dataset_year",
            y=year_segment.columns[1:].tolist(),
            title="Segment Distribution by Year",
            labels={
                'value': 'Percentage',
                'variable': 'Segment'
            },
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate price changes by segment
        price_change = df.groupby(['dataset_year', segment_col])['price'].median().reset_index()
        price_change_pivot = price_change.pivot(index=segment_col, columns='dataset_year', values='price')
        
        if 2023 in price_change_pivot.columns and 2025 in price_change_pivot.columns:
            price_change_pivot['change'] = price_change_pivot[2025] - price_change_pivot[2023]
            price_change_pivot['pct_change'] = (price_change_pivot['change'] / price_change_pivot[2023]) * 100
            
            price_change_data = price_change_pivot.reset_index()
            
            fig = px.bar(
                price_change_data,
                x=segment_col,
                y='pct_change',
                title="Price Change by Segment (2023-2025)",
                color='pct_change',
                labels={
                    segment_col: 'Segment',
                    'pct_change': 'Price Change (%)'
                },
                color_continuous_scale=px.colors.sequential.Blues
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Segment summary
    st.markdown("### Segment Summary")
    
    # Format segment stats for display
    summary_table = segment_stats.copy()
    
    # Format price columns
    for col in ['Mean Price', 'Median Price', 'Min Price', 'Max Price']:
        summary_table[col] = summary_table[col].map('${:,.0f}'.format)
    
    # Add percentage column
    summary_table['Market Share'] = (summary_table['Count'] / summary_table['Count'].sum() * 100).map('{:.1f}%'.format)
    
    # Display summary table
    st.dataframe(summary_table, use_container_width=True) 