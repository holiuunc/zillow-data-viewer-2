import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def display_property_comparison(df_2023, df_2025, df_combined):
    """
    Display property comparison dashboard
    
    Parameters:
    -----------
    df_2023 : DataFrame or None
        2023 property data
    df_2025 : DataFrame or None
        2025 property data
    df_combined : DataFrame or None
        Combined property data
    """
    st.header("🏠 Property Comparison")
    st.markdown("Analyze property attributes and how they relate to price")
    
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
    
    # Check for price column as it's required for most of the analysis
    if 'price' not in df.columns:
        st.error("Price column not found in the dataset. Cannot proceed with price-based analysis.")
        return
    
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
                value=(price_min, min(price_max, 1000000)),
                step=10000
            )
            
            # Bedroom filter if available
            if 'bedrooms' in df.columns:
                bed_min = int(df['bedrooms'].min())
                bed_max = int(df['bedrooms'].max())
                bed_range = st.slider(
                    "Bedrooms",
                    min_value=bed_min,
                    max_value=bed_max,
                    value=(bed_min, bed_max)
                )
                df = df[(df['bedrooms'] >= bed_range[0]) & (df['bedrooms'] <= bed_range[1])]
        
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
            
            # Bathroom filter if available
            if 'bathrooms' in df.columns:
                bath_min = int(df['bathrooms'].min())
                bath_max = int(df['bathrooms'].max())
                bath_range = st.slider(
                    "Bathrooms",
                    min_value=bath_min,
                    max_value=bath_max,
                    value=(bath_min, bath_max)
                )
                df = df[(df['bathrooms'] >= bath_range[0]) & (df['bathrooms'] <= bath_range[1])]
    
    # Apply price filter
    df = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])]
    
    if len(df) == 0:
        st.warning("No properties match your filters. Please adjust the criteria.")
        return
    
    # Get numeric columns for scatter plot options
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col != 'price' and not col.startswith('dataset_'):
            numeric_cols.append(col)
    
    # Get categorical columns for coloring options
    categorical_cols = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) or col == 'homeType' or col == 'dataset_year':
            categorical_cols.append(col)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Property Attributes", "Price Factors", "Scatter Analysis", "Summary Stats"])
    
    # Tab 1: Property Attributes
    with tab1:
        st.markdown("### Property Attribute Distribution")
        
        # Select attribute to analyze
        attribute_options = ['bedrooms', 'bathrooms', 'homeType', 'yearBuilt', 'livingArea']
        attribute_options = [col for col in attribute_options if col in df.columns]
        
        if not attribute_options:
            st.warning("No property attribute columns found in the dataset.")
        else:
            selected_attribute = st.selectbox(
                "Select Attribute",
                options=attribute_options,
                index=0
            )
            
            if selected_attribute:
                # Distribution of the selected attribute
                if pd.api.types.is_numeric_dtype(df[selected_attribute]):
                    # For numeric attributes like bedrooms, bathrooms, year built, living area
                    fig = px.histogram(
                        df,
                        x=selected_attribute,
                        title=f"Distribution of {selected_attribute}",
                        labels={selected_attribute: selected_attribute.replace('_', ' ').title()},
                        color="dataset_year" if "dataset_year" in df.columns else None,
                        nbins=30
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Average price by attribute value
                    avg_price = df.groupby(selected_attribute)['price'].mean().reset_index()
                    fig = px.bar(
                        avg_price,
                        x=selected_attribute,
                        y='price',
                        title=f"Average Price by {selected_attribute}",
                        labels={
                            selected_attribute: selected_attribute.replace('_', ' ').title(),
                            'price': 'Average Price ($)'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # For categorical attributes like homeType
                    value_counts = df[selected_attribute].value_counts().reset_index()
                    value_counts.columns = [selected_attribute, 'count']
                    
                    fig = px.bar(
                        value_counts,
                        x=selected_attribute,
                        y='count',
                        title=f"Distribution of {selected_attribute}",
                        labels={
                            selected_attribute: selected_attribute.replace('_', ' ').title(),
                            'count': 'Number of Properties'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Average price by attribute value
                    avg_price = df.groupby(selected_attribute)['price'].mean().reset_index()
                    avg_price = avg_price.sort_values('price')
                    
                    fig = px.bar(
                        avg_price,
                        x=selected_attribute,
                        y='price',
                        title=f"Average Price by {selected_attribute}",
                        labels={
                            selected_attribute: selected_attribute.replace('_', ' ').title(),
                            'price': 'Average Price ($)'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Price Factors
    with tab2:
        st.markdown("### Price Factors Analysis")
        
        # Create calculated feature: price per square foot if livingArea exists
        if 'livingArea' in df.columns:
            # Only calculate for rows with valid livingArea
            mask = (df['livingArea'] > 0) & df['livingArea'].notna()
            if not df.loc[mask, 'livingArea'].empty:
                df.loc[mask, 'pricePerSqFt'] = df.loc[mask, 'price'] / df.loc[mask, 'livingArea']
        
        # Display price per square foot distribution if available
        if 'pricePerSqFt' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Statistics for price per square foot
                ppsf_stats = df['pricePerSqFt'].describe()
                st.metric("Median Price/SqFt", f"${ppsf_stats['50%']:.2f}")
                st.metric("Average Price/SqFt", f"${ppsf_stats['mean']:.2f}")
            
            with col2:
                # Price per square foot distribution
                fig = px.histogram(
                    df,
                    x='pricePerSqFt',
                    nbins=30,
                    title="Price per Square Foot Distribution",
                    labels={'pricePerSqFt': 'Price per Square Foot ($)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Price correlations
        if numeric_cols:
            # Calculate correlations with price
            corr_cols = numeric_cols + ['price']
            if 'pricePerSqFt' in df.columns:
                corr_cols.append('pricePerSqFt')
            
            corr_df = df[corr_cols].corr()['price'].sort_values(ascending=False).reset_index()
            corr_df.columns = ['Feature', 'Correlation with Price']
            
            # Remove price itself from the list
            corr_df = corr_df[corr_df['Feature'] != 'price']
            
            fig = px.bar(
                corr_df,
                x='Feature',
                y='Correlation with Price',
                title="Features Correlation with Price",
                labels={
                    'Feature': 'Property Feature',
                    'Correlation with Price': 'Correlation Coefficient'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance visualization
            st.markdown("### Feature Impact on Price")
            st.info("Select a feature to see how it impacts property prices")
            
            if numeric_cols:
                # Select feature to analyze
                selected_feature = st.selectbox(
                    "Select Feature",
                    options=numeric_cols,
                    index=0 if len(numeric_cols) > 0 else None
                )
                
                if selected_feature:
                    # Create a scatter plot of the feature vs price
                    fig = px.scatter(
                        df,
                        x=selected_feature,
                        y='price',
                        title=f"{selected_feature.replace('_', ' ').title()} vs Price",
                        labels={
                            selected_feature: selected_feature.replace('_', ' ').title(),
                            'price': 'Price ($)'
                        },
                        color='homeType' if 'homeType' in df.columns else None,
                        trendline="ols",
                        trendline_color_override="red"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Scatter Analysis
    with tab3:
        st.markdown("### Property Scatter Analysis")
        
        # Allow the user to customize a scatter plot
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox(
                "X-Axis",
                options=numeric_cols,
                index=0 if len(numeric_cols) > 0 else None
            )
            
            color_by = st.selectbox(
                "Color By",
                options=["None"] + categorical_cols,
                index=0
            )
        
        with col2:
            y_axis = st.selectbox(
                "Y-Axis",
                options=['price'] + numeric_cols,
                index=0
            )
            
            size_by = st.selectbox(
                "Size By",
                options=["None"] + numeric_cols,
                index=0
            )
        
        # Create the scatter plot
        if x_axis and y_axis:
            scatter_kwargs = {
                'x': x_axis,
                'y': y_axis,
                'title': f"{x_axis.replace('_', ' ').title()} vs {y_axis.replace('_', ' ').title()}",
                'labels': {
                    x_axis: x_axis.replace('_', ' ').title(),
                    y_axis: y_axis.replace('_', ' ').title()
                },
                'hover_data': ['price'] if y_axis != 'price' else None,
            }
            
            if color_by != "None":
                scatter_kwargs['color'] = color_by
            
            if size_by != "None":
                scatter_kwargs['size'] = size_by
            
            fig = px.scatter(df, **scatter_kwargs)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Summary Stats
    with tab4:
        st.markdown("### Summary Statistics")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Properties", f"{len(df):,}")
            
            if 'homeType' in df.columns:
                most_common_type = df['homeType'].value_counts().index[0]
                st.metric("Most Common Type", most_common_type)
        
        with col2:
            st.metric("Median Price", f"${df['price'].median():,.0f}")
            
            if 'yearBuilt' in df.columns:
                median_year = int(df['yearBuilt'].median())
                st.metric("Median Year Built", f"{median_year}")
        
        with col3:
            st.metric("Average Price", f"${df['price'].mean():,.0f}")
            
            if 'livingArea' in df.columns:
                median_area = int(df['livingArea'].median())
                st.metric("Median Size", f"{median_area} sqft")
        
        # Summary tables
        if 'homeType' in df.columns:
            st.markdown("### Property Types")
            
            # Property type counts
            type_counts = df['homeType'].value_counts().reset_index()
            type_counts.columns = ['Home Type', 'Count']
            type_counts['Percentage'] = type_counts['Count'] / len(df) * 100
            
            # Property type price stats
            type_prices = df.groupby('homeType')['price'].agg(['mean', 'median', 'min', 'max']).reset_index()
            type_prices.columns = ['Home Type', 'Mean Price', 'Median Price', 'Min Price', 'Max Price']
            
            # Merge counts and prices
            type_summary = type_counts.merge(type_prices, on='Home Type')
            
            # Format prices
            for col in ['Mean Price', 'Median Price', 'Min Price', 'Max Price']:
                type_summary[col] = type_summary[col].map('${:,.0f}'.format)
            
            # Format percentage
            type_summary['Percentage'] = type_summary['Percentage'].map('{:.1f}%'.format)
            
            st.dataframe(type_summary, use_container_width=True)
        
        # Location summary if available
        location_cols = ['city', 'zipcode', 'neighborhood']
        available_loc_cols = [col for col in location_cols if col in df.columns]
        
        if available_loc_cols:
            selected_loc = st.selectbox(
                "Location Grouping",
                options=available_loc_cols,
                index=0
            )
            
            st.markdown(f"### Properties by {selected_loc.title()}")
            
            # Location counts
            loc_counts = df[selected_loc].value_counts().reset_index()
            loc_counts.columns = [selected_loc.title(), 'Count']
            
            # Location price stats
            loc_prices = df.groupby(selected_loc)['price'].agg(['mean', 'median']).reset_index()
            loc_prices.columns = [selected_loc.title(), 'Mean Price', 'Median Price']
            
            # Merge counts and prices
            loc_summary = loc_counts.merge(loc_prices, on=selected_loc.title())
            
            # Format prices
            for col in ['Mean Price', 'Median Price']:
                loc_summary[col] = loc_summary[col].map('${:,.0f}'.format)
            
            # Sort by count
            loc_summary = loc_summary.sort_values('Count', ascending=False)
            
            st.dataframe(loc_summary, use_container_width=True) 