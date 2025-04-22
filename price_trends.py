import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_price_trends(df_2023, df_2025, df_combined):
    """
    Display price trends dashboard
    
    Parameters:
    -----------
    df_2023 : DataFrame or None
        2023 property data
    df_2025 : DataFrame or None
        2025 property data
    df_combined : DataFrame or None
        Combined property data
    """
    st.header("📈 Price Trends Dashboard")
    st.markdown("Analyze price changes by location and property type")
    
    # Check for required datasets
    if df_combined is None and (df_2023 is None or df_2025 is None):
        st.warning("This visualization requires data from both 2023 and 2025. Please load both datasets.")
        return
    
    # Prepare the data
    if df_combined is not None:
        df = df_combined
        
        # Make sure dataset_year is available
        if 'dataset_year' not in df.columns:
            st.error("The combined dataset is missing the 'dataset_year' column needed for comparison.")
            return
    else:
        # Create a combined dataframe
        if 'dataset_year' not in df_2023.columns:
            df_2023 = df_2023.copy()
            df_2023['dataset_year'] = 2023
        
        if 'dataset_year' not in df_2025.columns:
            df_2025 = df_2025.copy()
            df_2025['dataset_year'] = 2025
        
        # Find common columns
        common_cols = list(set(df_2023.columns) & set(df_2025.columns))
        
        # Combine datasets
        df = pd.concat([
            df_2023[common_cols], 
            df_2025[common_cols]
        ], ignore_index=True)
    
    # Check for required columns
    required_cols = ['price', 'dataset_year']
    optional_geo_cols = ['city', 'zipcode', 'neighborhood']
    
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        st.error(f"Missing required columns for price trends: {', '.join(missing_cols)}")
        return
    
    # Check if we have any geographic columns
    available_geo_cols = [col for col in optional_geo_cols if col in df.columns]
    if not available_geo_cols:
        st.warning("No geographic columns (city, zipcode, neighborhood) found for location-based analysis.")
    
    # Convert dataset_year to string for better display
    df['dataset_year'] = df['dataset_year'].astype(str)
    
    # Dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### Filters")
        
        # Property type filter
        if 'homeType' in df.columns:
            home_types = df['homeType'].dropna().unique()
            selected_home_types = st.multiselect(
                "Home Types",
                options=home_types,
                default=home_types
            )
            if selected_home_types:
                df = df[df['homeType'].isin(selected_home_types)]
        
        # Geographic selection
        selected_geo_col = None
        geo_level = None
        
        if available_geo_cols:
            geo_level = st.selectbox(
                "Geographic Level",
                options=available_geo_cols
            )
            
            if geo_level:
                geo_values = df[geo_level].dropna().unique()
                selected_geo_values = st.multiselect(
                    f"Select {geo_level.title()}s",
                    options=geo_values,
                    default=list(geo_values)[:5] if len(geo_values) > 5 else geo_values
                )
                
                if selected_geo_values:
                    df = df[df[geo_level].isin(selected_geo_values)]
                    selected_geo_col = geo_level
        
        # Property size filter
        if 'livingArea' in df.columns:
            min_area = float(df['livingArea'].min())
            max_area = float(df['livingArea'].max())
            living_area_range = st.slider(
                "Living Area (sq ft)",
                min_value=min_area,
                max_value=max_area,
                value=(min_area, max_area),
                step=100.0
            )
            df = df[(df['livingArea'] >= living_area_range[0]) & 
                    (df['livingArea'] <= living_area_range[1])]
        
        # Price normalization option
        normalize_option = st.radio(
            "Price Metric",
            ["Total Price", "Price per sq ft"],
            index=0
        )
        
        if normalize_option == "Price per sq ft" and 'livingArea' in df.columns:
            # Ensure we have valid living area values
            mask = (df['livingArea'] > 0) & df['livingArea'].notna() & df['price'].notna()
            if 'pricePerSqFt' not in df.columns:
                df.loc[mask, 'pricePerSqFt'] = df.loc[mask, 'price'] / df.loc[mask, 'livingArea']
            price_col = 'pricePerSqFt'
        else:
            price_col = 'price'
    
    with col1:
        # Calculate price statistics by year
        price_by_year = df.groupby('dataset_year')[price_col].agg(['mean', 'median', 'min', 'max', 'count'])
        
        # Display metrics
        st.markdown("### Price Change Overview")
        
        # Calculate year-over-year changes
        if '2023' in price_by_year.index and '2025' in price_by_year.index:
            pct_change_mean = ((price_by_year.loc['2025', 'mean'] - price_by_year.loc['2023', 'mean']) / 
                              price_by_year.loc['2023', 'mean'] * 100)
            pct_change_median = ((price_by_year.loc['2025', 'median'] - price_by_year.loc['2023', 'median']) / 
                                price_by_year.loc['2023', 'median'] * 100)
            
            metric_cols = st.columns(3)
            
            if price_col == 'price':
                metric_cols[0].metric(
                    "Median Price 2023",
                    f"${price_by_year.loc['2023', 'median']:,.0f}"
                )
                metric_cols[1].metric(
                    "Median Price 2025",
                    f"${price_by_year.loc['2025', 'median']:,.0f}",
                    f"{pct_change_median:.1f}%"
                )
                metric_cols[2].metric(
                    "Sample Size",
                    f"{price_by_year.loc['2023', 'count']:,} vs {price_by_year.loc['2025', 'count']:,}"
                )
            else:
                metric_cols[0].metric(
                    "Median Price/sqft 2023",
                    f"${price_by_year.loc['2023', 'median']:.2f}"
                )
                metric_cols[1].metric(
                    "Median Price/sqft 2025",
                    f"${price_by_year.loc['2025', 'median']:.2f}",
                    f"{pct_change_median:.1f}%"
                )
                metric_cols[2].metric(
                    "Sample Size",
                    f"{price_by_year.loc['2023', 'count']:,} vs {price_by_year.loc['2025', 'count']:,}"
                )
        
        # Overall price distribution by year
        st.markdown("### Price Distribution by Year")
        
        fig = px.box(
            df,
            x="dataset_year",
            y=price_col,
            color="dataset_year",
            title=f"{'Price' if price_col == 'price' else 'Price per sq ft'} Distribution (2023 vs 2025)",
            labels={
                "dataset_year": "Year",
                price_col: "Price ($)" if price_col == 'price' else "Price per sq ft ($)"
            }
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic analysis
    if selected_geo_col:
        st.markdown(f"### Price Trends by {selected_geo_col.title()}")
        
        # Calculate median prices by geography and year
        geo_price_data = df.groupby([selected_geo_col, 'dataset_year'])[price_col].median().reset_index()
        geo_price_data = geo_price_data.pivot(index=selected_geo_col, columns='dataset_year', values=price_col)
        
        # Calculate percent change
        if '2023' in geo_price_data.columns and '2025' in geo_price_data.columns:
            geo_price_data['pct_change'] = ((geo_price_data['2025'] - geo_price_data['2023']) / 
                                           geo_price_data['2023'] * 100)
        
        # Sort by percent change
        if 'pct_change' in geo_price_data.columns:
            geo_price_data = geo_price_data.sort_values('pct_change', ascending=False)
        
        # Get top and bottom areas by price change
        num_areas = min(10, len(geo_price_data))
        top_areas = geo_price_data.head(num_areas).index.tolist()
        bottom_areas = geo_price_data.tail(num_areas).index.tolist()
        
        # Create plot with tabs for different views
        tab1, tab2, tab3 = st.tabs(["All Areas", "Top Growth Areas", "Bottom Growth Areas"])
        
        # Tab 1: All areas
        with tab1:
            # Bar chart of percent change
            if 'pct_change' in geo_price_data.columns:
                fig = px.bar(
                    geo_price_data.reset_index(),
                    x=selected_geo_col,
                    y='pct_change',
                    title=f"Price Change by {selected_geo_col.title()} (2023-2025)",
                    labels={
                        selected_geo_col: selected_geo_col.title(),
                        'pct_change': 'Percent Change (%)'
                    },
                    color='pct_change',
                    color_continuous_scale='RdBu_r',
                    range_color=(-30, 30),
                )
                fig.update_layout(height=500)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Table of all areas
            st.dataframe(geo_price_data.reset_index())
        
        # Tab 2: Top growth areas
        with tab2:
            filtered_df_top = df[df[selected_geo_col].isin(top_areas)]
            
            if not filtered_df_top.empty:
                fig = px.box(
                    filtered_df_top,
                    x=selected_geo_col,
                    y=price_col,
                    color="dataset_year",
                    title=f"Top {num_areas} Areas by Price Growth",
                    labels={
                        selected_geo_col: selected_geo_col.title(),
                        price_col: "Price ($)" if price_col == 'price' else "Price per sq ft ($)",
                        "dataset_year": "Year"
                    }
                )
                fig.update_layout(height=500)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Bottom growth areas
        with tab3:
            filtered_df_bottom = df[df[selected_geo_col].isin(bottom_areas)]
            
            if not filtered_df_bottom.empty:
                fig = px.box(
                    filtered_df_bottom,
                    x=selected_geo_col,
                    y=price_col,
                    color="dataset_year",
                    title=f"Bottom {num_areas} Areas by Price Growth",
                    labels={
                        selected_geo_col: selected_geo_col.title(),
                        price_col: "Price ($)" if price_col == 'price' else "Price per sq ft ($)",
                        "dataset_year": "Year"
                    }
                )
                fig.update_layout(height=500)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    # Property type analysis
    if 'homeType' in df.columns:
        st.markdown("### Price Trends by Property Type")
        
        # Calculate median prices by property type and year
        home_type_data = df.groupby(['homeType', 'dataset_year'])[price_col].median().reset_index()
        home_type_pivot = home_type_data.pivot(index='homeType', columns='dataset_year', values=price_col)
        
        # Calculate percent change
        if '2023' in home_type_pivot.columns and '2025' in home_type_pivot.columns:
            home_type_pivot['pct_change'] = ((home_type_pivot['2025'] - home_type_pivot['2023']) / 
                                            home_type_pivot['2023'] * 100)
            
            # Bar chart of price change by property type
            fig = px.bar(
                home_type_pivot.reset_index(),
                x='homeType',
                y='pct_change',
                title="Price Change by Property Type (2023-2025)",
                labels={
                    'homeType': 'Property Type',
                    'pct_change': 'Percent Change (%)'
                },
                color='pct_change',
                color_continuous_scale='RdBu_r',
                range_color=(-30, 30),
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            # Table with property type prices
            st.dataframe(home_type_pivot.reset_index())
        
        # Box plot of prices by property type and year
        fig = px.box(
            df,
            x="homeType",
            y=price_col,
            color="dataset_year",
            title=f"{'Price' if price_col == 'price' else 'Price per sq ft'} by Property Type",
            labels={
                "homeType": "Property Type",
                price_col: "Price ($)" if price_col == 'price' else "Price per sq ft ($)",
                "dataset_year": "Year"
            }
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Price range distribution
    st.markdown("### Price Range Distribution")
    
    # Create price range bins
    price_bins = [0, 200000, 400000, 600000, 800000, 1000000, 1500000, 2000000, float('inf')]
    bin_labels = ['<200k', '200k-400k', '400k-600k', '600k-800k', '800k-1M', '1M-1.5M', '1.5M-2M', '>2M']
    
    if price_col == 'pricePerSqFt':
        price_bins = [0, 100, 200, 300, 400, 500, 750, 1000, float('inf')]
        bin_labels = ['<$100', '$100-200', '$200-300', '$300-400', '$400-500', '$500-750', '$750-1000', '>$1000']
    
    # Create price range categories
    df['price_range'] = pd.cut(df[price_col], bins=price_bins, labels=bin_labels)
    
    # Count properties in each price range by year
    price_range_counts = df.groupby(['dataset_year', 'price_range']).size().reset_index(name='count')
    
    # Convert to percentages
    total_counts = price_range_counts.groupby('dataset_year')['count'].sum().reset_index()
    price_range_counts = price_range_counts.merge(total_counts, on='dataset_year', suffixes=('', '_total'))
    price_range_counts['percentage'] = price_range_counts['count'] / price_range_counts['count_total'] * 100
    
    # Create grouped bar chart
    fig = px.bar(
        price_range_counts,
        x="price_range",
        y="percentage",
        color="dataset_year",
        barmode="group",
        title=f"{'Price' if price_col == 'price' else 'Price per sq ft'} Range Distribution (2023 vs 2025)",
        labels={
            "price_range": f"{'Price' if price_col == 'price' else 'Price per sq ft'} Range",
            "percentage": "Percentage of Properties (%)",
            "dataset_year": "Year"
        }
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Price distribution shift (kernel density)
    fig = px.histogram(
        df,
        x=price_col,
        color="dataset_year",
        marginal="rug",
        opacity=0.7,
        nbins=50,
        histnorm='probability density',
        title=f"{'Price' if price_col == 'price' else 'Price per sq ft'} Distribution Shift",
        labels={
            price_col: "Price ($)" if price_col == 'price' else "Price per sq ft ($)",
            "dataset_year": "Year"
        }
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True) 