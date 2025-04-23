import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk

def display_property_map(df_2023, df_2025, df_combined):
    """
    Display an interactive property map visualization
    
    Parameters:
    -----------
    df_2023 : DataFrame or None
        2023 property data
    df_2025 : DataFrame or None
        2025 property data
    df_combined : DataFrame or None
        Combined property data
    """
    st.header("🗺️ Interactive Property Map")
    st.markdown("Explore properties geographically and discover price patterns across neighborhoods")
    
    # Select dataset to display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Determine available datasets
        available_datasets = []
        if df_2023 is not None:
            available_datasets.append("2023 Dataset")
        if df_2025 is not None:
            available_datasets.append("2025 Dataset")
        if df_combined is not None:
            available_datasets.append("Combined Dataset")
        
        # Add overlay option if both 2023 and 2025 datasets are available
        if df_2023 is not None and df_2025 is not None:
            available_datasets.append("Overlay (2023 & 2025)")
        
        if not available_datasets:
            st.error("No data available. Please load data first.")
            return
        
        selected_dataset = st.radio("Select dataset to display:", available_datasets, horizontal=True)
        
        if selected_dataset == "2023 Dataset":
            df = df_2023
            year = 2023
        elif selected_dataset == "2025 Dataset":
            df = df_2025
            year = 2025
        elif selected_dataset == "Overlay (2023 & 2025)":
            # Handle overlay case
            if 'zpid' not in df_2023.columns or 'zpid' not in df_2025.columns:
                st.error("Cannot create overlay: 'zpid' column missing from one or both datasets.")
                return
            
            # Find properties that exist in both datasets
            common_properties = set(df_2023['zpid']).intersection(set(df_2025['zpid']))
            
            if len(common_properties) == 0:
                st.warning("No matching properties found between 2023 and 2025 datasets.")
                return
            
            # Create a new dataframe with overlay information
            overlay_data = []
            
            for prop_id in common_properties:
                if prop_id in df_2023['zpid'].values and prop_id in df_2025['zpid'].values:
                    prop_2023 = df_2023[df_2023['zpid'] == prop_id].iloc[0]
                    prop_2025 = df_2025[df_2025['zpid'] == prop_id].iloc[0]
                    
                    # Use the most recent data for location
                    overlay_row = prop_2025.copy()
                    
                    # Calculate price change
                    price_2023 = pd.to_numeric(prop_2023.get('price', np.nan), errors='coerce')
                    price_2025 = pd.to_numeric(prop_2025.get('price', np.nan), errors='coerce')
                    
                    if pd.notna(price_2023) and pd.notna(price_2025) and price_2023 > 0:
                        price_change = price_2025 - price_2023
                        price_change_pct = (price_change / price_2023) * 100
                        overlay_row['price_change'] = price_change
                        overlay_row['price_change_pct'] = price_change_pct
                        overlay_row['price_2023'] = price_2023
                    
                    overlay_data.append(overlay_row)
            
            df = pd.DataFrame(overlay_data)
            year = "overlay"
            
            st.info(f"Displaying {len(df)} properties that appear in both 2023 and 2025 datasets")
        else:
            df = df_combined
            year = None
    
    with col2:
        st.markdown("### Map Controls")
        map_type = st.selectbox("Map Type", ["Scatter", "Heatmap", "3D Columns"])
    
    # Filter sidebar
    with st.expander("Property Filters", expanded=True):
        filter_cols = st.columns(4)
        
        with filter_cols[0]:
            # Price filter - Fixed to handle non-numeric values
            if 'price' in df.columns:
                # Ensure price column is numeric by converting to float and ignoring errors
                numeric_prices = pd.to_numeric(df['price'], errors='coerce')
                # Drop NaN values for min/max calculation
                numeric_prices = numeric_prices.dropna()
                
                if len(numeric_prices) > 0:
                    min_price = float(numeric_prices.min())
                    max_price = float(numeric_prices.max())
                    
                    # Use number inputs for manual entry + slider for range
                    price_col1, price_col2 = st.columns(2)
                    with price_col1:
                        min_price_input = st.number_input("Min Price ($)", 
                                                         value=int(min_price),
                                                         min_value=int(min_price),
                                                         max_value=int(max_price),
                                                         step=10000)
                    with price_col2:
                        max_price_input = st.number_input("Max Price ($)", 
                                                         value=int(max_price),
                                                         min_value=int(min_price),
                                                         max_value=int(max_price),
                                                         step=10000)
                    
                    # Slider for price range (uses the manual inputs as values)
                    price_range = st.slider(
                        "Price Range",
                        min_value=min_price,
                        max_value=max_price,
                        value=(float(min_price_input), float(max_price_input)),
                        step=10000.0
                    )
                else:
                    st.warning("No valid price data available")
                    price_range = None
            else:
                price_range = None
        
        with filter_cols[1]:
            # Bedrooms filter
            if 'bedrooms' in df.columns:
                # Convert to numeric, coerce errors to NaN
                numeric_beds = pd.to_numeric(df['bedrooms'], errors='coerce')
                bed_options = sorted([int(b) for b in numeric_beds.dropna().unique() if b > 0 and b < 10])
                if bed_options:
                    min_bed = min(bed_options)
                    max_bed = max(bed_options)
                    
                    # Use number inputs for manual entry + slider for range
                    beds_col1, beds_col2 = st.columns(2)
                    with beds_col1:
                        min_beds_input = st.number_input("Min Beds", 
                                                        value=min_bed,
                                                        min_value=min_bed,
                                                        max_value=max_bed,
                                                        step=1)
                    with beds_col2:
                        max_beds_input = st.number_input("Max Beds", 
                                                        value=max_bed,
                                                        min_value=min_bed,
                                                        max_value=max_bed,
                                                        step=1)
                    
                    bedrooms = st.slider("Bedrooms Range", 
                                        min_bed, max_bed, 
                                        (min_beds_input, max_beds_input))
                else:
                    bedrooms = None
            else:
                bedrooms = None
        
        with filter_cols[2]:
            # Bathrooms filter
            if 'bathrooms' in df.columns:
                # Convert to numeric, coerce errors to NaN
                numeric_baths = pd.to_numeric(df['bathrooms'], errors='coerce')
                bath_options = sorted([float(b) for b in numeric_baths.dropna().unique() if b > 0 and b < 10])
                if bath_options:
                    min_bath = min(bath_options)
                    max_bath = max(bath_options)
                    
                    # Use number inputs for manual entry + slider for range
                    baths_col1, baths_col2 = st.columns(2)
                    with baths_col1:
                        min_baths_input = st.number_input("Min Baths", 
                                                         value=float(min_bath),
                                                         min_value=float(min_bath),
                                                         max_value=float(max_bath),
                                                         step=0.5)
                    with baths_col2:
                        max_baths_input = st.number_input("Max Baths", 
                                                         value=float(max_bath),
                                                         min_value=float(min_bath),
                                                         max_value=float(max_bath),
                                                         step=0.5)
                    
                    bathrooms = st.slider("Bathrooms Range", 
                                          min_bath, max_bath, 
                                          (float(min_baths_input), float(max_baths_input)))
                else:
                    bathrooms = None
            else:
                bathrooms = None
        
        with filter_cols[3]:
            # Home type filter
            if 'homeType' in df.columns:
                home_types = df['homeType'].dropna().unique()
                selected_home_types = st.multiselect(
                    "Home Types",
                    options=home_types,
                    default=home_types
                )
            else:
                selected_home_types = None
    
    # Apply filters
    filtered_df = df.copy()
    
    # Ensure price column is numeric for filtering
    if 'price' in filtered_df.columns:
        filtered_df['price'] = pd.to_numeric(filtered_df['price'], errors='coerce')
    
    if price_range and 'price' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & 
                                  (filtered_df['price'] <= price_range[1])]
    
    # Ensure bedrooms column is numeric for filtering
    if 'bedrooms' in filtered_df.columns:
        filtered_df['bedrooms'] = pd.to_numeric(filtered_df['bedrooms'], errors='coerce')
        
    if bedrooms and 'bedrooms' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['bedrooms'] >= bedrooms[0]) & 
                                  (filtered_df['bedrooms'] <= bedrooms[1])]
    
    # Ensure bathrooms column is numeric for filtering
    if 'bathrooms' in filtered_df.columns:
        filtered_df['bathrooms'] = pd.to_numeric(filtered_df['bathrooms'], errors='coerce')
        
    if bathrooms and 'bathrooms' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['bathrooms'] >= bathrooms[0]) & 
                                  (filtered_df['bathrooms'] <= bathrooms[1])]
    
    if selected_home_types and 'homeType' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['homeType'].isin(selected_home_types)]
    
    # Check if we have necessary columns for mapping
    map_cols = ['latitude', 'longitude', 'price']
    if not all(col in filtered_df.columns for col in map_cols):
        missing_cols = [col for col in map_cols if col not in filtered_df.columns]
        st.error(f"Missing required columns for mapping: {', '.join(missing_cols)}")
        return
    
    # Filter out missing coordinates
    map_df = filtered_df.dropna(subset=['latitude', 'longitude'])
    
    if len(map_df) == 0:
        st.warning("No properties to display with the current filters.")
        return
    
    # Display map
    st.markdown(f"### Displaying {len(map_df)} properties")
    
    # Get map center point
    center_lat = map_df['latitude'].mean()
    center_lon = map_df['longitude'].mean()
    
    # Create property tooltips - ensure numeric conversion for display
    map_df['tooltip'] = map_df.apply(
        lambda row: create_tooltip(row, year),
        axis=1
    )
    
    if map_type == "Scatter":
        # Create color scale
        if 'price' in map_df.columns:
            # Ensure price is numeric for scatter plot
            valid_price_df = map_df.copy()
            valid_price_df['price'] = pd.to_numeric(valid_price_df['price'], errors='coerce')
            valid_price_df = valid_price_df.dropna(subset=['price'])
            
            if len(valid_price_df) > 0:
                # Configure color scale based on dataset type
                if year == "overlay":
                    color_column = "price_change_pct"
                    # Create a better diverging color scale for overlay view
                    color_scale = [
                        [0, "rgb(127,0,0)"],      # Dark red for negative
                        [0.2, "rgb(215,48,39)"],   # Medium red
                        [0.4, "rgb(252,141,89)"],  # Light red/orange
                        [0.5, "rgb(150,150,150)"], # Grey for neutral (instead of white/yellow)
                        [0.6, "rgb(145,191,219)"], # Light blue
                        [0.8, "rgb(44,123,182)"],  # Medium blue
                        [1.0, "rgb(0,0,127)"]      # Dark blue for positive
                    ]
                    
                    # Add column with +/- sign for display
                    valid_price_df['price_change_formatted'] = valid_price_df['price_change_pct'].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
                    
                    # Set range to make 0 the center
                    color_range = [-100, 100] # Fixed range for better visualization
                    
                    hover_data = {
                        'price': ':$,.0f',
                        'price_2023': ':$,.0f' if 'price_2023' in valid_price_df.columns else False,
                        'price_change': ':$,.0f' if 'price_change' in valid_price_df.columns else False,
                        'price_change_formatted': True,
                        'latitude': False,
                        'longitude': False,
                        'tooltip': False,
                        'price_change_pct': False
                    }
                else:
                    color_column = "price"
                    color_scale = px.colors.sequential.Plasma
                    color_range = (valid_price_df['price'].min(), valid_price_df['price'].max())
                    hover_data = {
                        'price': ':$,.0f',
                        'latitude': False,
                        'longitude': False,
                        'tooltip': False
                    }
                
                fig = px.scatter_mapbox(
                    valid_price_df,
                    lat="latitude",
                    lon="longitude",
                    color=color_column,
                    color_continuous_scale=color_scale,
                    range_color=color_range,
                    zoom=10,
                    mapbox_style="open-street-map",
                    hover_name="tooltip" if 'tooltip' in valid_price_df.columns else None,
                    hover_data=hover_data if color_column in valid_price_df.columns else None,
                    title=f"Property Map ({selected_dataset})"
                )
                fig.update_layout(height=700, margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid price data available for map display")
    
    elif map_type == "Heatmap":
        # Create heatmap layer with PyDeck
        # Ensure price is numeric for heatmap
        valid_price_df = map_df.copy()
        valid_price_df['price'] = pd.to_numeric(valid_price_df['price'], errors='coerce')
        valid_price_df = valid_price_df.dropna(subset=['price'])
        
        if len(valid_price_df) > 0:
            layer = pdk.Layer(
                "HeatmapLayer",
                valid_price_df,
                get_position=["longitude", "latitude"],
                get_weight="price",
                aggregation="MEAN",
                threshold=0.05,
                pickable=True,
            )
            
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=10,
                pitch=0
            )
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/light-v9",
                tooltip={"text": "Avg Price: ${weight}"}
            )
            
            st.pydeck_chart(r)
    
    elif map_type == "3D Columns":
        # Create a column layer with height based on price
        # Ensure price is numeric for 3D columns
        valid_price_df = map_df.copy()
        valid_price_df['price'] = pd.to_numeric(valid_price_df['price'], errors='coerce')
        valid_price_df = valid_price_df.dropna(subset=['price'])
        
        if len(valid_price_df) > 0:
            max_price = valid_price_df['price'].max()
            min_price = valid_price_df['price'].min()
            
            # Scale prices to reasonable heights (0-500 meters)
            valid_price_df['price_scaled'] = 50 + 500 * (valid_price_df['price'] - min_price) / (max_price - min_price)
            
            # Create tooltip HTML based on available data
            if year == "overlay":
                tooltip_html = """
                <b>Price:</b> ${price}<br>
                <b>Price 2023:</b> ${price_2023}<br>
                <b>Price Change:</b> ${price_change} ({price_change_pct}%)<br>
                <b>Address:</b> {streetAddress}
                """
            else:
                tooltip_html = """
                <b>Price:</b> ${price}<br>
                <b>Address:</b> {streetAddress}
                """
                
                # Add market status if available
                if 'homeStatus' in valid_price_df.columns:
                    tooltip_html += "<br><b>Status:</b> {homeStatus}"
            
            layer = pdk.Layer(
                "ColumnLayer",
                data=valid_price_df,
                get_position=["longitude", "latitude"],
                get_elevation="price_scaled",
                elevation_scale=1,
                radius=50,
                get_fill_color=["price_scaled / 500 * 255", "100", "255 - price_scaled / 500 * 255", 140],
                pickable=True,
                auto_highlight=True,
            )
            
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=10,
                pitch=45,
                bearing=0
            )
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/light-v9",
                tooltip={
                    "html": tooltip_html,
                    "style": {
                        "backgroundColor": "white",
                        "color": "black"
                    }
                }
            )
            
            st.pydeck_chart(r)
        else:
            st.warning("No valid price data available for 3D column display")
    
    # Show data summary
    with st.expander("Data Summary"):
        if 'city' in filtered_df.columns:
            # City summary
            city_counts = filtered_df['city'].value_counts().reset_index()
            city_counts.columns = ['City', 'Count']
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Properties by City")
                st.dataframe(city_counts, height=250)
            
            with col2:
                if len(city_counts) > 0:
                    # Top 10 cities
                    fig = px.pie(
                        city_counts.head(10), 
                        values='Count', 
                        names='City', 
                        title='Top 10 Cities by Property Count'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Price statistics - ensure numeric values for calculations
        if 'price' in filtered_df.columns:
            # Convert to numeric and handle non-numeric values
            numeric_prices = pd.to_numeric(filtered_df['price'], errors='coerce')
            valid_prices = numeric_prices.dropna()
            
            if len(valid_prices) > 0:
                st.markdown("### Price Statistics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Median Price", f"${valid_prices.median():,.0f}")
                col2.metric("Mean Price", f"${valid_prices.mean():,.0f}")
                col3.metric("Min Price", f"${valid_prices.min():,.0f}")
                col4.metric("Max Price", f"${valid_prices.max():,.0f}")
                
                # Create a temporary dataframe with only valid prices for the histogram
                temp_df = pd.DataFrame({'price': valid_prices})
                
                # Price distribution
                fig = px.histogram(
                    temp_df,
                    x="price",
                    nbins=50,
                    title="Price Distribution"
                )
                fig.update_xaxes(title="Price ($)")
                fig.update_yaxes(title="Count")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid price data available for statistics")

def create_tooltip(row, year):
    """Create tooltip text with property information"""
    # Start with price
    price = pd.to_numeric(row.get('price', 0), errors='coerce')
    tooltip = f"${int(price):,}<br>"
    
    # Add address if available
    address_parts = []
    if 'streetAddress' in row and pd.notna(row['streetAddress']):
        address_parts.append(str(row['streetAddress']))
    if 'city' in row and pd.notna(row['city']):
        address_parts.append(str(row['city']))
    if 'state' in row and pd.notna(row['state']):
        address_parts.append(str(row['state']))
    
    if address_parts:
        tooltip += f"<b>Address:</b> {', '.join(address_parts)}<br>"
    
    # Add property specs
    specs = []
    if 'bedrooms' in row and pd.notna(pd.to_numeric(row.get('bedrooms', np.nan), errors='coerce')):
        beds = int(pd.to_numeric(row['bedrooms'], errors='coerce'))
        specs.append(f"{beds}bd")
    
    if 'bathrooms' in row and pd.notna(pd.to_numeric(row.get('bathrooms', np.nan), errors='coerce')):
        baths = pd.to_numeric(row['bathrooms'], errors='coerce')
        specs.append(f"{baths}ba")
    
    if 'livingArea' in row and pd.notna(pd.to_numeric(row.get('livingArea', np.nan), errors='coerce')):
        area = int(pd.to_numeric(row['livingArea'], errors='coerce'))
        specs.append(f"{area}sqft")
    
    if specs:
        tooltip += f"{' '.join(specs)}<br>"
    
    # Add home type if available
    if 'homeType' in row and pd.notna(row['homeType']):
        tooltip += f"<b>Type:</b> {row['homeType']}<br>"
    
    # Add market status if available
    if 'homeStatus' in row and pd.notna(row['homeStatus']):
        tooltip += f"<b>Status:</b> {row['homeStatus']}<br>"
    
    # Add price change info if in overlay mode
    if year == "overlay" and 'price_change' in row and 'price_2023' in row:
        price_2023 = pd.to_numeric(row.get('price_2023', np.nan), errors='coerce')
        price_change = pd.to_numeric(row.get('price_change', np.nan), errors='coerce')
        price_change_pct = pd.to_numeric(row.get('price_change_pct', np.nan), errors='coerce')
        
        if pd.notna(price_2023):
            tooltip += f"<b>2023 Price:</b> ${int(price_2023):,}<br>"
        
        if pd.notna(price_change) and pd.notna(price_change_pct):
            change_sign = "+" if price_change >= 0 else ""
            tooltip += f"<b>Change:</b> {change_sign}${int(price_change):,} ({change_sign}{price_change_pct:.1f}%)<br>"
    
    return tooltip 