import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_property_time_comparison(df_2023, df_2025, df_combined):
    """
    Display comparison of properties that appear in both 2023 and 2025 datasets
    
    Parameters:
    -----------
    df_2023 : DataFrame or None
        2023 property data
    df_2025 : DataFrame or None
        2025 property data
    df_combined : DataFrame or None
        Combined property data
    """
    st.header("📊 Property Time Comparison (2023 vs 2025)")
    st.markdown("Compare properties that appear in both 2023 and 2025 datasets")
    
    # Check if we have both datasets
    if df_2023 is None or df_2025 is None:
        st.warning("This feature requires both 2023 and 2025 datasets to be loaded.")
        return
    
    # Ensure there's a zpid column for matching
    if 'zpid' not in df_2023.columns or 'zpid' not in df_2025.columns:
        st.error("Cannot match properties: 'zpid' column missing from one or both datasets.")
        return
    
    # Find properties that exist in both datasets
    common_properties = set(df_2023['zpid']).intersection(set(df_2025['zpid']))
    
    if len(common_properties) == 0:
        st.warning("No matching properties found between 2023 and 2025 datasets.")
        return
    
    st.info(f"Found {len(common_properties)} properties that appear in both 2023 and 2025 datasets.")
    
    # Filter to just the common properties
    df_2023_common = df_2023[df_2023['zpid'].isin(common_properties)]
    df_2025_common = df_2025[df_2025['zpid'].isin(common_properties)]
    
    # Create a combined view for easier comparison
    comparison_data = []
    
    for prop_id in common_properties:
        prop_2023 = df_2023_common[df_2023_common['zpid'] == prop_id].iloc[0]
        prop_2025 = df_2025_common[df_2025_common['zpid'] == prop_id].iloc[0]
        
        # Common fields to extract (customize based on your data)
        address = prop_2023.get('streetAddress', 'Unknown')
        city = prop_2023.get('city', 'Unknown')
        state = prop_2023.get('state', 'Unknown')
        
        # Get price values and ensure they are numeric
        price_2023 = pd.to_numeric(prop_2023.get('price', np.nan), errors='coerce')
        price_2025 = pd.to_numeric(prop_2025.get('price', np.nan), errors='coerce')
        
        # Calculate price change
        if pd.notna(price_2023) and pd.notna(price_2025) and price_2023 > 0:
            price_change = price_2025 - price_2023
            price_change_pct = (price_change / price_2023) * 100
        else:
            price_change = np.nan
            price_change_pct = np.nan
        
        # Extract other interesting fields and ensure they are numeric
        living_area_2023 = pd.to_numeric(prop_2023.get('livingArea', np.nan), errors='coerce')
        living_area_2025 = pd.to_numeric(prop_2025.get('livingArea', np.nan), errors='coerce')
        
        bedrooms_2023 = pd.to_numeric(prop_2023.get('bedrooms', np.nan), errors='coerce')
        bedrooms_2025 = pd.to_numeric(prop_2025.get('bedrooms', np.nan), errors='coerce')
        
        bathrooms_2023 = pd.to_numeric(prop_2023.get('bathrooms', np.nan), errors='coerce')
        bathrooms_2025 = pd.to_numeric(prop_2025.get('bathrooms', np.nan), errors='coerce')
        
        # Add to comparison data
        comparison_data.append({
            'zpid': prop_id,
            'address': f"{address}, {city}, {state}",
            'price_2023': price_2023,
            'price_2025': price_2025,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'living_area_2023': living_area_2023,
            'living_area_2025': living_area_2025,
            'bedrooms_2023': bedrooms_2023,
            'bedrooms_2025': bedrooms_2025,
            'bathrooms_2023': bathrooms_2023,
            'bathrooms_2025': bathrooms_2025
        })
    
    # Convert to DataFrame
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display overview statistics
    st.markdown("## Price Change Overview")
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate median price change
    median_price_change = df_comparison['price_change'].median()
    median_price_change_pct = df_comparison['price_change_pct'].median()
    
    # Calculate how many properties increased/decreased in price
    price_increases = (df_comparison['price_change'] > 0).sum()
    price_decreases = (df_comparison['price_change'] < 0).sum()
    price_unchanged = (df_comparison['price_change'] == 0).sum()
    
    with col1:
        st.metric("Median Price Change", f"${median_price_change:,.0f}", f"{median_price_change_pct:.1f}%")
    
    with col2:
        st.metric("Properties with Price Increase", f"{price_increases} ({price_increases/len(df_comparison)*100:.1f}%)")
    
    with col3:
        st.metric("Properties with Price Decrease", f"{price_decreases} ({price_decreases/len(df_comparison)*100:.1f}%)")
    
    # Display price change distribution
    fig = px.histogram(
        df_comparison, 
        x='price_change_pct',
        title="Distribution of Price Changes (2023 to 2025)",
        labels={'price_change_pct': 'Price Change (%)'},
        nbins=30
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Property Selection
    st.markdown("## Property Detail View")
    st.write("Select a property to see detailed comparison:")
    
    # Create a selection dataframe with property identifiers
    selection_df = df_comparison[['zpid', 'address', 'price_2023', 'price_2025', 'price_change_pct']].copy()
    selection_df['price_2023'] = selection_df['price_2023'].map('${:,.0f}'.format)
    selection_df['price_2025'] = selection_df['price_2025'].map('${:,.0f}'.format)
    selection_df['price_change_pct'] = selection_df['price_change_pct'].map('{:+.1f}%'.format)
    
    # Sort by absolute price change percentage to show most significant changes first
    selection_df = selection_df.sort_values(by='price_change_pct', key=lambda x: x.str.replace('+', '').str.replace('%', '').astype(float).abs(), ascending=False)
    
    # Display as a table with selection
    selected_index = st.selectbox("Select Property", options=selection_df.index, format_func=lambda x: f"{selection_df.loc[x, 'address']} - {selection_df.loc[x, 'price_change_pct']} change")
    
    if selected_index is not None:
        selected_zpid = df_comparison.loc[selected_index, 'zpid']
        property_2023 = df_2023[df_2023['zpid'] == selected_zpid].iloc[0]
        property_2025 = df_2025[df_2025['zpid'] == selected_zpid].iloc[0]
        
        st.markdown(f"### Selected Property: {df_comparison.loc[selected_index, 'address']}")
        
        # Display property comparison side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 2023 Data")
            display_property_card(property_2023)
        
        with col2:
            st.markdown("#### 2025 Data")
            display_property_card(property_2025)
        
        # Display price trend
        st.markdown("#### Price Trend")
        
        fig = go.Figure()
        
        # Make sure to use numeric values
        price_2023 = pd.to_numeric(property_2023['price'], errors='coerce')
        price_2025 = pd.to_numeric(property_2025['price'], errors='coerce')
        
        fig.add_trace(go.Scatter(
            x=['2023', '2025'],
            y=[price_2023, price_2025],
            mode='lines+markers',
            name='Price',
            line=dict(width=3)
        ))
        
        fig.update_layout(
            title="Property Price Trend",
            xaxis_title="Year",
            yaxis_title="Price ($)",
            template="plotly_white"
        )
        
        # Add annotations for price values
        fig.add_annotation(
            x='2023',
            y=price_2023,
            text=f"${price_2023:,.0f}",
            showarrow=True,
            arrowhead=1
        )
        
        fig.add_annotation(
            x='2025',
            y=price_2025,
            text=f"${price_2025:,.0f}",
            showarrow=True,
            arrowhead=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display comparison table
    st.markdown("## All Properties Comparison")
    
    # Create a display version of the comparison dataframe with formatted values
    display_df = df_comparison.copy()
    
    # Format monetary values
    for col in ['price_2023', 'price_2025', 'price_change']:
        display_df[col] = display_df[col].map('${:,.0f}'.format)
    
    # Format percentage values
    display_df['price_change_pct'] = display_df['price_change_pct'].map('{:+.1f}%'.format)
    
    # Display as sortable table
    st.dataframe(
        display_df[['address', 'price_2023', 'price_2025', 'price_change', 'price_change_pct']], 
        use_container_width=True
    )

def display_property_card(property_data):
    """Helper function to display property information in a card format"""
    
    # Ensure numeric values for display
    price = pd.to_numeric(property_data.get('price', 0), errors='coerce')
    bedrooms = pd.to_numeric(property_data.get('bedrooms', 'N/A'), errors='coerce')
    bathrooms = pd.to_numeric(property_data.get('bathrooms', 'N/A'), errors='coerce')
    living_area = pd.to_numeric(property_data.get('livingArea', 0), errors='coerce')
    year_built = pd.to_numeric(property_data.get('yearBuilt', 0), errors='coerce')
    
    property_info = {
        "Price": f"${price:,.0f}",
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Living Area": f"{living_area:,.0f} sq ft" if pd.notna(living_area) else 'N/A',
        "Year Built": int(year_built) if pd.notna(year_built) else 'N/A'
    }
    
    # Add home type if available
    if 'homeType' in property_data and pd.notna(property_data['homeType']):
        property_info["Home Type"] = property_data['homeType']
    
    # Display in a clean format
    for key, value in property_info.items():
        st.write(f"**{key}:** {value}") 