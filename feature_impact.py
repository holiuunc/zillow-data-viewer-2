import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Correlations", 
        "Feature Analysis", 
        "Price Predictors",
        "Geographic Impact"
    ])
    
    # Tab 1: Price Correlations
    with tab1:
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
                                title="Correlation Heatmap of Top Features"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Scatter plots of top correlated features
                        st.markdown("### Top Feature Relationships")
                        
                        top_corr_features = corr_df.head(min(3, len(corr_df)))['Feature'].tolist()
                        
                        if top_corr_features:
                            for feature in top_corr_features:
                                # Create a temporary dataframe with only valid values
                                temp_df = df[[feature, 'price']].dropna()
                                if len(temp_df) > 10:  # Need enough points for a meaningful scatter plot
                                    fig = px.scatter(
                                        temp_df,
                                        x=feature,
                                        y='price',
                                        title=f"{feature.replace('_', ' ').title()} vs Price",
                                        labels={
                                            feature: feature.replace('_', ' ').title(),
                                            'price': 'Price ($)'
                                        },
                                        opacity=0.6,
                                        trendline="ols"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No valid correlations found with price.")
                except Exception as e:
                    st.error(f"Error calculating correlations: {e}")
                    st.warning("Try adjusting your filters or selecting a different dataset.")
    
    # Tab 2: Feature Analysis
    with tab2:
        st.markdown("### Property Feature Analysis")
        
        if not numeric_features and not categorical_features:
            st.warning("No features available for detailed analysis.")
        else:
            # Feature selection
            feature_type = st.radio(
                "Feature Type",
                options=["Numeric Features", "Categorical Features"],
                index=0 if numeric_features else 1
            )
            
            if feature_type == "Numeric Features" and numeric_features:
                selected_feature = st.selectbox(
                    "Select Feature to Analyze",
                    options=numeric_features
                )
                
                if selected_feature:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribution of feature
                        fig = px.histogram(
                            df,
                            x=selected_feature,
                            title=f"Distribution of {selected_feature.replace('_', ' ').title()}",
                            labels={selected_feature: selected_feature.replace('_', ' ').title()},
                            nbins=30
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Feature statistics
                        feature_stats = df[selected_feature].describe()
                        
                        stats_data = {
                            "Statistic": ["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"],
                            "Value": [
                                feature_stats["count"],
                                feature_stats["mean"],
                                feature_stats["std"],
                                feature_stats["min"],
                                feature_stats["25%"],
                                feature_stats["50%"],
                                feature_stats["75%"],
                                feature_stats["max"]
                            ]
                        }
                        
                        st.dataframe(pd.DataFrame(stats_data), hide_index=True)
                        
                        # Feature impact on price
                        if df[selected_feature].nunique() > 5:
                            # Continuous feature - use scatter plot
                            st.markdown(f"### {selected_feature.replace('_', ' ').title()} Impact on Price")
                            
                            fig = px.scatter(
                                df,
                                x=selected_feature,
                                y='price',
                                title=f"Price vs {selected_feature.replace('_', ' ').title()}",
                                trendline="ols",
                                trendline_color_override="red",
                                opacity=0.6,
                                labels={
                                    selected_feature: selected_feature.replace('_', ' ').title(),
                                    'price': 'Price ($)'
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Discrete feature - use box plot
                            fig = px.box(
                                df,
                                x=selected_feature,
                                y='price',
                                title=f"Price Distribution by {selected_feature.replace('_', ' ').title()}",
                                labels={
                                    selected_feature: selected_feature.replace('_', ' ').title(),
                                    'price': 'Price ($)'
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            elif feature_type == "Categorical Features" and categorical_features:
                selected_cat_feature = st.selectbox(
                    "Select Feature to Analyze",
                    options=categorical_features
                )
                
                if selected_cat_feature:
                    # Count of categories
                    cat_counts = df[selected_cat_feature].value_counts().reset_index()
                    cat_counts.columns = [selected_cat_feature, 'Count']
                    
                    fig = px.bar(
                        cat_counts,
                        x=selected_cat_feature,
                        y='Count',
                        title=f"Count of Properties by {selected_cat_feature.replace('_', ' ').title()}",
                        labels={selected_cat_feature: selected_cat_feature.replace('_', ' ').title()}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Price by category
                    cat_price = df.groupby(selected_cat_feature)['price'].agg(['mean', 'median', 'count']).reset_index()
                    cat_price.columns = [selected_cat_feature, 'Mean Price', 'Median Price', 'Count']
                    
                    # Sort by median price
                    cat_price = cat_price.sort_values('Median Price', ascending=False)
                    
                    fig = px.bar(
                        cat_price,
                        x=selected_cat_feature,
                        y='Median Price',
                        title=f"Median Price by {selected_cat_feature.replace('_', ' ').title()}",
                        labels={
                            selected_cat_feature: selected_cat_feature.replace('_', ' ').title(),
                            'Median Price': 'Median Price ($)'
                        },
                        hover_data=['Mean Price', 'Count']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Box plot of price by category
                    fig = px.box(
                        df,
                        x=selected_cat_feature,
                        y='price',
                        title=f"Price Distribution by {selected_cat_feature.replace('_', ' ').title()}",
                        labels={
                            selected_cat_feature: selected_cat_feature.replace('_', ' ').title(),
                            'price': 'Price ($)'
                        }
                    )
                    fig.update_layout(xaxis={'categoryorder':'total descending'})
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Price Predictors
    with tab3:
        st.markdown("### Price Prediction Analysis")
        
        if len(numeric_features) < 2:
            st.warning("Not enough numeric features for prediction analysis.")
        else:
            # Select features for price prediction
            st.markdown("#### Select Features for Price Prediction Model")
            selected_pred_features = st.multiselect(
                "Select Features",
                options=numeric_features,
                default=numeric_features[:min(5, len(numeric_features))]
            )
            
            if not selected_pred_features:
                st.warning("Please select at least one feature for prediction analysis.")
            else:
                # Prepare data for prediction
                X = df[selected_pred_features].copy()
                y = df['price']
                
                # Handle missing values
                X = X.fillna(X.mean())
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Create and fit model
                model = LinearRegression()
                model.fit(X_scaled, y)
                
                # Get feature coefficients
                coef_df = pd.DataFrame({
                    'Feature': selected_pred_features,
                    'Coefficient': model.coef_
                })
                
                # Calculate feature importance (absolute value of coefficients)
                coef_df['Absolute Importance'] = abs(coef_df['Coefficient'])
                coef_df = coef_df.sort_values('Absolute Importance', ascending=False)
                
                # Display model results
                st.markdown("#### Feature Importance in Price Prediction")
                
                fig = px.bar(
                    coef_df,
                    x='Feature',
                    y='Coefficient',
                    title="Feature Coefficients in Price Prediction Model",
                    labels={
                        'Feature': 'Property Feature',
                        'Coefficient': 'Coefficient Value'
                    },
                    color='Coefficient',
                    color_continuous_scale=px.colors.diverging.RdBu_r
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Model performance
                y_pred = model.predict(X_scaled)
                
                # Calculate metrics
                r2 = np.corrcoef(y, y_pred)[0, 1]**2
                mae = np.mean(np.abs(y - y_pred))
                mape = np.mean(np.abs((y - y_pred) / y)) * 100
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("R² (Variance Explained)", f"{r2:.4f}")
                col2.metric("Mean Absolute Error", f"${mae:,.0f}")
                col3.metric("Mean Absolute % Error", f"{mape:.2f}%")
                
                # Display scatter plot of predicted vs actual prices
                st.markdown("#### Predicted vs Actual Prices")
                
                pred_df = pd.DataFrame({
                    'Actual Price': y,
                    'Predicted Price': y_pred
                })
                
                fig = px.scatter(
                    pred_df,
                    x='Actual Price',
                    y='Predicted Price',
                    title="Predicted vs Actual Prices",
                    labels={
                        'Actual Price': 'Actual Price ($)',
                        'Predicted Price': 'Predicted Price ($)'
                    },
                    opacity=0.6
                )
                
                # Add perfect prediction line
                min_val = min(pred_df['Actual Price'].min(), pred_df['Predicted Price'].min())
                max_val = max(pred_df['Actual Price'].max(), pred_df['Predicted Price'].max())
                
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Geographic Impact
    with tab4:
        st.markdown("### Geographic Impact on Price")
        
        geo_features = [col for col in ['city', 'zipcode', 'neighborhood', 'state'] if col in df.columns]
        
        if not geo_features:
            st.warning("No geographic features found in the dataset.")
        else:
            selected_geo_feature = st.selectbox(
                "Select Geographic Feature",
                options=geo_features
            )
            
            if selected_geo_feature:
                # Get price statistics by geographic feature
                geo_stats = df.groupby(selected_geo_feature)['price'].agg(['mean', 'median', 'count']).reset_index()
                geo_stats.columns = [selected_geo_feature, 'Mean Price', 'Median Price', 'Count']
                
                # Filter to areas with enough data points
                min_count = max(5, int(0.01 * len(df)))
                geo_stats = geo_stats[geo_stats['Count'] >= min_count]
                
                # Sort by median price
                geo_stats = geo_stats.sort_values('Median Price', ascending=False)
                
                # Display top and bottom areas
                st.markdown(f"#### Price by {selected_geo_feature.title()}")
                
                if len(geo_stats) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Top 10 Areas by Price")
                        top_areas = geo_stats.head(10)
                        
                        fig = px.bar(
                            top_areas,
                            x=selected_geo_feature,
                            y='Median Price',
                            title=f"Top 10 {selected_geo_feature.title()}s by Median Price",
                            labels={
                                selected_geo_feature: selected_geo_feature.title(),
                                'Median Price': 'Median Price ($)'
                            },
                            hover_data=['Mean Price', 'Count'],
                            color='Median Price',
                            color_continuous_scale=px.colors.sequential.Viridis
                        )
                        fig.update_layout(xaxis={'categoryorder':'total descending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("##### Bottom 10 Areas by Price")
                        bottom_areas = geo_stats.tail(10).sort_values('Median Price')
                        
                        fig = px.bar(
                            bottom_areas,
                            x=selected_geo_feature,
                            y='Median Price',
                            title=f"Bottom 10 {selected_geo_feature.title()}s by Median Price",
                            labels={
                                selected_geo_feature: selected_geo_feature.title(),
                                'Median Price': 'Median Price ($)'
                            },
                            hover_data=['Mean Price', 'Count'],
                            color='Median Price',
                            color_continuous_scale=px.colors.sequential.Viridis
                        )
                        fig.update_layout(xaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Box plot of price distribution by geographic feature
                    if len(geo_stats) <= 15:  # Only show if we have a reasonable number of areas
                        st.markdown(f"#### Price Distribution by {selected_geo_feature.title()}")
                        
                        fig = px.box(
                            df[df[selected_geo_feature].isin(geo_stats[selected_geo_feature])],
                            x=selected_geo_feature,
                            y='price',
                            title=f"Price Distribution by {selected_geo_feature.title()}",
                            labels={
                                selected_geo_feature: selected_geo_feature.title(),
                                'price': 'Price ($)'
                            }
                        )
                        fig.update_layout(xaxis={'categoryorder':'total descending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Map visualization if we have coordinates
                    if 'latitude' in df.columns and 'longitude' in df.columns:
                        st.markdown(f"#### Geographic Price Visualization")
                        
                        # Aggregate by geographic feature to get average coordinates
                        map_data = df.groupby(selected_geo_feature).agg({
                            'latitude': 'mean',
                            'longitude': 'mean',
                            'price': 'median',
                            'zpid': 'count'  # Count of properties
                        }).reset_index()
                        
                        map_data.columns = [selected_geo_feature, 'latitude', 'longitude', 'median_price', 'count']
                        
                        fig = px.scatter_mapbox(
                            map_data,
                            lat="latitude",
                            lon="longitude",
                            color="median_price",
                            size="count",
                            hover_name=selected_geo_feature,
                            hover_data=["median_price", "count"],
                            color_continuous_scale=px.colors.sequential.Viridis,
                            size_max=25,
                            zoom=10,
                            title=f"Median Price by {selected_geo_feature.title()}",
                            labels={
                                'median_price': 'Median Price ($)',
                                'count': 'Property Count'
                            }
                        )
                        
                        fig.update_layout(mapbox_style="open-street-map", height=600)
                        st.plotly_chart(fig, use_container_width=True) 