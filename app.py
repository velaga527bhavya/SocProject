# CO2 Emission Prediction - Streamlit Web App
# Save this file as: app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="CO2 Emission Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model, scaler, and data
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('co2_model.joblib')
        scaler = joblib.load('scaler.joblib')
        df_clean = joblib.load('cleaned_data.joblib')
        return model, scaler, df_clean
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Main header
st.title("🌍 CO2 Emission Prediction System")
st.markdown("### Predict carbon dioxide emissions using Machine Learning")
st.markdown("---")

# Load resources
model, scaler, df_clean = load_model_and_data()

if model is None:
    st.error("⚠️ Model files not found! Please run the training script first.")
    st.stop()

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio("Choose a page:", 
                        ["🏠 Home", "🔮 Predict by Country", "📈 Manual Prediction", 
                         "📊 Analytics", "ℹ️ About"])

# Feature names
features = ['year', 'population', 'gdp', 'coal_co2', 'oil_co2', 
            'gas_co2', 'cement_co2', 'energy_per_capita']

# HOME PAGE
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Countries", df_clean['country'].nunique())
    with col2:
        st.metric("📅 Years Covered", f"{df_clean['year'].min()}-{df_clean['year'].max()}")
    with col3:
        st.metric("🎯 Model Accuracy", "95%+")
    
    st.markdown("---")
    st.subheader("🚀 Quick Start")
    st.write("""
    1. **Predict by Country**: Select a country from the dropdown to see predictions
    2. **Manual Prediction**: Enter custom values for detailed predictions
    3. **Analytics**: Explore trends and visualizations
    """)
    
    # Recent predictions visualization
    st.subheader("🌐 Top 10 CO2 Emitting Countries (Latest Data)")
    latest_year = df_clean['year'].max()
    top_countries = df_clean[df_clean['year'] == latest_year].nlargest(10, 'co2')
    
    fig = px.bar(top_countries, x='country', y='co2', 
                 title=f"Top 10 CO2 Emitters ({latest_year})",
                 labels={'co2': 'CO2 Emissions (million tonnes)', 'country': 'Country'},
                 color='co2', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

# PREDICT BY COUNTRY PAGE
elif page == "🔮 Predict by Country":
    st.header("🔮 Predict CO2 Emissions by Country")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Country selection
        countries = sorted(df_clean['country'].unique())
        selected_country = st.selectbox("Select a country:", countries, index=countries.index('United States') if 'United States' in countries else 0)
    
    with col2:
        # Year selection
        country_data = df_clean[df_clean['country'] == selected_country]
        available_years = sorted(country_data['year'].unique(), reverse=True)
        selected_year = st.selectbox("Select year:", available_years)
    
    if st.button("🎯 Predict Emissions", type="primary"):
        # Get data for selected country and year
        data_point = country_data[country_data['year'] == selected_year].iloc[0]
        
        # Prepare features
        X_input = data_point[features].values.reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        actual = data_point['co2']
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Predicted CO2", f"{prediction:,.2f} MT", 
                     delta=f"{prediction - actual:,.2f} MT")
        with col2:
            st.metric("📈 Actual CO2", f"{actual:,.2f} MT")
        with col3:
            accuracy = (1 - abs(prediction - actual) / actual) * 100
            st.metric("✅ Accuracy", f"{accuracy:.2f}%")
        
        # Visualization
        fig = go.Figure(data=[
            go.Bar(name='Actual', x=['CO2 Emissions'], y=[actual], marker_color='lightblue'),
            go.Bar(name='Predicted', x=['CO2 Emissions'], y=[prediction], marker_color='coral')
        ])
        fig.update_layout(title=f"CO2 Emissions: {selected_country} ({selected_year})",
                         yaxis_title="CO2 Emissions (million tonnes)",
                         barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature values
        with st.expander("📋 View Input Features"):
            feature_df = pd.DataFrame({
                'Feature': features,
                'Value': [f"{data_point[f]:,.2f}" for f in features]
            })
            st.dataframe(feature_df, use_container_width=True)

# MANUAL PREDICTION PAGE
elif page == "📈 Manual Prediction":
    st.header("📈 Manual CO2 Emission Prediction")
    st.write("Enter custom values to predict CO2 emissions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2023)
        population = st.number_input("Population", min_value=0, value=331000000, step=1000000)
        gdp = st.number_input("GDP (billion $)", min_value=0.0, value=21000.0, step=100.0)
        coal_co2 = st.number_input("Coal CO2 (MT)", min_value=0.0, value=1000.0, step=10.0)
    
    with col2:
        oil_co2 = st.number_input("Oil CO2 (MT)", min_value=0.0, value=2000.0, step=10.0)
        gas_co2 = st.number_input("Gas CO2 (MT)", min_value=0.0, value=1500.0, step=10.0)
        cement_co2 = st.number_input("Cement CO2 (MT)", min_value=0.0, value=100.0, step=1.0)
        energy_per_capita = st.number_input("Energy per Capita (kWh)", min_value=0.0, value=80000.0, step=1000.0)
    
    if st.button("🔮 Predict", type="primary"):
        # Prepare input
        input_data = np.array([[year, population, gdp, coal_co2, oil_co2, 
                               gas_co2, cement_co2, energy_per_capita]])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # Display result
        st.success(f"### Predicted CO2 Emission: {prediction:,.2f} million tonnes")
        
        # Breakdown visualization
        breakdown = {
            'Source': ['Coal', 'Oil', 'Gas', 'Cement'],
            'Emissions': [coal_co2, oil_co2, gas_co2, cement_co2]
        }
        df_breakdown = pd.DataFrame(breakdown)
        
        fig = px.pie(df_breakdown, values='Emissions', names='Source',
                    title='CO2 Emission Sources Breakdown')
        st.plotly_chart(fig, use_container_width=True)

# ANALYTICS PAGE
elif page == "📊 Analytics":
    st.header("📊 CO2 Emissions Analytics")
    
    # Country selection for trends
    countries_for_trend = st.multiselect(
        "Select countries to compare:",
        sorted(df_clean['country'].unique()),
        default=['United States', 'China', 'India'][:min(3, df_clean['country'].nunique())]
    )
    
    if countries_for_trend:
        # Time series data
        trend_data = df_clean[df_clean['country'].isin(countries_for_trend)]
        
        # Line chart
        fig = px.line(trend_data, x='year', y='co2', color='country',
                     title='CO2 Emissions Over Time',
                     labels={'co2': 'CO2 Emissions (MT)', 'year': 'Year'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("📈 Statistics")
        for country in countries_for_trend:
            country_stats = trend_data[trend_data['country'] == country]['co2']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"{country} - Average", f"{country_stats.mean():,.2f} MT")
            with col2:
                st.metric("Maximum", f"{country_stats.max():,.2f} MT")
            with col3:
                st.metric("Minimum", f"{country_stats.min():,.2f} MT")
            with col4:
                st.metric("Latest", f"{country_stats.iloc[-1]:,.2f} MT")

# ABOUT PAGE
elif page == "ℹ️ About":
    st.header("ℹ️ About This Application")
    
    st.write("""
    ### 🌍 CO2 Emission Prediction System
    
    This application uses **Machine Learning** to predict CO2 emissions based on various factors.
    
    #### 🎯 Features Used:
    - Year
    - Population
    - GDP (Gross Domestic Product)
    - Coal, Oil, Gas CO2 emissions
    - Cement CO2 emissions
    - Energy per capita
    
    #### 🤖 Model Information:
    - **Algorithm**: Random Forest Regressor
    - **Accuracy**: 95%+ R² Score
    - **Training Data**: Our World in Data (OWID) CO2 Dataset
    
    #### 📊 Dataset:
    - **Source**: Our World in Data
    - **Countries**: 200+
    - **Time Period**: 1750-2023
    
    #### 👨‍💻 Technology Stack:
    - Python
    - Scikit-learn
    - Streamlit
    - Plotly
    - Joblib
    
    #### 📧 Contact:
    For questions or feedback, please contact the developer.
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Use the sidebar to navigate between different features!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ using Streamlit | Data from Our World in Data</p>
    </div>
""", unsafe_allow_html=True)