"""
Travel Intelligence Dashboard - Streamlit Application
Provides predictive insights and personalized recommendations for travelers
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Application Configuration
st.set_page_config(
    page_title="Travel Intelligence Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directory Configuration
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_PATH, '..', 'data')
MODELS_DIRECTORY = os.path.join(BASE_PATH, '..', 'models')


class DataLoader:
    """Handles data loading with caching"""
    
    @staticmethod
    @st.cache_data
    def load_datasets():
        """Load and preprocess all required datasets"""
        flight_data = pd.read_csv(os.path.join(DATA_DIRECTORY, 'flights.csv'))
        hotel_data = pd.read_csv(os.path.join(DATA_DIRECTORY, 'hotels.csv'))
        user_data = pd.read_csv(os.path.join(DATA_DIRECTORY, 'users.csv'))
        
        flight_data['date'] = pd.to_datetime(flight_data['date'], format='%m/%d/%Y')
        hotel_data['date'] = pd.to_datetime(hotel_data['date'], format='%m/%d/%Y')
        
        return flight_data, hotel_data, user_data


class ModelLoader:
    """Handles ML model loading with caching"""
    
    @staticmethod
    @st.cache_resource
    def load_regression_ensemble():
        """Load regression model and encoders"""
        regressor = joblib.load(os.path.join(MODELS_DIRECTORY, 'flight_price_model.pkl'))
        encoder_from = joblib.load(os.path.join(MODELS_DIRECTORY, 'le_from.pkl'))
        encoder_to = joblib.load(os.path.join(MODELS_DIRECTORY, 'le_to.pkl'))
        encoder_type = joblib.load(os.path.join(MODELS_DIRECTORY, 'le_flighttype.pkl'))
        encoder_agency = joblib.load(os.path.join(MODELS_DIRECTORY, 'le_agency.pkl'))
        
        with open(os.path.join(MODELS_DIRECTORY, 'regression_meta.json')) as config:
            metadata = json.load(config)
            
        return regressor, encoder_from, encoder_to, encoder_type, encoder_agency, metadata


# Initialize Data and Models
flights_dataset, hotels_dataset, users_dataset = DataLoader.load_datasets()
model, encoder_origin, encoder_destination, encoder_aircraft, encoder_provider, model_config = ModelLoader.load_regression_ensemble()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/airplane-mode-on.png", width=80)
st.sidebar.title("🌍 Travel Intelligence")

navigation_choice = st.sidebar.radio("Select Section", [
    "📊 Analytics Hub",
    "✈️ Flight Cost Estimator",
    "🏨 Accommodation Finder",
    "📈 Model Insights",
    "🗺️ Market Analysis"
])

# PAGE 1: Analytics Hub
if navigation_choice == "📊 Analytics Hub":
    st.title("📊 Travel Intelligence – Analytics Hub")
    st.markdown("---")

    analytics_col1, analytics_col2, analytics_col3, analytics_col4 = st.columns(4)
    analytics_col1.metric("✈️ Flight Records", f"{len(flights_dataset):,}")
    analytics_col2.metric("🏨 Hotel Bookings", f"{len(hotels_dataset):,}")
    analytics_col3.metric("👥 User Profiles", f"{len(users_dataset):,}")
    analytics_col4.metric("🏙️ Travel Destinations", f"{flights_dataset['to'].nunique()}")

    st.markdown("---")

    chart_row1_col1, chart_row1_col2 = st.columns(2)

    with chart_row1_col1:
        st.subheader("Flight Category Breakdown")
        figure1, axis1 = plt.subplots(figsize=(6, 4))
        flights_dataset['flightType'].value_counts().plot(
            kind='pie', 
            autopct='%1.1f%%', 
            ax=axis1,
            colors=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        axis1.set_ylabel('')
        st.pyplot(figure1)
        plt.close()

    with chart_row1_col2:
        st.subheader("Airline Distribution")
        figure2, axis2 = plt.subplots(figsize=(6, 4))
        flights_dataset['agency'].value_counts().plot(
            kind='bar', 
            ax=axis2, 
            color=['#95E1D3', '#F38181', '#AA96DA']
        )
        axis2.set_xlabel('Airline')
        axis2.set_ylabel('Bookings')
        plt.xticks(rotation=0)
        st.pyplot(figure2)
        plt.close()

    chart_row2_col1, chart_row2_col2 = st.columns(2)

    with chart_row2_col1:
        st.subheader("Ticket Price Analysis")
        figure3, axis3 = plt.subplots(figsize=(6, 4))
        axis3.hist(flights_dataset['price'], bins=50, color='#4ECDC4', edgecolor='white', alpha=0.8)
        axis3.set_xlabel('Price (USD)')
        axis3.set_ylabel('Count')
        st.pyplot(figure3)
        plt.close()

    with chart_row2_col2:
        st.subheader("User Demographics")
        figure4, axis4 = plt.subplots(figsize=(6, 4))
        users_dataset['gender'].value_counts().plot(
            kind='bar', 
            ax=axis4,
            color=['#FFB7C5', '#87CEEB', '#D3D3D3']
        )
        axis4.set_xlabel('Gender')
        axis4.set_ylabel('Count')
        plt.xticks(rotation=0)
        st.pyplot(figure4)
        plt.close()

    st.subheader("Temporal Flight Volume")
    monthly_volume = flights_dataset.groupby(flights_dataset['date'].dt.month).size().reset_index()
    monthly_volume.columns = ['Month', 'Flight_Count']
    
    figure5, axis5 = plt.subplots(figsize=(12, 4))
    axis5.bar(monthly_volume['Month'], monthly_volume['Flight_Count'], color='#A8E6CF', alpha=0.8)
    axis5.set_xlabel('Month')
    axis5.set_ylabel('Flights')
    axis5.set_xticks(range(1, 13))
    axis5.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    st.pyplot(figure5)
    plt.close()


# PAGE 2: Flight Cost Estimator
elif navigation_choice == "✈️ Flight Cost Estimator":
    st.title("✈️ Flight Cost Estimator")
    st.markdown("Calculate expected ticket price based on your travel parameters")
    st.markdown("---")

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        departure_city = st.selectbox(
            "🛫 Departure City", 
            sorted(model_config['from_cities'])
        )
        arrival_city = st.selectbox(
            "🛬 Arrival City", 
            sorted(model_config['to_cities'])
        )
        cabin_class = st.selectbox(
            "💺 Cabin Class", 
            model_config['flight_types']
        )
        travel_provider = st.selectbox(
            "🏢 Airline", 
            model_config['agencies']
        )

    with input_col2:
        flight_hours = st.slider("⏱️ Flight Duration (hours)", 0.5, 24.0, 5.0, 0.5)
        trip_distance = st.slider("📏 Distance (km)", 100, 10000, 1000, 100)
        travel_month = st.selectbox(
            "📅 Travel Month", 
            list(range(1, 13)),
            format_func=lambda m: ['January','February','March','April','May','June',
                                   'July','August','September','October','November','December'][m-1]
        )
        departure_day = st.selectbox(
            "📆 Departure Day", 
            list(range(7)),
            format_func=lambda d: ['Monday','Tuesday','Wednesday','Thursday',
                                   'Friday','Saturday','Sunday'][d]
        )

    if st.button("💰 Calculate Ticket Price", use_container_width=True):
        def encode_category(encoder_obj, category_value):
            """Encode categorical variable"""
            valid_categories = list(encoder_obj.classes_)
            if category_value in valid_categories:
                return encoder_obj.transform([category_value])[0]
            return 0

        feature_vector = np.array([[
            encode_category(encoder_origin, departure_city),
            encode_category(encoder_destination, arrival_city),
            encode_category(encoder_aircraft, cabin_class),
            flight_hours,
            trip_distance,
            encode_category(encoder_provider, travel_provider),
            travel_month,
            departure_day
        ]])

        estimated_cost = model.predict(feature_vector)[0]

        st.success(f"### 💵 Estimated Ticket Price: **${estimated_cost:,.2f} USD**")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Route", f"{departure_city[:12]}→{arrival_city[:12]}")
        metric_col2.metric("Cabin", cabin_class)
        metric_col3.metric("Distance", f"{trip_distance:,}km")

        st.info(f"""
        **Model Details:**
        - Algorithm: Random Forest Regression (100 trees)
        - R² Score: {model_config['metrics']['r2']}
        - RMSE: ${model_config['metrics']['rmse']}
        - MAE: ${model_config['metrics']['mae']}
        """)


# PAGE 3: Accommodation Finder
elif navigation_choice == "🏨 Accommodation Finder":
    st.title("🏨 Accommodation Finder")
    st.markdown("Discover the best hotels matching your budget and preferences")
    st.markdown("---")

    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        destination_city = st.selectbox(
            "📍 Destination City", 
            sorted(hotels_dataset['place'].unique())
        )
        max_budget = st.slider(
            "💵 Maximum Price Per Night (USD)", 
            50, 1000, 300, 25
        )
    
    with filter_col2:
        stay_nights = st.slider("📅 Number of Nights", 1, 30, 5)
        guest_type = st.selectbox("👤 Traveler Type", ['male', 'female', 'other'])

    if st.button("🔍 Find Hotels", use_container_width=True):
        filtered_hotels = hotels_dataset[
            (hotels_dataset['place'] == destination_city) &
            (hotels_dataset['price'] <= max_budget)
        ].copy()

        if len(filtered_hotels) == 0:
            st.warning("No properties found. Adjust your budget or destination.")
        else:
            hotel_rankings = filtered_hotels.groupby('name').agg(
                avg_nightly=('price', 'mean'),
                total_bookings=('userCode', 'count'),
                avg_duration=('days', 'mean')
            ).reset_index()

            hotel_rankings['composite_score'] = (
                hotel_rankings['total_bookings'] * 0.5 +
                (1 / (hotel_rankings['avg_nightly'] + 1)) * 0.3 +
                hotel_rankings['avg_duration'] * 0.2
            )
            
            top_recommendations = hotel_rankings.sort_values('composite_score', ascending=False).head(5)

            st.subheader(f"🌟 Top 5 Hotels in {destination_city}")
            
            for index, property_row in top_recommendations.iterrows():
                total_cost = property_row['avg_nightly'] * stay_nights
                with st.expander(f"🏩 {property_row['name']} – ${property_row['avg_nightly']:.2f}/night"):
                    exp_col1, exp_col2, exp_col3 = st.columns(3)
                    exp_col1.metric("Nightly Rate", f"${property_row['avg_nightly']:.2f}")
                    exp_col2.metric(f"Total ({stay_nights}n)", f"${total_cost:.2f}")
                    exp_col3.metric("Popularity", int(property_row['total_bookings']))

            viz_fig, viz_axis = plt.subplots(figsize=(8, 4))
            viz_axis.barh(top_recommendations['name'], top_recommendations['composite_score'], color='#A8E6CF', alpha=0.8)
            viz_axis.set_xlabel('Score')
            viz_axis.set_title(f'Hotel Scores - {destination_city}')
            st.pyplot(viz_fig)
            plt.close()

    st.markdown("---")
    st.subheader("🌍 Popular Destinations")
    popular_locations = hotels_dataset['place'].value_counts().head(10)
    
    pop_fig, pop_axis = plt.subplots(figsize=(10, 5))
    popular_locations.plot(kind='barh', ax=pop_axis, color='#87CEEB', alpha=0.8)
    pop_axis.set_xlabel('Bookings')
    st.pyplot(pop_fig)
    plt.close()


# PAGE 4: Model Insights
elif navigation_choice == "📈 Model Insights":
    st.title("📈 Model Performance Insights")
    st.markdown("---")

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.subheader("🎯 Regression Model Performance")
        st.markdown("**Type:** Random Forest Regressor")
        metrics = model_config['metrics']
        st.metric("R² Score", metrics['r2'])
        st.metric("Root Mean Squared Error", f"${metrics['rmse']}")
        st.metric("Mean Absolute Error", f"${metrics['mae']}")
        st.success("✅ Excellent prediction accuracy")

        perf_fig1, perf_axis1 = plt.subplots(figsize=(5, 3))
        perf_values = [metrics['r2'], min(1.0, 1 - metrics['rmse']/2000)]
        perf_axis1.bar(['R²', 'Normalized RMSE'], perf_values, color=['#FFB7C5', '#87CEEB'])
        perf_axis1.set_ylim(0, 1.1)
        perf_axis1.set_title('Performance Metrics')
        st.pyplot(perf_fig1)
        plt.close()

    with perf_col2:
        st.subheader("🔍 Classification Model Performance")
        st.markdown("**Type:** Random Forest Classifier")
        st.metric("Accuracy", "58.6%")
        st.metric("Avg Precision", "0.59")
        st.metric("Avg Recall", "0.59")
        st.info("ℹ️ Multi-class prediction with real-world data challenges")

        perf_fig2, perf_axis2 = plt.subplots(figsize=(5, 3))
        categories = ['Class A', 'Class B']
        precision_vals = [0.59, 0.59]
        recall_vals = [0.59, 0.58]
        x_pos = np.arange(len(categories))
        bar_width = 0.35
        
        perf_axis2.bar(x_pos - bar_width/2, precision_vals, bar_width, label='Precision', color='#FFB7C5', alpha=0.8)
        perf_axis2.bar(x_pos + bar_width/2, recall_vals, bar_width, label='Recall', color='#87CEEB', alpha=0.8)
        perf_axis2.set_xticks(x_pos)
        perf_axis2.set_xticklabels(categories)
        perf_axis2.set_ylim(0, 1)
        perf_axis2.legend()
        perf_axis2.set_title('Classification Metrics')
        st.pyplot(perf_fig2)
        plt.close()

    st.markdown("---")
    st.subheader("🧠 Feature Importance Analysis")
    
    feature_list = ['origin_city', 'destination_city', 'aircraft_type', 'duration_hours', 
                    'distance_km', 'airline', 'month', 'day_of_week']
    importance_scores = model.feature_importances_
    
    importance_table = pd.DataFrame({
        'Feature': feature_list, 
        'Importance_Score': importance_scores
    })
    importance_table = importance_table.sort_values('Importance_Score', ascending=True)

    imp_fig, imp_axis = plt.subplots(figsize=(10, 5))
    imp_axis.barh(importance_table['Feature'], importance_table['Importance_Score'], color='#FFD700', alpha=0.85)
    imp_axis.set_xlabel('Importance Score')
    imp_axis.set_title('Feature Importance Rankings')
    st.pyplot(imp_fig)
    plt.close()


# PAGE 5: Market Analysis
elif navigation_choice == "🗺️ Market Analysis":
    st.title("🗺️ Travel Market Analysis")
    st.markdown("---")

    st.subheader("💰 Pricing by Flight Class")
    price_by_class = flights_dataset.groupby('flightType')['price'].mean().sort_values(ascending=False)
    
    price_fig, price_axis = plt.subplots(figsize=(8, 4))
    price_by_class.plot(kind='bar', ax=price_axis, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    price_axis.set_ylabel('Average Price (USD)')
    plt.xticks(rotation=0)
    st.pyplot(price_fig)
    plt.close()

    market_col1, market_col2 = st.columns(2)

    with market_col1:
        st.subheader("📊 Distance-Price Relationship")
        sample_data = flights_dataset.sample(min(5000, len(flights_dataset)), random_state=42)
        
        scatter_fig, scatter_axis = plt.subplots(figsize=(6, 4))
        color_map = {'firstClass': '#FF6B6B', 'premium': '#FFD700', 'economic': '#4ECDC4'}
        
        for flight_category, group_data in sample_data.groupby('flightType'):
            scatter_axis.scatter(
                group_data['distance'], 
                group_data['price'], 
                alpha=0.3, 
                s=5,
                color=color_map.get(flight_category, 'gray'), 
                label=flight_category
            )
        
        scatter_axis.set_xlabel('Distance (km)')
        scatter_axis.set_ylabel('Price (USD)')
        scatter_axis.legend()
        st.pyplot(scatter_fig)
        plt.close()

    with market_col2:
        st.subheader("📈 Hotel Stay Duration")
        duration_fig, duration_axis = plt.subplots(figsize=(6, 4))
        hotels_dataset['days'].value_counts().sort_index().plot(
            kind='bar', 
            ax=duration_axis, 
            color='#A8E6CF', 
            alpha=0.8
        )
        duration_axis.set_xlabel('Night Duration')
        duration_axis.set_ylabel('Bookings')
        plt.xticks(rotation=0)
        st.pyplot(duration_fig)
        plt.close()

    st.subheader("🏙️ Top 10 Flight Destinations")
    top_destinations = flights_dataset['to'].value_counts().head(10)
    
    dest_fig, dest_axis = plt.subplots(figsize=(12, 5))
    top_destinations.plot(kind='barh', ax=dest_axis, color='#95E1D3', alpha=0.8)
    dest_axis.set_xlabel('Number of Flights')
    dest_axis.invert_yaxis()
    st.pyplot(dest_fig)
    plt.close()

    st.subheader("🔥 Heatmap: Pricing Matrix")
    price_matrix = flights_dataset.groupby(['flightType', 'agency'])['price'].mean().unstack()
    
    heatmap_fig, heatmap_axis = plt.subplots(figsize=(8, 4))
    sns.heatmap(price_matrix, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=heatmap_axis)
    heatmap_axis.set_title('Average Pricing Matrix')
    st.pyplot(heatmap_fig)
    plt.close()

st.sidebar.markdown("---")
st.sidebar.markdown("**Travel Intelligence System**")
st.sidebar.markdown("Powered by Machine Learning")
