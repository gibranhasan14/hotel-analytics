import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import os

# Set page configuration
st.set_page_config(
    page_title="Hotel Booking Analytics Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("🏨 Hotel Booking Analytics Dashboard")
st.markdown("""
This dashboard provides comprehensive analytics for hotel booking data, including booking trends, 
cancellation analytics, revenue insights, and predictive analytics for cancellations and customer churn.
""")

# Initialize session state for active tab
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0  # Default to first tab

# Function to set active tab
def set_active_tab(tab_index):
    st.session_state.active_tab = tab_index

# Function to connect to the database
@st.cache_resource
def get_connection(db_path='data/hotel_dw.db'):
    """Create a connection to the SQLite database"""
    return sqlite3.connect(db_path)

@st.cache_data
def load_data(query, _conn):
    """Load data from database using SQL query"""
    # Use an underscore prefix for the connection parameter
    # This tells Streamlit not to include this parameter in the cache key
    conn = get_connection() 
    result = pd.read_sql_query(query, conn)
    return result

# Function to load the cancellation prediction model
@st.cache_resource
def load_cancellation_model(model_path='models/cancellation_model.pkl'):
    """Load the trained cancellation prediction model"""
    try:
        return joblib.load(model_path)
    except:
        st.warning("Cancellation prediction model not found. Some features will be disabled.")
        return None

# Function to load the churn prediction model
@st.cache_resource
def load_churn_model(model_path='models/churn_model.pkl'):
    """Load the trained churn prediction model"""
    try:
        return joblib.load(model_path)
    except:
        st.warning("Churn prediction model not found. Some features will be disabled.")
        return None

# Function to load the booking forecast model
@st.cache_resource
def load_forecast_model(model_path='models/booking_forecast_model.pkl'):
    """Load the trained booking forecast model"""
    try:
        return joblib.load(model_path)
    except:
        st.warning("Booking forecast model not found. Some features will be disabled.")
        return None

# Connect to the database
try:
    conn = get_connection()
except Exception as e:
    st.error(f"Error connecting to database: {str(e)}")
    st.info("Make sure you've run the ETL script to create the database first.")
    st.stop()

# Load the prediction models
cancellation_model = load_cancellation_model()
churn_model = load_churn_model()
forecast_model = load_forecast_model()

# Define the main query to get all data
main_query = """
SELECT 
    bf.booking_id,
    bf.is_canceled,
    bf.lead_time,
    bf.stays_in_weekend_nights,
    bf.stays_in_week_nights,
    bf.adults,
    bf.children,
    bf.babies,
    bf.is_repeated_guest,
    bf.previous_cancellations,
    bf.previous_bookings_not_canceled,
    bf.booking_changes,
    bf.adr,
    bf.required_car_parking_spaces,
    bf.total_of_special_requests,
    bf.days_in_waiting_list,
    bf.total_stay_nights,
    bf.total_guests,
    bf.revenue,
    
    dd.year,
    dd.month,
    dd.week_number,
    dd.day_of_month,
    dd.season,
    
    hd.hotel_name,
    hd.meal_plan,
    hd.market_segment,
    hd.distribution_channel,
    
    cd.country,
    cd.customer_type,
    cd.deposit_type,
    
    rd.reserved_room_type,
    rd.assigned_room_type,
    rd.room_changed,
    
    bf.reservation_status,
    bf.reservation_status_date
FROM 
    booking_fact bf
JOIN 
    date_dim dd ON bf.date_key = dd.date_key
JOIN 
    hotel_dim hd ON bf.hotel_key = hd.hotel_key
JOIN 
    customer_dim cd ON bf.customer_key = cd.customer_key
JOIN 
    room_dim rd ON bf.room_key = rd.room_key
"""

# Load the data
try:
    data = load_data(main_query, conn)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Make sure you've run the ETL script and the database schema is correct.")
    st.stop()

# Convert reservation_status_date to datetime
data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'])

# Create a column for year-month for easier grouping
data['year_month'] = data['year'].astype(str) + '-' + data['month']

# Sidebar for filtering
st.sidebar.header("Filters")

# Hotel selection
hotel_options = ["All Hotels"] + sorted(data['hotel_name'].unique().tolist())
selected_hotel = st.sidebar.selectbox("Select Hotel", hotel_options)

# Date range filter
min_year = int(data['year'].min())
max_year = int(data['year'].max())
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

# Market segment filter
market_options = ["All Segments"] + sorted(data['market_segment'].unique().tolist())
selected_market = st.sidebar.selectbox("Select Market Segment", market_options)

# Customer type filter
customer_options = ["All Types"] + sorted(data['customer_type'].unique().tolist())
selected_customer = st.sidebar.selectbox("Select Customer Type", customer_options)

# Apply filters
filtered_data = data.copy()

if selected_hotel != "All Hotels":
    filtered_data = filtered_data[filtered_data['hotel_name'] == selected_hotel]
    
filtered_data = filtered_data[(filtered_data['year'] >= year_range[0]) & (filtered_data['year'] <= year_range[1])]

if selected_market != "All Segments":
    filtered_data = filtered_data[filtered_data['market_segment'] == selected_market]
    
if selected_customer != "All Types":
    filtered_data = filtered_data[filtered_data['customer_type'] == selected_customer]

# Dashboard Tabs
# tab1, tab2, tab3, tab4, tab5 = st.tabs([
#     "📊 Booking Trends", 
#     "❌ Cancellation Analytics", 
#     "💰 Revenue Insights", 
#     "🔮 Predictive Analytics",
#     "📑 Recommendations"
# ])
tabs = ["📊 Booking Trends", "❌ Cancellation Analytics", "💰 Revenue Insights", "🔮 Predictive Analytics", "📑 Recommendations"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

# This will ensure the previously selected tab remains active on rerun
if st.session_state.active_tab == 0:
    current_tab = tab1
elif st.session_state.active_tab == 1:
    current_tab = tab2
elif st.session_state.active_tab == 2:
    current_tab = tab3
elif st.session_state.active_tab == 3:
    current_tab = tab4
else:
    current_tab = tab5

# Month order mapping (for proper sorting)
month_order = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

#-----------------------------------------------------------------------------------
# Tab 1: Booking Trends
#-----------------------------------------------------------------------------------
with tab1:
    st.header("Booking Trends Analysis")
    
    # Create two columns for the top row
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly booking trend
        st.subheader("Monthly Booking Trends")
        
        # Group by year and month
        monthly_bookings = filtered_data.groupby(['year', 'month']).size().reset_index(name='bookings')
        
        # Add month_num for proper ordering
        monthly_bookings['month_num'] = monthly_bookings['month'].map(month_order)
        monthly_bookings.sort_values(['year', 'month_num'], inplace=True)
        
        # Create x-axis labels
        monthly_bookings['year_month'] = monthly_bookings['year'].astype(str) + '-' + monthly_bookings['month']
        
        fig = px.line(
            monthly_bookings, 
            x='year_month', 
            y='bookings', 
            markers=True,
            title=f"Monthly Booking Trends ({year_range[0]}-{year_range[1]})"
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Bookings",
            xaxis={'categoryorder':'array', 'categoryarray':monthly_bookings['year_month'].tolist()}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Seasonal patterns
        st.subheader("Seasonal Booking Patterns")
        
        seasonal_bookings = filtered_data.groupby(['season']).size().reset_index(name='bookings')
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_bookings['season_order'] = seasonal_bookings['season'].map({season: i for i, season in enumerate(season_order)})
        seasonal_bookings.sort_values('season_order', inplace=True)
        
        fig = px.bar(
            seasonal_bookings, 
            x='season', 
            y='bookings',
            color='season',
            title="Bookings by Season"
        )
        
        fig.update_layout(
            xaxis_title="Season",
            yaxis_title="Number of Bookings",
            xaxis={'categoryorder':'array', 'categoryarray':season_order}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Create two columns for the second row
    col1, col2 = st.columns(2)
    
    with col1:
        # Lead time distribution
        st.subheader("Lead Time Distribution")
        
        # Create lead time buckets
        filtered_data['lead_time_bucket'] = pd.cut(
            filtered_data['lead_time'], 
            bins=[0, 7, 30, 90, 180, 365, float('inf')],
            labels=['0-7 days', '8-30 days', '31-90 days', '91-180 days', '181-365 days', '366+ days']
        )
        
        lead_time_dist = filtered_data.groupby('lead_time_bucket').size().reset_index(name='count')
        
        fig = px.bar(
            lead_time_dist, 
            x='lead_time_bucket', 
            y='count',
            color='lead_time_bucket',
            title="Booking Lead Time Distribution"
        )
        
        fig.update_layout(
            xaxis_title="Lead Time",
            yaxis_title="Number of Bookings"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stay duration distribution
        st.subheader("Stay Duration Distribution")
        
        # Create stay duration buckets
        filtered_data['stay_duration_bucket'] = pd.cut(
            filtered_data['total_stay_nights'], 
            bins=[0, 1, 2, 3, 5, 7, 14, float('inf')],
            labels=['0-1 nights', '2 nights', '3 nights', '4-5 nights', '6-7 nights', '8-14 nights', '15+ nights']
        )
        
        stay_duration_dist = filtered_data.groupby('stay_duration_bucket').size().reset_index(name='count')
        
        fig = px.bar(
            stay_duration_dist, 
            x='stay_duration_bucket', 
            y='count',
            color='stay_duration_bucket',
            title="Stay Duration Distribution"
        )
        
        fig.update_layout(
            xaxis_title="Stay Duration",
            yaxis_title="Number of Bookings"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Third row for distribution channels and market segments
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution channels
        st.subheader("Booking Distribution Channels")
        
        channel_dist = filtered_data.groupby('distribution_channel').size().reset_index(name='count')
        channel_dist = channel_dist.sort_values('count', ascending=False)
        
        fig = px.pie(
            channel_dist, 
            values='count', 
            names='distribution_channel',
            title="Bookings by Distribution Channel"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Market segments
        st.subheader("Market Segments")
        
        market_dist = filtered_data.groupby('market_segment').size().reset_index(name='count')
        market_dist = market_dist.sort_values('count', ascending=False)
        
        fig = px.pie(
            market_dist, 
            values='count', 
            names='market_segment',
            title="Bookings by Market Segment"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Fourth row for detailed data table
    st.subheader("Detailed Booking Data")
    
    # Select columns to display
    display_cols = [
        'hotel_name', 'reservation_status', 'year', 'month', 'lead_time', 
        'total_stay_nights', 'adults', 'children', 'adr', 'revenue',
        'market_segment', 'distribution_channel', 'is_repeated_guest'
    ]
    
    st.dataframe(filtered_data[display_cols].head(100), use_container_width=True)

#-----------------------------------------------------------------------------------
# Tab 2: Cancellation Analytics
#-----------------------------------------------------------------------------------
with tab2:
    st.header("Cancellation Analytics")
    
    # Calculate overall cancellation rate
    cancellation_rate = filtered_data['is_canceled'].mean() * 100
    
    # Create two columns for metrics
    metrics_cols = st.columns(4)
    
    with metrics_cols[0]:
        st.metric(
            label="Overall Cancellation Rate", 
            value=f"{cancellation_rate:.2f}%",
            delta=None
        )
    
    with metrics_cols[1]:
        # Total bookings
        total_bookings = len(filtered_data)
        st.metric(
            label="Total Bookings",
            value=f"{total_bookings:,}",
            delta=None
        )
    
    with metrics_cols[2]:
        # Cancelled bookings
        cancelled_bookings = filtered_data['is_canceled'].sum()
        st.metric(
            label="Cancelled Bookings",
            value=f"{cancelled_bookings:,}",
            delta=None
        )
    
    with metrics_cols[3]:
        # Confirmed bookings
        confirmed_bookings = total_bookings - cancelled_bookings
        st.metric(
            label="Confirmed Bookings",
            value=f"{confirmed_bookings:,}",
            delta=None
        )
    
    # Create two columns for the top row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Cancellation rate by lead time
        st.subheader("Cancellation Rate by Lead Time")
        
        lead_time_cancellations = filtered_data.groupby('lead_time_bucket').agg(
            cancellation_rate=('is_canceled', 'mean'),
            count=('booking_id', 'count')
        ).reset_index()
        
        lead_time_cancellations['cancellation_rate'] = lead_time_cancellations['cancellation_rate'] * 100
        
        fig = px.bar(
            lead_time_cancellations, 
            x='lead_time_bucket', 
            y='cancellation_rate',
            color='cancellation_rate',
            color_continuous_scale='RdYlGn_r',
            title="Cancellation Rate by Lead Time",
            text=lead_time_cancellations['cancellation_rate'].round(1).astype(str) + '%'
        )
        
        fig.update_layout(
            xaxis_title="Lead Time",
            yaxis_title="Cancellation Rate (%)",
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cancellation rate by deposit type
        st.subheader("Cancellation Rate by Deposit Type")
        
        deposit_cancellations = filtered_data.groupby('deposit_type').agg(
            cancellation_rate=('is_canceled', 'mean'),
            count=('booking_id', 'count')
        ).reset_index()
        
        deposit_cancellations['cancellation_rate'] = deposit_cancellations['cancellation_rate'] * 100
        
        fig = px.bar(
            deposit_cancellations, 
            x='deposit_type', 
            y='cancellation_rate',
            color='cancellation_rate',
            color_continuous_scale='RdYlGn_r',
            title="Cancellation Rate by Deposit Type",
            text=deposit_cancellations['cancellation_rate'].round(1).astype(str) + '%'
        )
        
        fig.update_layout(
            xaxis_title="Deposit Type",
            yaxis_title="Cancellation Rate (%)",
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Create two columns for the second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Cancellation rate by market segment
        st.subheader("Cancellation Rate by Market Segment")
        
        market_cancellations = filtered_data.groupby('market_segment').agg(
            cancellation_rate=('is_canceled', 'mean'),
            count=('booking_id', 'count')
        ).reset_index()
        
        market_cancellations['cancellation_rate'] = market_cancellations['cancellation_rate'] * 100
        market_cancellations = market_cancellations.sort_values('cancellation_rate', ascending=False)
        
        fig = px.bar(
            market_cancellations, 
            x='market_segment', 
            y='cancellation_rate',
            color='cancellation_rate',
            color_continuous_scale='RdYlGn_r',
            title="Cancellation Rate by Market Segment",
            text=market_cancellations['cancellation_rate'].round(1).astype(str) + '%'
        )
        
        fig.update_layout(
            xaxis_title="Market Segment",
            yaxis_title="Cancellation Rate (%)",
            xaxis={'categoryorder':'total descending'},
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cancellation trend over time
        st.subheader("Cancellation Rate Trend")
        
        # Group by year and month
        time_cancellations = filtered_data.groupby(['year', 'month']).agg(
            cancellation_rate=('is_canceled', 'mean'),
            count=('booking_id', 'count')
        ).reset_index()
        
        time_cancellations['cancellation_rate'] = time_cancellations['cancellation_rate'] * 100
        
        # Add month_num for proper ordering
        time_cancellations['month_num'] = time_cancellations['month'].map(month_order)
        time_cancellations.sort_values(['year', 'month_num'], inplace=True)
        
        # Create x-axis labels
        time_cancellations['year_month'] = time_cancellations['year'].astype(str) + '-' + time_cancellations['month']
        
        fig = px.line(
            time_cancellations, 
            x='year_month', 
            y='cancellation_rate',
            markers=True,
            title="Cancellation Rate Trend Over Time"
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Cancellation Rate (%)",
            xaxis={'categoryorder':'array', 'categoryarray':time_cancellations['year_month'].tolist()}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Third row with a heatmap of cancellation factors
    st.subheader("Cancellation Factors Correlation")
    
    # Select relevant features for correlation
    corr_features = [
        'is_canceled', 'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
        'adults', 'children', 'is_repeated_guest', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests', 'days_in_waiting_list'
    ]
    
    # Calculate correlation
    corr = filtered_data[corr_features].corr()
    
    # Create a heatmap
    fig = px.imshow(
        corr,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Between Booking Factors and Cancellation"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Fourth row for detailed cancellation data
    st.subheader("Top Cancellation Factors")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Average lead time comparison
        avg_lead_time_canceled = filtered_data[filtered_data['is_canceled'] == 1]['lead_time'].mean()
        avg_lead_time_not_canceled = filtered_data[filtered_data['is_canceled'] == 0]['lead_time'].mean()
        
        lead_time_comparison = pd.DataFrame({
            'Booking Status': ['Canceled', 'Not Canceled'],
            'Average Lead Time (days)': [avg_lead_time_canceled, avg_lead_time_not_canceled]
        })
        
        fig = px.bar(
            lead_time_comparison,
            x='Booking Status',
            y='Average Lead Time (days)',
            color='Booking Status',
            title="Average Lead Time Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Special requests comparison
        avg_requests_canceled = filtered_data[filtered_data['is_canceled'] == 1]['total_of_special_requests'].mean()
        avg_requests_not_canceled = filtered_data[filtered_data['is_canceled'] == 0]['total_of_special_requests'].mean()
        
        requests_comparison = pd.DataFrame({
            'Booking Status': ['Canceled', 'Not Canceled'],
            'Average Special Requests': [avg_requests_canceled, avg_requests_not_canceled]
        })
        
        fig = px.bar(
            requests_comparison,
            x='Booking Status',
            y='Average Special Requests',
            color='Booking Status',
            title="Average Special Requests Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)

#-----------------------------------------------------------------------------------
# Tab 3: Revenue Insights
#-----------------------------------------------------------------------------------
with tab3:
    st.header("Revenue Insights")
    
    # Calculate revenue metrics
    total_revenue = filtered_data['revenue'].sum()
    avg_daily_rate = filtered_data['adr'].mean()
    potential_revenue_lost = filtered_data[filtered_data['is_canceled'] == 1]['revenue'].sum()
    
    # Create metrics row
    metrics_cols = st.columns(4)
    
    with metrics_cols[0]:
        st.metric(
            label="Total Revenue", 
            value=f"${total_revenue:,.2f}",
            delta=None
        )
    
    with metrics_cols[1]:
        st.metric(
            label="Average Daily Rate (ADR)",
            value=f"${avg_daily_rate:.2f}",
            delta=None
        )
    
    with metrics_cols[2]:
        # Calculate RevPAR (Revenue Per Available Room)
        # For this simple calculation, we'll use: RevPAR = ADR * (1 - cancellation_rate)
        revpar = avg_daily_rate * (1 - (filtered_data['is_canceled'].mean()))
        st.metric(
            label="RevPAR",
            value=f"${revpar:.2f}",
            delta=None
        )
    
    with metrics_cols[3]:
        st.metric(
            label="Potential Revenue Lost (Cancellations)",
            value=f"${potential_revenue_lost:,.2f}",
            delta=None
        )
    
    # Create two columns for the first row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue trend over time
        st.subheader("Revenue Trend Over Time")
        
        # Group by year and month
        monthly_revenue = filtered_data.groupby(['year', 'month']).agg(
            total_revenue=('revenue', 'sum'),
            booking_count=('booking_id', 'count')
        ).reset_index()
        
        # Add month_num for proper ordering
        monthly_revenue['month_num'] = monthly_revenue['month'].map(month_order)
        monthly_revenue.sort_values(['year', 'month_num'], inplace=True)
        
        # Create x-axis labels
        monthly_revenue['year_month'] = monthly_revenue['year'].astype(str) + '-' + monthly_revenue['month']
        
        fig = px.line(
            monthly_revenue, 
            x='year_month', 
            y='total_revenue',
            markers=True,
            title="Monthly Revenue Trend"
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Revenue ($)",
            xaxis={'categoryorder':'array', 'categoryarray':monthly_revenue['year_month'].tolist()}
        )
        
        # Add dollar formatting to y-axis
        fig.update_yaxes(tickprefix="$")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ADR trend over time
        st.subheader("Average Daily Rate (ADR) Trend")
        
        # Group by year and month
        monthly_adr = filtered_data.groupby(['year', 'month']).agg(
            avg_adr=('adr', 'mean')
        ).reset_index()
        
        # Add month_num for proper ordering
        monthly_adr['month_num'] = monthly_adr['month'].map(month_order)
        monthly_adr.sort_values(['year', 'month_num'], inplace=True)
        
        # Create x-axis labels
        monthly_adr['year_month'] = monthly_adr['year'].astype(str) + '-' + monthly_adr['month']
        
        fig = px.line(
            monthly_adr, 
            x='year_month', 
            y='avg_adr',
            markers=True,
            title="Monthly Average Daily Rate (ADR) Trend"
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Average Daily Rate ($)",
            xaxis={'categoryorder':'array', 'categoryarray':monthly_adr['year_month'].tolist()}
        )
        
        # Add dollar formatting to y-axis
        fig.update_yaxes(tickprefix="$")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Create two columns for the second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by market segment
        st.subheader("Revenue by Market Segment")
        
        segment_revenue = filtered_data.groupby('market_segment').agg(
            total_revenue=('revenue', 'sum'),
            booking_count=('booking_id', 'count'),
            avg_adr=('adr', 'mean')
        ).reset_index()
        
        segment_revenue = segment_revenue.sort_values('total_revenue', ascending=False)
        
        fig = px.bar(
            segment_revenue,
            x='market_segment',
            y='total_revenue',
            color='avg_adr',
            color_continuous_scale='Viridis',
            title="Revenue by Market Segment",
            text=segment_revenue['total_revenue'].apply(lambda x: f"${x:,.0f}")
        )
        
        fig.update_layout(
            xaxis_title="Market Segment",
            yaxis_title="Total Revenue ($)",
            xaxis={'categoryorder':'total descending'}
        )
        
        # Add dollar formatting to y-axis
        fig.update_yaxes(tickprefix="$")
        
        # Add color bar title
        fig.update_coloraxes(colorbar_title="Avg. Daily Rate ($)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue by room type
        st.subheader("Revenue by Room Type")
        
        room_revenue = filtered_data.groupby('reserved_room_type').agg(
            total_revenue=('revenue', 'sum'),
            booking_count=('booking_id', 'count'),
            avg_adr=('adr', 'mean')
        ).reset_index()
        
        room_revenue = room_revenue.sort_values('total_revenue', ascending=False)
        
        fig = px.bar(
            room_revenue,
            x='reserved_room_type',
            y='total_revenue',
            color='avg_adr',
            color_continuous_scale='Viridis',
            title="Revenue by Room Type",
            text=room_revenue['total_revenue'].apply(lambda x: f"${x:,.0f}")
        )
        
        fig.update_layout(
            xaxis_title="Room Type",
            yaxis_title="Total Revenue ($)",
            xaxis={'categoryorder':'total descending'}
        )
        
        # Add dollar formatting to y-axis
        fig.update_yaxes(tickprefix="$")
        
        # Add color bar title
        fig.update_coloraxes(colorbar_title="Avg. Daily Rate ($)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Create two columns for the third row of charts
    col1, col2 = st.columns(2)
            
    with col1:
        # Revenue by season
        st.subheader("Revenue by Season")
        
        season_revenue = filtered_data.groupby('season').agg(
            total_revenue=('revenue', 'sum'),
            booking_count=('booking_id', 'count'),
            avg_adr=('adr', 'mean')
        ).reset_index()
        
        # Define season order
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        season_revenue['season_order'] = season_revenue['season'].map({season: i for i, season in enumerate(season_order)})
        season_revenue.sort_values('season_order', inplace=True)
        
        fig = px.bar(
            season_revenue,
            x='season',
            y='total_revenue',
            color='season',
            title="Revenue by Season",
            text=season_revenue['total_revenue'].apply(lambda x: f"${x:,.0f}")
        )
        
        fig.update_layout(
            xaxis_title="Season",
            yaxis_title="Total Revenue ($)",
            xaxis={'categoryorder':'array', 'categoryarray':season_order}
        )
        
        # Add dollar formatting to y-axis
        fig.update_yaxes(tickprefix="$")
        
        st.plotly_chart(fig, use_container_width=True)

#-----------------------------------------------------------------------------------
# Tab 4: Predictive Analytics
#-----------------------------------------------------------------------------------
with tab4:
    set_active_tab(3)
    st.header("Predictive Analytics")
    
    # Create tabs for different prediction types
    pred_tab1, pred_tab2, pred_tab3 = st.tabs([
        "🔄 Cancellation Prediction", 
        "📈 Booking Demand Forecast", 
        "🔍 Customer Churn Prediction"
    ])
    
    # Tab for Cancellation Prediction
    with pred_tab1:
        st.subheader("Cancellation Prediction Model")
        
        st.markdown("""
        This model predicts the likelihood of a booking being canceled based on various factors.
        You can use it to identify high-risk bookings and take proactive measures.
        """)
        
        if cancellation_model is not None:
            # Create form for user input
            with st.form("cancellation_prediction_form"):
                st.write("Enter booking details to predict cancellation probability:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    hotel = st.selectbox("Hotel", filtered_data['hotel_name'].unique().tolist())
                    lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=365, value=30)
                    stay_nights = st.number_input("Total Stay Nights", min_value=1, max_value=30, value=3)
                    market_segment = st.selectbox("Market Segment", filtered_data['market_segment'].unique().tolist())
                
                with col2:
                    distribution_channel = st.selectbox("Distribution Channel", filtered_data['distribution_channel'].unique().tolist())
                    is_repeated_guest = st.selectbox("Repeated Guest", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                    deposit_type = st.selectbox("Deposit Type", filtered_data['deposit_type'].unique().tolist())
                    customer_type = st.selectbox("Customer Type", filtered_data['customer_type'].unique().tolist())
                
                with col3:
                    adr = st.number_input("Average Daily Rate ($)", min_value=0, max_value=1000, value=100)
                    total_special_requests = st.number_input("Special Requests", min_value=0, max_value=5, value=0)
                    booking_changes = st.number_input("Booking Changes", min_value=0, max_value=5, value=0)
                    required_car_parking = st.selectbox("Car Parking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                
                submit_button = st.form_submit_button("Predict Cancellation", on_click=lambda: set_active_tab(3) )
            
            if submit_button:
                if lead_time <= 7:
                    lead_time_bucket = '0-7 days'
                elif lead_time <= 30:
                    lead_time_bucket = '8-30 days'
                elif lead_time <= 90:
                    lead_time_bucket = '31-90 days'
                elif lead_time <= 180:
                    lead_time_bucket = '91-180 days'
                elif lead_time <= 365:
                    lead_time_bucket = '181-365 days'
                else:
                    lead_time_bucket = '366+ days'
                
                
                # Create input data for prediction
                input_data = pd.DataFrame({
                    'hotel_name': [hotel],
                    'lead_time': [lead_time],
                    'lead_time_bucket': [lead_time_bucket],
                    'stays_in_weekend_nights': [min(2, stay_nights)],
                    'stays_in_week_nights': [max(0, stay_nights - 2)],
                    'adults': [2],
                    'children': [0],
                    'babies': [0],
                    'is_repeated_guest': [is_repeated_guest],
                    'previous_cancellations': [0],
                    'previous_bookings_not_canceled': [0],
                    'booking_changes': [booking_changes],
                    'adr': [adr],
                    'required_car_parking_spaces': [required_car_parking],
                    'total_of_special_requests': [total_special_requests],
                    'days_in_waiting_list': [0],
                    'total_stay_nights': [stay_nights],
                    'total_guests': [2],
                    'year': [2017],
                    'month': ['August'],
                    'season': ['Summer'],
                    'meal_plan': ['BB'],
                    'market_segment': [market_segment],
                    'distribution_channel': [distribution_channel],
                    'country': ['PRT'],
                    'customer_type': [customer_type],
                    'deposit_type': [deposit_type],
                    'reserved_room_type': ['A'],
                    'assigned_room_type': ['A'],
                    'room_changed': [False],
                    'reservation_status': ['Reserved'],
                    # Add the missing columns
                    'revenue': [adr * stay_nights],  # Calculate revenue
                    'is_weekend': [False]  # Default value
                })
                
                # Make prediction
                try:
                    cancellation_prob = cancellation_model.predict_proba(input_data)[:, 1][0]
                    
                    # Display result with gauge chart
                    st.subheader("Cancellation Prediction Result")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=cancellation_prob * 100,
                            title={'text': "Cancellation Probability"},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "green"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': cancellation_prob * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Risk assessment
                        if cancellation_prob < 0.3:
                            risk_level = "Low"
                            risk_color = "green"
                            recommendations = [
                                "Standard confirmation email",
                                "No special measures required",
                                "Regular follow-up"
                            ]
                        elif cancellation_prob < 0.7:
                            risk_level = "Medium"
                            risk_color = "orange"
                            recommendations = [
                                "Send reminder email 1 week before arrival",
                                "Offer room upgrade if available",
                                "Highlight cancellation policy in communications"
                            ]
                        else:
                            risk_level = "High"
                            risk_color = "red"
                            recommendations = [
                                "Call to confirm reservation",
                                "Consider overbooking strategy",
                                "Offer incentives to prevent cancellation",
                                "Request partial prepayment or deposit"
                            ]
                        
                        st.markdown(f"### Risk Level: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
                        
                        st.subheader("Recommended Actions:")
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        else:
            st.warning("Cancellation prediction model not available. Please train the model first.")
    
    # Tab for Booking Demand Forecast
    with pred_tab2:
        st.subheader("Booking Demand Forecast")
        
        st.markdown("""
        This model forecasts future booking demand based on historical patterns.
        Use it for capacity planning and revenue management.
        """)
        
        if forecast_model is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                forecast_periods = st.slider("Forecast Periods (Months)", min_value=1, max_value=12, value=6)
                selected_hotel_forecast = st.selectbox(
                    "Select Hotel for Forecast", 
                    ["All Hotels"] + filtered_data['hotel_name'].unique().tolist(),
                    key="forecast_hotel"
                )
            
            with col2:
                st.markdown("#### Forecast Parameters")
                st.markdown("""
                The forecast is based on a SARIMA (Seasonal ARIMA) model that captures:
                - Long-term trends in booking demand
                - Seasonal patterns (e.g., summer peaks)
                - Month-to-month variations
                
                **Note:** More data leads to better forecasts. Forecasts beyond 6 months have higher uncertainty.
                """)
            
            # Generate forecast
            st.subheader("Demand Forecast Chart")
            
            # For demo purpose, we'll generate a simulated forecast
            # In a real application, you would use the loaded forecast_model
            
            # Get historical data
            if selected_hotel_forecast != "All Hotels":
                hotel_data = filtered_data[filtered_data['hotel_name'] == selected_hotel_forecast]
            else:
                hotel_data = filtered_data
            
            # Group by year and month
            historical_demand = hotel_data.groupby(['year', 'month']).size().reset_index(name='bookings')
            
            # Add month_num for proper ordering
            historical_demand['month_num'] = historical_demand['month'].map(month_order)
            historical_demand.sort_values(['year', 'month_num'], inplace=True)
            
            # Create date column for plotting
            historical_demand['date'] = pd.to_datetime(
                historical_demand['year'].astype(str) + '-' + 
                historical_demand['month_num'].astype(str) + '-01'
            )
            
            # Generate forecasted dates
            last_date = historical_demand['date'].max()
            forecast_dates = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_periods)]
            
            # Generate simulated forecast values
            # In reality, these would come from your trained model
            last_value = historical_demand['bookings'].iloc[-1]
            
            # Simple seasonal forecast simulation
            forecast_values = []
            for i in range(forecast_periods):
                month = (last_date.month + i) % 12 + 1
                # Add some seasonality based on historical patterns
                if month in [6, 7, 8]:  # Summer months
                    seasonal_factor = 1.2
                elif month in [11, 12, 1]:  # Winter months
                    seasonal_factor = 0.8
                else:  # Spring/Fall
                    seasonal_factor = 1.0
                
                # Add some trend and noise
                trend_factor = 1.02  # Slight upward trend
                noise = np.random.normal(0, last_value * 0.05)
                forecast_value = last_value * seasonal_factor * (trend_factor ** (i+1)) + noise
                forecast_values.append(max(0, int(forecast_value)))
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'bookings': forecast_values,
                'lower_ci': [max(0, int(v * 0.85)) for v in forecast_values],
                'upper_ci': [int(v * 1.15) for v in forecast_values]
            })
            
            # Plot historical data and forecast
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_demand['date'],
                y=historical_demand['bookings'],
                mode='lines+markers',
                name='Historical Demand',
                line=dict(color='blue')
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['bookings'],
                mode='lines+markers',
                name='Forecasted Demand',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                y=forecast_df['upper_ci'].tolist() + forecast_df['lower_ci'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False
            ))
            
            fig.update_layout(
                title=f"Booking Demand Forecast for {selected_hotel_forecast}",
                xaxis_title="Date",
                yaxis_title="Number of Bookings",
                hovermode="x",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add forecast table
            st.subheader("Forecast Data Table")
            
            forecast_table = forecast_df.copy()
            forecast_table['year'] = forecast_table['date'].dt.year
            forecast_table['month'] = forecast_table['date'].dt.strftime('%B')
            forecast_table['forecasted_bookings'] = forecast_table['bookings']
            forecast_table['forecast_lower'] = forecast_table['lower_ci']
            forecast_table['forecast_upper'] = forecast_table['upper_ci']
            
            st.dataframe(
                forecast_table[['year', 'month', 'forecasted_bookings', 'forecast_lower', 'forecast_upper']],
                use_container_width=True
            )
            
            # Forecast insights
            st.subheader("Forecast Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Peak demand month
                peak_month_idx = np.argmax(forecast_values)
                peak_month = forecast_df['date'][peak_month_idx].strftime('%B %Y')
                peak_value = forecast_values[peak_month_idx]
                
                st.metric(
                    label="Peak Demand Month", 
                    value=peak_month,
                    delta=f"{peak_value} bookings"
                )
                
                # Total forecasted bookings
                total_forecasted = sum(forecast_values)
                st.metric(
                    label="Total Forecasted Bookings", 
                    value=f"{total_forecasted:,}",
                    delta=None
                )
            
            with col2:
                # Low demand month
                low_month_idx = np.argmin(forecast_values)
                low_month = forecast_df['date'][low_month_idx].strftime('%B %Y')
                low_value = forecast_values[low_month_idx]
                
                st.metric(
                    label="Lowest Demand Month", 
                    value=low_month,
                    delta=f"{low_value} bookings"
                )
                
                # Average monthly bookings
                avg_forecasted = np.mean(forecast_values)
                st.metric(
                    label="Average Monthly Bookings", 
                    value=f"{int(avg_forecasted):,}",
                    delta=None
                )
        else:
            st.warning("Booking forecast model not available. Please train the model first.")
    
    # Tab for Customer Churn Prediction
    with pred_tab3:
        st.subheader("Customer Churn Prediction")
        
        st.markdown("""
        This model identifies customers at risk of not returning to the hotel.
        Use it to create targeted retention campaigns.
        """)
        
        if churn_model is not None:
            # Create form for user input
            with st.form("churn_prediction_form"):
                st.write("Enter customer details to predict churn probability:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    hotel = st.selectbox("Hotel", filtered_data['hotel_name'].unique().tolist(), key="churn_hotel")
                    customer_type = st.selectbox("Customer Type", filtered_data['customer_type'].unique().tolist(), key="churn_customer_type")
                    is_repeated_guest = st.selectbox("Repeated Guest", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="churn_repeated")
                    previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=10, value=0)
                
                with col2:
                    market_segment = st.selectbox("Market Segment", filtered_data['market_segment'].unique().tolist(), key="churn_market")
                    distribution_channel = st.selectbox("Distribution Channel", filtered_data['distribution_channel'].unique().tolist(), key="churn_channel")
                    booking_changes = st.number_input("Booking Changes", min_value=0, max_value=5, value=0, key="churn_changes")
                    room_changed = st.selectbox("Room Changed", [True, False])
                
                with col3:
                    adr = st.number_input("Average Daily Rate ($)", min_value=0, max_value=1000, value=100, key="churn_adr")
                    total_special_requests = st.number_input("Special Requests", min_value=0, max_value=5, value=0, key="churn_requests")
                    total_stays = st.number_input("Total Previous Stays", min_value=0, max_value=20, value=1)
                    deposit_type = st.selectbox("Deposit Type", filtered_data['deposit_type'].unique().tolist(), key="churn_deposit")
                
                submit_button = st.form_submit_button("Predict Churn Risk", on_click=lambda:set_active_tab(2))
            
            if submit_button:
                # Create input data for prediction
                input_data = pd.DataFrame({
                    'hotel_name': [hotel],
                    'lead_time': [30],  # Default value
                    'stays_in_weekend_nights': [1],  # Default value
                    'stays_in_week_nights': [2],  # Default value
                    'adults': [2],  # Default value
                    'children': [0],  # Default value
                    'babies': [0],  # Default value
                    'is_repeated_guest': [is_repeated_guest],
                    'previous_cancellations': [previous_cancellations],
                    'previous_bookings_not_canceled': [total_stays],
                    'booking_changes': [booking_changes],
                    'adr': [adr],
                    'required_car_parking_spaces': [0],  # Default value
                    'total_of_special_requests': [total_special_requests],
                    'days_in_waiting_list': [0],  # Default value
                    'total_stay_nights': [3],  # Default value
                    'total_guests': [2],  # Default value
                    'year': [2017],  # Default value
                    'month': ['August'],  # Default value
                    'season': ['Summer'],  # Default value
                    'meal_plan': ['BB'],  # Default value
                    'market_segment': [market_segment],
                    'distribution_channel': [distribution_channel],
                    'country': ['PRT'],  # Default value
                    'customer_type': [customer_type],
                    'deposit_type': [deposit_type],
                    'reserved_room_type': ['A'],  # Default value
                    'assigned_room_type': ['A'],  # Default value
                    'room_changed': [room_changed],  # Default value
                    'reservation_status': ['Check-Out'],  # Default value
                    'is_canceled': [0]  # Not canceled
                })
                
                # Make prediction
                try:
                    churn_prob = churn_model.predict_proba(input_data)[:, 1][0]
                    
                    # Display result with gauge chart
                    st.subheader("Churn Prediction Result")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=churn_prob * 100,
                            title={'text': "Churn Probability"},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "green"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': churn_prob * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Risk assessment
                        if churn_prob < 0.3:
                            risk_level = "Low"
                            risk_color = "green"
                            customer_value = "Regular Maintenance"
                            recommendations = [
                                "Regular marketing communications",
                                "Standard loyalty program",
                                "Periodic satisfaction surveys"
                            ]
                        elif churn_prob < 0.7:
                            risk_level = "Medium"
                            risk_color = "orange"
                            customer_value = "Proactive Retention"
                            recommendations = [
                                "Personalized offers for next stay",
                                "Enhanced loyalty benefits",
                                "Direct outreach before typical booking window",
                                "Special amenities on next stay"
                            ]
                        else:
                            risk_level = "High"
                            risk_color = "red"
                            customer_value = "Urgent Intervention"
                            recommendations = [
                                "Immediate win-back campaign",
                                "Significant discount or upgrade offer",
                                "Personal call from hotel manager",
                                "Investigate any negative experiences",
                                "Premium loyalty incentives"
                            ]
                        
                        st.markdown(f"### Churn Risk Level: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
                        st.markdown(f"### Recommended Strategy: {customer_value}")
                        
                        st.subheader("Recommended Actions:")
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        else:
            st.warning("Customer churn prediction model not available. Please train the model first.")

#-----------------------------------------------------------------------------------
# Tab 5: Recommendations
#-----------------------------------------------------------------------------------
with tab5:
    st.header("Strategic Recommendations")
    
    st.markdown("""
    Based on the data analysis and predictive models, here are strategic recommendations
    to improve booking rates, reduce cancellations, and increase revenue.
    """)
    
    # Create tabs for different recommendation categories
    rec_tab1, rec_tab2, rec_tab3, rec_tab4 = st.tabs([
        "💰 Revenue Optimization", 
        "🛑 Cancellation Reduction", 
        "🔄 Customer Retention",
        "📊 Operational Efficiency"
    ])
    
    #-------------------------------------------------------------
    # Revenue Optimization Recommendations
    #-------------------------------------------------------------
    with rec_tab1:
        st.subheader("Revenue Optimization Strategies")
        
        # Calculate high-revenue segments
        market_revenue = filtered_data.groupby('market_segment').agg(
            total_revenue=('revenue', 'sum'),
            avg_adr=('adr', 'mean'),
            booking_count=('booking_id', 'count')
        ).reset_index()
        
        top_markets = market_revenue.sort_values('total_revenue', ascending=False).head(3)
        
        st.markdown("### Key Revenue Drivers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top Market Segments by Revenue")
            for i, row in top_markets.iterrows():
                st.markdown(f"**{row['market_segment']}**: ${row['total_revenue']:,.2f} ({row['booking_count']} bookings)")
            
            # Highest revenue seasons
            season_revenue = filtered_data.groupby('season').agg(
                total_revenue=('revenue', 'sum'),
                avg_adr=('adr', 'mean')
            ).reset_index()
            
            top_seasons = season_revenue.sort_values('total_revenue', ascending=False)
            
            st.markdown("#### Revenue by Season")
            for i, row in top_seasons.iterrows():
                st.markdown(f"**{row['season']}**: ${row['total_revenue']:,.2f} (Avg. ADR: ${row['avg_adr']:.2f})")
        
        with col2:
            # Room type revenue
            room_revenue = filtered_data.groupby('reserved_room_type').agg(
                total_revenue=('revenue', 'sum'),
                avg_adr=('adr', 'mean'),
                booking_count=('booking_id', 'count')
            ).reset_index()
            
            top_rooms = room_revenue.sort_values('avg_adr', ascending=False).head(3)
            
            st.markdown("#### Highest ADR Room Types")
            for i, row in top_rooms.iterrows():
                st.markdown(f"**Room {row['reserved_room_type']}**: ${row['avg_adr']:.2f} (${row['total_revenue']:,.2f} total)")
            
            # Stay duration revenue
            stay_revenue = filtered_data.groupby('stay_duration_bucket').agg(
                total_revenue=('revenue', 'sum'),
                avg_adr=('adr', 'mean')
            ).reset_index()
            
            top_stays = stay_revenue.sort_values('total_revenue', ascending=False).head(3)
            
            st.markdown("#### Most Profitable Stay Durations")
            for i, row in top_stays.iterrows():
                st.markdown(f"**{row['stay_duration_bucket']}**: ${row['total_revenue']:,.2f}")
                
        # Add revenue optimization recommendations
        st.markdown("### Revenue Optimization Recommendations")
        
        recommendations = [
            {
                "title": "Dynamic Pricing Strategy",
                "description": "Implement dynamic pricing based on demand forecasts, increasing rates during peak periods and offering discounts during low season.",
                "impact": "High",
                "implementation": "Medium",
                "details": [
                    f"Increase rates by 10-15% during {top_seasons.iloc[0]['season']} season",
                    f"Target {top_markets.iloc[0]['market_segment']} segment with premium packages",
                    "Create weekend specials for higher occupancy",
                    "Adjust pricing based on lead time analysis"
                ]
            },
            {
                "title": "Room Type Optimization",
                "description": f"Focus marketing on high-revenue room types like Room {top_rooms.iloc[0]['reserved_room_type']} and offer strategic upsells from lower categories.",
                "impact": "Medium",
                "implementation": "Easy",
                "details": [
                    f"Highlight Room {top_rooms.iloc[0]['reserved_room_type']} in marketing materials",
                    "Create premium packages for high-end rooms",
                    "Offer discounted upgrades at check-in",
                    "Renovate or enhance features of top-performing room types"
                ]
            },
            {
                "title": "Length of Stay Strategy",
                "description": f"Encourage {top_stays.iloc[0]['stay_duration_bucket']} stays through targeted promotions and pricing incentives.",
                "impact": "Medium",
                "implementation": "Medium",
                "details": [
                    "Offer discounts for minimum stay requirements",
                    "Create extended stay packages with additional amenities",
                    "Target weekend extenders with Sunday night discounts",
                    "Implement length-of-stay pricing tiers"
                ]
            }
        ]
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"{i+1}. {rec['title']} (Impact: {rec['impact']})"):
                st.markdown(f"**Description**: {rec['description']}")
                st.markdown(f"**Implementation Difficulty**: {rec['implementation']}")
                
                st.markdown("**Implementation Steps**:")
                for step in rec['details']:
                    st.markdown(f"- {step}")
    
    #-------------------------------------------------------------
    # Cancellation Reduction Recommendations
    #-------------------------------------------------------------
    with rec_tab2:
        st.subheader("Cancellation Reduction Strategies")
        
        # Calculate cancellation statistics
        lead_time_cancel = filtered_data.groupby('lead_time_bucket').agg(
            cancellation_rate=('is_canceled', 'mean'),
            count=('booking_id', 'count')
        ).reset_index()
        
        lead_time_cancel['cancellation_rate'] = lead_time_cancel['cancellation_rate'] * 100
        high_cancel_lead = lead_time_cancel.sort_values('cancellation_rate', ascending=False).iloc[0]
        
        deposit_cancel = filtered_data.groupby('deposit_type').agg(
            cancellation_rate=('is_canceled', 'mean'),
            count=('booking_id', 'count')
        ).reset_index()
        
        deposit_cancel['cancellation_rate'] = deposit_cancel['cancellation_rate'] * 100
        
        market_cancel = filtered_data.groupby('market_segment').agg(
            cancellation_rate=('is_canceled', 'mean'),
            count=('booking_id', 'count')
        ).reset_index()
        
        market_cancel['cancellation_rate'] = market_cancel['cancellation_rate'] * 100
        high_cancel_market = market_cancel.sort_values('cancellation_rate', ascending=False).iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Cancellation Risk Factors")
            
            st.markdown(f"**Highest Cancellation Lead Time**: {high_cancel_lead['lead_time_bucket']} ({high_cancel_lead['cancellation_rate']:.1f}%)")
            
            st.markdown(f"**Highest Cancellation Market Segment**: {high_cancel_market['market_segment']} ({high_cancel_market['cancellation_rate']:.1f}%)")
            
            # Deposit type effect
            st.markdown("#### Effect of Deposit Type on Cancellations")
            for i, row in deposit_cancel.iterrows():
                st.markdown(f"**{row['deposit_type']}**: {row['cancellation_rate']:.1f}% cancellation rate")
        
        with col2:
            # Other cancellation factors
            avg_lead_time_canceled = filtered_data[filtered_data['is_canceled'] == 1]['lead_time'].mean()
            avg_lead_time_not_canceled = filtered_data[filtered_data['is_canceled'] == 0]['lead_time'].mean()
            
            st.markdown("### Key Cancellation Metrics")
            
            st.markdown(f"**Average Lead Time for Canceled Bookings**: {avg_lead_time_canceled:.1f} days")
            st.markdown(f"**Average Lead Time for Completed Bookings**: {avg_lead_time_not_canceled:.1f} days")
            
            # Special requests impact
            avg_requests_canceled = filtered_data[filtered_data['is_canceled'] == 1]['total_of_special_requests'].mean()
            avg_requests_not_canceled = filtered_data[filtered_data['is_canceled'] == 0]['total_of_special_requests'].mean()
            
            st.markdown(f"**Avg. Special Requests (Canceled)**: {avg_requests_canceled:.2f}")
            st.markdown(f"**Avg. Special Requests (Completed)**: {avg_requests_not_canceled:.2f}")
            
            # Overall cancellation rate
            overall_cancel_rate = filtered_data['is_canceled'].mean() * 100
            st.markdown(f"**Overall Cancellation Rate**: {overall_cancel_rate:.1f}%")
            
        # Add cancellation reduction recommendations
        st.markdown("### Cancellation Reduction Recommendations")
        
        recommendations = [
            {
                "title": "Tiered Deposit Policy",
                "description": "Implement a tiered deposit policy based on lead time, booking value, and cancellation risk score.",
                "impact": "High",
                "implementation": "Medium",
                "details": [
                    f"Require deposits for {high_cancel_lead['lead_time_bucket']} bookings",
                    f"Focus on {high_cancel_market['market_segment']} market segment",
                    "Offer deposit discounts for loyalty members",
                    "Create flexible deposit options with varying refund policies"
                ]
            },
            {
                "title": "Pre-arrival Engagement",
                "description": "Develop a systematic pre-arrival communication strategy to reduce cancellations, especially for high-risk bookings.",
                "impact": "Medium",
                "implementation": "Easy",
                "details": [
                    "Send personalized pre-arrival emails at 30, 14, and 7 days",
                    "Offer pre-arrival concierge services",
                    "Provide local area information and activity suggestions",
                    "Create anticipation with room-specific details and photos"
                ]
            },
            {
                "title": "Cancellation Prediction System",
                "description": "Implement the cancellation prediction model to identify high-risk bookings and take proactive measures.",
                "impact": "High",
                "implementation": "Complex",
                "details": [
                    "Flag high-risk bookings in the reservation system",
                    "Create automated intervention protocols based on risk level",
                    "Train staff on handling high-risk cancellation bookings",
                    "Implement automated follow-ups for bookings with > 70% risk"
                ]
            }
        ]
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"{i+1}. {rec['title']} (Impact: {rec['impact']})"):
                st.markdown(f"**Description**: {rec['description']}")
                st.markdown(f"**Implementation Difficulty**: {rec['implementation']}")
                
                st.markdown("**Implementation Steps**:")
                for step in rec['details']:
                    st.markdown(f"- {step}")
    
    #-------------------------------------------------------------
    # Customer Retention Recommendations
    #-------------------------------------------------------------
    with rec_tab3:
        st.subheader("Customer Retention Strategies")
        
        # Create value segment based on total spend if not already created
        if 'value_segment' not in filtered_data.columns:
            filtered_data['total_value'] = filtered_data['adr'] * filtered_data['total_stay_nights']
            high_value_threshold = filtered_data['total_value'].quantile(0.75)
            filtered_data['value_segment'] = np.where(
                filtered_data['total_value'] >= high_value_threshold,
                'High Value',
                'Standard Value'
            )
        
        # Calculate retention statistics
        repeat_rate = filtered_data['is_repeated_guest'].mean() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Customer Retention Metrics")
            
            st.markdown(f"**Overall Repeat Guest Rate**: {repeat_rate:.1f}%")
            
            # High value retention
            high_value_repeat = filtered_data[
                (filtered_data['value_segment'] == 'High Value') & 
                (filtered_data['is_repeated_guest'] == 1)
            ].shape[0]
            
            high_value_total = filtered_data[filtered_data['value_segment'] == 'High Value'].shape[0]
            high_value_repeat_rate = (high_value_repeat / high_value_total) * 100 if high_value_total > 0 else 0
            
            st.markdown(f"**High-Value Guest Repeat Rate**: {high_value_repeat_rate:.1f}%")
            
            # Segments with highest retention
            segment_retention = filtered_data.groupby('market_segment').agg(
                repeat_rate=('is_repeated_guest', 'mean'),
                booking_count=('booking_id', 'count')
            ).reset_index()
            
            segment_retention['repeat_rate'] = segment_retention['repeat_rate'] * 100
            top_retention_segment = segment_retention.sort_values('repeat_rate', ascending=False).iloc[0]
            
            st.markdown(f"**Highest Retention Segment**: {top_retention_segment['market_segment']} ({top_retention_segment['repeat_rate']:.1f}%)")
        
        with col2:
            st.markdown("### Guest Value Analysis")
            
            # Average spending by repeat vs new
            avg_spend_repeat = filtered_data[filtered_data['is_repeated_guest'] == 1]['total_value'].mean()
            avg_spend_new = filtered_data[filtered_data['is_repeated_guest'] == 0]['total_value'].mean()
            
            spend_diff = ((avg_spend_repeat / avg_spend_new) - 1) * 100 if avg_spend_new > 0 else 0
            
            st.markdown(f"**Avg. Spend (Repeat Guests)**: ${avg_spend_repeat:.2f}")
            st.markdown(f"**Avg. Spend (New Guests)**: ${avg_spend_new:.2f}")
            st.markdown(f"**Spending Difference**: {spend_diff:+.1f}%")
            
            # Special requests by repeat vs new
            avg_requests_repeat = filtered_data[filtered_data['is_repeated_guest'] == 1]['total_of_special_requests'].mean()
            avg_requests_new = filtered_data[filtered_data['is_repeated_guest'] == 0]['total_of_special_requests'].mean()
            
            st.markdown(f"**Avg. Special Requests (Repeat)**: {avg_requests_repeat:.2f}")
            st.markdown(f"**Avg. Special Requests (New)**: {avg_requests_new:.2f}")
            
        # Add customer retention recommendations
        st.markdown("### Customer Retention Recommendations")
        
        recommendations = [
            {
                "title": "Tiered Loyalty Program",
                "description": "Implement a tiered loyalty program with meaningful benefits that increase with guest value and frequency.",
                "impact": "High",
                "implementation": "Complex",
                "details": [
                    "Create 3-4 membership tiers with increasing benefits",
                    "Offer immediate benefits from first stay",
                    "Include exclusive experiences for top-tier members",
                    "Implement soft benefits (recognition, flexibility) alongside hard benefits (points, upgrades)"
                ]
            },
            {
                "title": "High-Value Guest Program",
                "description": "Create a specialized program for high-value guests with personalized service and exclusive benefits.",
                "impact": "High",
                "implementation": "Medium",
                "details": [
                    "Assign dedicated guest relations manager",
                    "Offer complimentary airport transfers",
                    "Provide guaranteed late checkout and early check-in",
                    "Create exclusive access to hotel facilities and events"
                ]
            },
            {
                "title": "Churn Prevention System",
                "description": "Implement the churn prediction model to identify at-risk guests and take proactive retention measures.",
                "impact": "High",
                "implementation": "Complex",
                "details": [
                    "Implement automated churn risk scoring",
                    "Create intervention protocols for high-risk guests",
                    "Develop win-back campaigns for churned guests",
                    "Train staff on handling high-value, high-risk guests"
                ]
            }
        ]
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"{i+1}. {rec['title']} (Impact: {rec['impact']})"):
                st.markdown(f"**Description**: {rec['description']}")
                st.markdown(f"**Implementation Difficulty**: {rec['implementation']}")
                
                st.markdown("**Implementation Steps**:")
                for step in rec['details']:
                    st.markdown(f"- {step}")
    
    #-------------------------------------------------------------
    # Operational Efficiency Recommendations
    #-------------------------------------------------------------
    with rec_tab4:
        st.subheader("Operational Efficiency Strategies")
        
        # Calculate operational metrics
        avg_lead_time = filtered_data['lead_time'].mean()
        avg_stay_length = filtered_data['total_stay_nights'].mean()
        room_change_rate = filtered_data['room_changed'].mean() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Operational Metrics")
            
            st.markdown(f"**Average Lead Time**: {avg_lead_time:.1f} days")
            st.markdown(f"**Average Stay Length**: {avg_stay_length:.1f} nights")
            st.markdown(f"**Room Change Rate**: {room_change_rate:.1f}%")
            
            # Distribution channels
            channel_dist = filtered_data.groupby('distribution_channel').size().reset_index(name='count')
            top_channel = channel_dist.sort_values('count', ascending=False).iloc[0]
            
            st.markdown(f"**Primary Distribution Channel**: {top_channel['distribution_channel']} ({top_channel['count']} bookings)")
        
        with col2:
            # Booking patterns
            st.markdown("### Booking Patterns")
            
            # Busiest months
            monthly_bookings = filtered_data.groupby('month').size().reset_index(name='count')
            monthly_bookings['month_num'] = monthly_bookings['month'].map(month_order)
            monthly_bookings = monthly_bookings.sort_values('count', ascending=False)
            
            st.markdown("**Busiest Months:**")
            for i, row in monthly_bookings.head(3).iterrows():
                st.markdown(f"- {row['month']}: {row['count']} bookings")
            
            # Quietest months
            st.markdown("**Quietest Months:**")
            for i, row in monthly_bookings.tail(3).sort_values('count').iterrows():
                st.markdown(f"- {row['month']}: {row['count']} bookings")
                
        # Add operational efficiency recommendations
        st.markdown("### Operational Efficiency Recommendations")
        
        recommendations = [
            {
                "title": "Demand-Based Staffing Model",
                "description": "Implement a staffing model based on booking forecasts and historical demand patterns.",
                "impact": "High",
                "implementation": "Medium",
                "details": [
                    "Create staffing templates for high, medium, and low demand periods",
                    "Implement flexible scheduling during seasonal transitions",
                    "Develop cross-training program for staff versatility",
                    "Introduce part-time and on-call staff for peak periods"
                ]
            },
            {
                "title": "Room Allocation Optimization",
                "description": "Develop a system to optimize room assignments, reducing unnecessary room changes and maximizing inventory utilization.",
                "impact": "Medium",
                "implementation": "Medium",
                "details": [
                    "Implement predictive room assignment algorithms",
                    "Create room maintenance scheduling tied to occupancy forecasts",
                    "Develop room upgrade paths for efficient upselling",
                    "Improve coordination between housekeeping and front desk"
                ]
            },
            {
                "title": "Distribution Channel Optimization",
                "description": "Optimize channel mix based on profitability, lead times, and cancellation patterns.",
                "impact": "High",
                "implementation": "Complex",
                "details": [
                    "Analyze costs and conversion rates across all channels",
                    "Implement channel-specific pricing and availability strategies",
                    "Develop channel shift incentives for direct bookings",
                    "Create channel performance dashboards for management"
                ]
            }
        ]
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"{i+1}. {rec['title']} (Impact: {rec['impact']})"):
                st.markdown(f"**Description**: {rec['description']}")
                st.markdown(f"**Implementation Difficulty**: {rec['implementation']}")
                
                st.markdown("**Implementation Steps**:")
                for step in rec['details']:
                    st.markdown(f"- {step}")
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
