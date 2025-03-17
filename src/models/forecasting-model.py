import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

db_path = "data/hotel_dw.db" 

class BookingDemandForecasting:
    """
    Time Series Forecasting model for hotel booking demand
    """
    
    def __init__(self, db_path=db_path):
        """
        Initialize the booking demand forecasting model
        
        Parameters:
        -----------
        db_path: str
            Path to the SQLite database for the data warehouse
        """
        self.db_path = db_path
        self.conn = None
        self.data = None
        self.time_series = None
        self.model = None
        
    def connect_to_db(self):
        """Establish connection to SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        print(f"Connected to database: {self.db_path}")
        
    def load_data(self):
        """Load data from the data warehouse"""
        print("Loading data from data warehouse...")
        
        if self.conn is None:
            self.connect_to_db()
            
        # SQL query to get booking counts by date and hotel
        query = """
        SELECT 
            dd.year, 
            dd.month,
            hd.hotel_name,
            COUNT(*) as booking_count,
            SUM(bf.adr * bf.total_stay_nights) as total_revenue,
            AVG(bf.adr) as avg_daily_rate
        FROM 
            booking_fact bf
        JOIN 
            date_dim dd ON bf.date_key = dd.date_key
        JOIN 
            hotel_dim hd ON bf.hotel_key = hd.hotel_key
        GROUP BY 
            dd.year, dd.month, hd.hotel_name
        ORDER BY 
            dd.year, CASE 
                WHEN dd.month = 'January' THEN 1
                WHEN dd.month = 'February' THEN 2
                WHEN dd.month = 'March' THEN 3
                WHEN dd.month = 'April' THEN 4
                WHEN dd.month = 'May' THEN 5
                WHEN dd.month = 'June' THEN 6
                WHEN dd.month = 'July' THEN 7
                WHEN dd.month = 'August' THEN 8
                WHEN dd.month = 'September' THEN 9
                WHEN dd.month = 'October' THEN 10
                WHEN dd.month = 'November' THEN 11
                WHEN dd.month = 'December' THEN 12
            END, hd.hotel_name
        """
        
        self.data = pd.read_sql_query(query, self.conn)
        print(f"Loaded {len(self.data)} records")
        
        # Convert year and month to datetime for time series analysis
        self.data['month_num'] = self.data['month'].map({
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        })
        
        self.data['date'] = pd.to_datetime(self.data['year'].astype(str) + '-' + 
                                          self.data['month_num'].astype(str) + '-01')
        
        return self.data
        
    def prepare_time_series(self, hotel_name=None):
        """
        Prepare time series data for forecasting
        
        Parameters:
        -----------
        hotel_name: str, optional
            Filter data for a specific hotel, or None for all hotels combined
        """
        if self.data is None:
            self.load_data()
            
        print(f"Preparing time series data for {'all hotels' if hotel_name is None else hotel_name}...")
        
        # Filter data if hotel is specified
        if hotel_name is not None:
            df = self.data[self.data['hotel_name'] == hotel_name].copy()
        else:
            # Aggregate across all hotels
            df = self.data.groupby('date').agg({
                'booking_count': 'sum',
                'total_revenue': 'sum',
                'avg_daily_rate': 'mean'
            }).reset_index()
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Ensure the time series is complete
        date_range = pd.date_range(df.index.min(), df.index.max(), freq='MS')
        
        # Reindex and fill missing values
        self.time_series = df.reindex(date_range)
        
        # Forward fill for any missing months
        self.time_series.fillna(method='ffill', inplace=True)
        
        print(f"Time series prepared with {len(self.time_series)} periods")
        
        return self.time_series
        
    def explore_time_series(self):
        """Explore and visualize the time series data"""
        if self.time_series is None:
            print("Time series not prepared yet!")
            return
            
        print("Exploring time series data...")
        
        # Calculate appropriate lag for ACF/PACF based on data size
        max_lag = min(int(len(self.time_series) * 0.4), 12)  # 40% of data size, max 12 lags
        
        # Plot the time series
        plt.figure(figsize=(14, 7))
        plt.plot(self.time_series.index, self.time_series['booking_count'], marker='o')
        plt.title('Hotel Booking Demand Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Bookings')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('booking_demand_time_series.png')
        
        # Seasonal decomposition
        if len(self.time_series) >= 12:  # Need at least a year of data
            decomposition = seasonal_decompose(self.time_series['booking_count'], model='multiplicative', period=min(12, len(self.time_series) // 2))
            
            plt.figure(figsize=(14, 10))
            
            plt.subplot(411)
            plt.plot(decomposition.observed)
            plt.title('Observed')
            plt.grid(True)
            
            plt.subplot(412)
            plt.plot(decomposition.trend)
            plt.title('Trend')
            plt.grid(True)
            
            plt.subplot(413)
            plt.plot(decomposition.seasonal)
            plt.title('Seasonality')
            plt.grid(True)
            
            plt.subplot(414)
            plt.plot(decomposition.resid)
            plt.title('Residuals')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('booking_demand_decomposition.png')
            
            # ACF and PACF plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            plot_acf(self.time_series['booking_count'].dropna(), ax=ax1, lags=max_lag)
            plot_pacf(self.time_series['booking_count'].dropna(), ax=ax2, lags=max_lag)
            plt.savefig('booking_demand_acf_pacf.png')
            
        print("Time series exploration completed, visualizations saved.")
    
    def build_sarima_model(self, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12)):
        """
        Build a SARIMA model for time series forecasting
        
        Parameters:
        -----------
        order: tuple
            (p, d, q) parameters for ARIMA
        seasonal_order: tuple
            (P, D, Q, s) seasonal parameters for SARIMA
        """
        if self.time_series is None:
            print("Time series not prepared yet!")
            return
            
        print("Building SARIMA model...")
        
        # Prepare data
        train_size = int(len(self.time_series) * 0.8)
        train = self.time_series.iloc[:train_size]
        test = self.time_series.iloc[train_size:]
        
        print(f"Training on {len(train)} periods, testing on {len(test)} periods")
        
        # Build SARIMA model
        model = SARIMAX(
            train['booking_count'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        print(f"Model summary:\n{fitted_model.summary()}")
        
        # Store the model
        self.model = fitted_model
        
        # Make predictions for test period
        forecast = fitted_model.get_forecast(steps=len(test))
        forecast_ci = forecast.conf_int()
        
        # Evaluate the model
        predictions = forecast.predicted_mean
        rmse = np.sqrt(mean_squared_error(test['booking_count'], predictions))
        mae = mean_absolute_error(test['booking_count'], predictions)
        r2 = r2_score(test['booking_count'], predictions)
        
        print(f"Model evaluation on test data:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R² Score: {r2:.2f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(14, 7))
        plt.plot(self.time_series.index, self.time_series['booking_count'], label='Actual')
        plt.plot(test.index, predictions, color='red', label='Predicted')
        plt.fill_between(
            test.index,
            forecast_ci.iloc[:, 0],
            forecast_ci.iloc[:, 1],
            color='pink', alpha=0.3
        )
        plt.title('Hotel Booking Demand: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Number of Bookings')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('booking_demand_forecast.png')
        
        return self.model
    
    def make_future_predictions(self, periods=12):
        """
        Make future predictions using the trained model
        
        Parameters:
        -----------
        periods: int
            Number of future periods to forecast
            
        Returns:
        --------
        DataFrame with forecasted values and confidence intervals
        """
        if self.model is None:
            print("Model not trained yet!")
            return None
            
        print(f"Forecasting demand for next {periods} months...")
        
        # Get forecast
        forecast = self.model.get_forecast(steps=periods)
        forecast_ci = forecast.conf_int()
        
        # Create forecast dataframe
        last_date = self.time_series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_bookings': forecast.predicted_mean.values,
            'lower_ci': forecast_ci.iloc[:, 0].values,
            'upper_ci': forecast_ci.iloc[:, 1].values
        })
        
        # Plot the forecast
        plt.figure(figsize=(14, 7))
        
        # Historical data
        plt.plot(self.time_series.index, self.time_series['booking_count'], label='Historical')
        
        # Forecast
        plt.plot(forecast_df['date'], forecast_df['predicted_bookings'], color='red', label='Forecast')
        plt.fill_between(
            forecast_df['date'],
            forecast_df['lower_ci'],
            forecast_df['upper_ci'],
            color='pink', alpha=0.3
        )
        
        plt.title('Hotel Booking Demand Forecast')
        plt.xlabel('Date')
        plt.ylabel('Number of Bookings')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('booking_demand_future_forecast.png')
        
        print("Future demand forecast generated.")
        return forecast_df
    
    def save_model(self, filename='models/booking_forecast_model.pkl'):
        """Save the trained model to disk"""
        if self.model is None:
            print("No model to save!")
            return False
            
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
        return True
    
    def load_model(self, filename='booking_forecast_model.pkl'):
        """Load a trained model from disk"""
        try:
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def run_pipeline(self, hotel_name=None):
        """
        Run the complete forecasting pipeline
        
        Parameters:
        -----------
        hotel_name: str, optional
            Filter data for a specific hotel, or None for all hotels combined
        """
        print("Starting booking demand forecasting pipeline...")
        
        try:
            # Load data
            self.load_data()
            
            # Prepare time series
            self.prepare_time_series(hotel_name)
            
            # Explore time series
            self.explore_time_series()
            
            # Build model - try different parameters based on data exploration
            self.build_sarima_model(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            
            # Make future predictions
            self.make_future_predictions(periods=12)
            
            # Save model
            self.save_model()
            
            print("Booking demand forecasting pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error in forecasting pipeline: {str(e)}")
            return False
        finally:
            if self.conn:
                self.conn.close()
                print("Database connection closed")


# Example usage
if __name__ == "__main__":
    forecaster = BookingDemandForecasting(db_path)
    forecaster.run_pipeline()  # For all hotels
    # Or for a specific hotel:
    # forecaster.run_pipeline(hotel_name="Resort Hotel")
