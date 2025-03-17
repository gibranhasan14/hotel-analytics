import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import os

csv_path = os.path.join("data", "hotel_bookings.csv")
db_path = os.path.join("data", "hotel_dw.db")

class HotelBookingsETL:
    """
    ETL pipeline for processing hotel bookings data into a star schema data warehouse
    """
    
    def __init__(self, csv_path, db_path='data/hotel_dw.db'):
        """
        Initialize the ETL process
        
        Parameters:
        -----------
        csv_path: str
            Path to the CSV file containing hotel bookings data
        db_path: str
            Path to the SQLite database for the data warehouse
        """
        self.csv_path = csv_path
        self.db_path = db_path
        self.conn = None
        self.raw_data = None
        
    def connect_to_db(self):
        """Establish connection to SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        print(f"Connected to database: {self.db_path}")
        
    def extract_data(self):
        """Extract data from CSV file"""
        print(f"Extracting data from: {self.csv_path}")
        self.raw_data = pd.read_csv(self.csv_path)
        print(f"Extracted {len(self.raw_data)} records")
        return self.raw_data
    
    def transform_data(self):
        """Transform raw data into dimensional model"""
        print("Transforming data...")
        
        # Create dimension tables
        date_dim = self._create_date_dimension()
        hotel_dim = self._create_hotel_dimension()
        customer_dim = self._create_customer_dimension()
        room_dim = self._create_room_dimension()
        agent_dim = self._create_agent_dimension()
        
        # Create fact table with foreign keys
        booking_fact = self._create_booking_fact(date_dim, hotel_dim, customer_dim, room_dim, agent_dim)
        
        return {
            'date_dim': date_dim,
            'hotel_dim': hotel_dim,
            'customer_dim': customer_dim,
            'room_dim': room_dim,
            'agent_dim': agent_dim,
            'booking_fact': booking_fact
        }
    
    def _create_date_dimension(self):
        """Create date dimension table"""
        print("Creating date dimension...")
        
        # Extract unique dates from the dataset
        df = self.raw_data.copy()
        
        # Convert string dates to datetime objects
        df['arrival_date'] = pd.to_datetime(
            df['arrival_date_year'].astype(str) + '-' + 
            df['arrival_date_month'] + '-' + 
            df['arrival_date_day_of_month'].astype(str)
        )
        
        # Create unique dates
        unique_dates = df[['arrival_date_year', 'arrival_date_month', 
                           'arrival_date_week_number', 'arrival_date_day_of_month', 
                           'arrival_date']].drop_duplicates()
        
        # Create date dimension table
        date_dim = pd.DataFrame()
        date_dim['date_key'] = range(1, len(unique_dates) + 1)
        date_dim['year'] = unique_dates['arrival_date_year'].values
        date_dim['month'] = unique_dates['arrival_date_month'].values
        date_dim['week_number'] = unique_dates['arrival_date_week_number'].values
        date_dim['day_of_month'] = unique_dates['arrival_date_day_of_month'].values
        date_dim['full_date'] = unique_dates['arrival_date'].values
        
        # Add derived columns
        date_dim['is_weekend'] = date_dim['full_date'].dt.dayofweek >= 5
        
        # Map month to season
        def get_season(month):
            if month in ['December', 'January', 'February']:
                return 'Winter'
            elif month in ['March', 'April', 'May']:
                return 'Spring'
            elif month in ['June', 'July', 'August']:
                return 'Summer'
            else:
                return 'Fall'
        
        date_dim['season'] = date_dim['month'].apply(get_season)
        
        # Create lookup dictionary for easier mapping in fact table
        self.date_lookup = dict(zip(
            zip(unique_dates['arrival_date_year'], 
                unique_dates['arrival_date_month'], 
                unique_dates['arrival_date_day_of_month']),
            date_dim['date_key']
        ))
        
        return date_dim
        
    def _create_hotel_dimension(self):
        """Create hotel dimension table"""
        print("Creating hotel dimension...")
        
        # Extract unique hotel configurations
        hotel_configs = self.raw_data[['hotel', 'meal', 'market_segment', 'distribution_channel']].drop_duplicates()
        
        # Create hotel dimension table
        hotel_dim = pd.DataFrame()
        hotel_dim['hotel_key'] = range(1, len(hotel_configs) + 1)
        hotel_dim['hotel_name'] = hotel_configs['hotel'].values
        hotel_dim['meal_plan'] = hotel_configs['meal'].values
        hotel_dim['market_segment'] = hotel_configs['market_segment'].values
        hotel_dim['distribution_channel'] = hotel_configs['distribution_channel'].values
        
        # Create lookup dictionary
        self.hotel_lookup = dict(zip(
            zip(hotel_configs['hotel'], hotel_configs['meal'], 
                hotel_configs['market_segment'], hotel_configs['distribution_channel']),
            hotel_dim['hotel_key']
        ))
        
        return hotel_dim
    
    def _create_customer_dimension(self):
        """Create customer dimension table"""
        print("Creating customer dimension...")
        
        # Extract unique customer configurations
        customer_configs = self.raw_data[['country', 'customer_type', 'deposit_type']].drop_duplicates()
        
        # Create customer dimension table
        customer_dim = pd.DataFrame()
        customer_dim['customer_key'] = range(1, len(customer_configs) + 1)
        customer_dim['country'] = customer_configs['country'].values
        customer_dim['customer_type'] = customer_configs['customer_type'].values
        customer_dim['deposit_type'] = customer_configs['deposit_type'].values
        
        # Create lookup dictionary
        self.customer_lookup = dict(zip(
            zip(customer_configs['country'], customer_configs['customer_type'], 
                customer_configs['deposit_type']),
            customer_dim['customer_key']
        ))
        
        return customer_dim
    
    def _create_room_dimension(self):
        """Create room dimension table"""
        print("Creating room dimension...")
        
        # Extract unique room configurations
        room_configs = self.raw_data[['reserved_room_type', 'assigned_room_type']].drop_duplicates()
        
        # Create room dimension table
        room_dim = pd.DataFrame()
        room_dim['room_key'] = range(1, len(room_configs) + 1)
        room_dim['reserved_room_type'] = room_configs['reserved_room_type'].values
        room_dim['assigned_room_type'] = room_configs['assigned_room_type'].values
        room_dim['room_changed'] = room_configs['reserved_room_type'] != room_configs['assigned_room_type']
        
        # Create lookup dictionary
        self.room_lookup = dict(zip(
            zip(room_configs['reserved_room_type'], room_configs['assigned_room_type']),
            room_dim['room_key']
        ))
        
        return room_dim
    
    def _create_agent_dimension(self):
        """Create agent dimension table"""
        print("Creating agent dimension...")
        
        # Extract unique agent/company configurations
        agent_configs = self.raw_data[['agent', 'company']].drop_duplicates()
        
        # Handle NULL values
        agent_configs['agent'] = agent_configs['agent'].fillna('Unknown')
        agent_configs['company'] = agent_configs['company'].fillna('Unknown')
        
        # Create agent dimension table
        agent_dim = pd.DataFrame()
        agent_dim['agent_key'] = range(1, len(agent_configs) + 1)
        agent_dim['agent_id'] = agent_configs['agent'].values
        agent_dim['company_id'] = agent_configs['company'].values
        
        # Create lookup dictionary
        self.agent_lookup = dict(zip(
            zip(agent_configs['agent'], agent_configs['company']),
            agent_dim['agent_key']
        ))
        
        return agent_dim
    
    def _create_booking_fact(self, date_dim, hotel_dim, customer_dim, room_dim, agent_dim):
        """Create booking fact table with foreign keys"""
        print("Creating booking fact table...")
        
        df = self.raw_data.copy()
        
        # Fill NA values for numerical columns
        df['children'] = df['children'].fillna(0)
        df['agent'] = df['agent'].fillna('Unknown')
        df['company'] = df['company'].fillna('Unknown')
        
        # Create booking fact table
        booking_fact = pd.DataFrame()
        booking_fact['booking_id'] = range(1, len(df) + 1)
        
        # Add foreign keys
        booking_fact['date_key'] = [
            self.date_lookup.get((row['arrival_date_year'], row['arrival_date_month'], row['arrival_date_day_of_month']), 1) 
            for _, row in df.iterrows()
        ]
        
        booking_fact['hotel_key'] = [
            self.hotel_lookup.get((row['hotel'], row['meal'], row['market_segment'], row['distribution_channel']), 1) 
            for _, row in df.iterrows()
        ]
        
        booking_fact['customer_key'] = [
            self.customer_lookup.get((row['country'], row['customer_type'], row['deposit_type']), 1) 
            for _, row in df.iterrows()
        ]
        
        booking_fact['room_key'] = [
            self.room_lookup.get((row['reserved_room_type'], row['assigned_room_type']), 1) 
            for _, row in df.iterrows()
        ]
        
        booking_fact['agent_key'] = [
            self.agent_lookup.get((row['agent'], row['company']), 1) 
            for _, row in df.iterrows()
        ]
        
        # Add measures
        booking_fact['is_canceled'] = df['is_canceled'].values
        booking_fact['lead_time'] = df['lead_time'].values
        booking_fact['stays_in_weekend_nights'] = df['stays_in_weekend_nights'].values
        booking_fact['stays_in_week_nights'] = df['stays_in_week_nights'].values
        booking_fact['adults'] = df['adults'].values
        booking_fact['children'] = df['children'].values
        booking_fact['babies'] = df['babies'].values
        booking_fact['is_repeated_guest'] = df['is_repeated_guest'].values
        booking_fact['previous_cancellations'] = df['previous_cancellations'].values
        booking_fact['previous_bookings_not_canceled'] = df['previous_bookings_not_canceled'].values
        booking_fact['booking_changes'] = df['booking_changes'].values
        booking_fact['adr'] = df['adr'].values
        booking_fact['required_car_parking_spaces'] = df['required_car_parking_spaces'].values
        booking_fact['total_of_special_requests'] = df['total_of_special_requests'].values
        booking_fact['days_in_waiting_list'] = df['days_in_waiting_list'].values
        booking_fact['reservation_status'] = df['reservation_status'].values
        booking_fact['reservation_status_date'] = df['reservation_status_date'].values
        
        # Add derived measures
        booking_fact['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        booking_fact['total_guests'] = df['adults'] + df['children'] + df['babies']
        booking_fact['revenue'] = booking_fact['adr'] * booking_fact['total_stay_nights']
        
        return booking_fact
    
    def load_data(self, transformed_data):
        """Load transformed data into the database"""
        print("Loading data into database...")
        
        if self.conn is None:
            self.connect_to_db()
        
        # Create tables in database
        for table_name, df in transformed_data.items():
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)
            print(f"Loaded {len(df)} records into {table_name}")
    
    def create_indexes(self):
        """Create indexes on the tables for better query performance"""
        print("Creating indexes...")
        
        if self.conn is None:
            self.connect_to_db()
            
        cursor = self.conn.cursor()
        
        # Primary key indexes
        cursor.execute("CREATE INDEX idx_date_dim_date_key ON date_dim(date_key)")
        cursor.execute("CREATE INDEX idx_hotel_dim_hotel_key ON hotel_dim(hotel_key)")
        cursor.execute("CREATE INDEX idx_customer_dim_customer_key ON customer_dim(customer_key)")
        cursor.execute("CREATE INDEX idx_room_dim_room_key ON room_dim(room_key)")
        cursor.execute("CREATE INDEX idx_agent_dim_agent_key ON agent_dim(agent_key)")
        cursor.execute("CREATE INDEX idx_booking_fact_booking_id ON booking_fact(booking_id)")
        
        # Foreign key indexes
        cursor.execute("CREATE INDEX idx_booking_fact_date_key ON booking_fact(date_key)")
        cursor.execute("CREATE INDEX idx_booking_fact_hotel_key ON booking_fact(hotel_key)")
        cursor.execute("CREATE INDEX idx_booking_fact_customer_key ON booking_fact(customer_key)")
        cursor.execute("CREATE INDEX idx_booking_fact_room_key ON booking_fact(room_key)")
        cursor.execute("CREATE INDEX idx_booking_fact_agent_key ON booking_fact(agent_key)")
        
        # Additional indexes for common queries
        cursor.execute("CREATE INDEX idx_booking_fact_is_canceled ON booking_fact(is_canceled)")
        cursor.execute("CREATE INDEX idx_date_dim_year_month ON date_dim(year, month)")
        cursor.execute("CREATE INDEX idx_booking_fact_lead_time ON booking_fact(lead_time)")
        
        self.conn.commit()
        print("Indexes created successfully")
    
    def run_etl_pipeline(self):
        """Run the complete ETL pipeline"""
        print("Starting ETL pipeline...")
        
        try:
            # Extract
            self.extract_data()
            
            # Transform
            transformed_data = self.transform_data()
            
            # Load
            self.load_data(transformed_data)
            
            # Create indexes
            self.create_indexes()
            
            print("ETL pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error in ETL pipeline: {str(e)}")
            return False
        finally:
            if self.conn:
                self.conn.close()
                print("Database connection closed")


# Example usage
if __name__ == "__main__":
    etl = HotelBookingsETL(csv_path, db_path="data/hotel_dw.db")
    etl.run_etl_pipeline()
