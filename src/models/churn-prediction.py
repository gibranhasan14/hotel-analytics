import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import joblib

db_path = "data/hotel_dw.db" 

class CustomerChurnPrediction:
    """
    Model to predict customer churn (guests unlikely to return)
    """
    
    def __init__(self, db_path=db_path):
        """
        Initialize the customer churn prediction model
        
        Parameters:
        -----------
        db_path: str
            Path to the SQLite database for the data warehouse
        """
        self.db_path = db_path
        self.conn = None
        self.data = None
        self.model = None
        self.repeat_booking_threshold = 60  # days to consider for repeat bookings
        
    def connect_to_db(self):
        """Establish connection to SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        print(f"Connected to database: {self.db_path}")
        
    def load_data(self):
        """Load data from the data warehouse"""
        print("Loading data from data warehouse...")
        
        if self.conn is None:
            self.connect_to_db()
            
        # SQL query to get customer data
        query = """
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
            
            dd.year,
            dd.month,
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
        
        self.data = pd.read_sql_query(query, self.conn)
        print(f"Loaded {len(self.data)} records")
        
        # Convert dates to datetime
        self.data['reservation_status_date'] = pd.to_datetime(self.data['reservation_status_date'])
        
        return self.data
    
    def create_churn_label(self):
        """Create churn labels based on customer behavior"""
        if self.data is None:
            self.load_data()
            
        print("Creating churn labels...")
        
        # Group by customer identifiers
        # Since we don't have unique customer IDs, we'll use a combination of features
        # that might identify returning customers
        customer_features = ['country', 'adults', 'children', 'babies']
        
        # Sort by date
        self.data.sort_values('reservation_status_date', inplace=True)
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Identify unique "customers" using the features
        df['customer_id'] = df[customer_features].apply(lambda row: '_'.join(row.astype(str)), axis=1)
        
        # Calculate time between bookings for each customer
        customer_bookings = df.groupby('customer_id').agg({
            'reservation_status_date': list,
            'booking_id': 'count',
            'is_canceled': 'mean',
            'is_repeated_guest': 'max'
        }).reset_index()
        
        # Find customers with multiple bookings and calculate gaps
        customer_bookings['has_multiple_bookings'] = customer_bookings['booking_id'] > 1
        
        # For each customer, find the gap between their last booking and the end of the dataset
        max_date = df['reservation_status_date'].max()
        
        def calculate_return_gap(dates):
            if len(dates) <= 1:
                return None
            dates.sort()
            gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            return np.mean(gaps)
        
        def calculate_days_since_last_booking(dates, max_date):
            if len(dates) == 0:
                return None
            dates.sort()
            return (max_date - dates[-1]).days
        
        customer_bookings['avg_return_gap_days'] = customer_bookings['reservation_status_date'].apply(calculate_return_gap)
        customer_bookings['days_since_last_booking'] = customer_bookings['reservation_status_date'].apply(
            lambda dates: calculate_days_since_last_booking(dates, max_date)
        )
        
        # Define churned customers as those with:
        # - Multiple bookings historically
        # - Last booking was more than X days ago (exceeding their usual return gap)
        # - Not marked as repeated guest in their last booking
        
        def is_churned(row):
            # If they only had one booking, we can't tell if they churned
            if not row['has_multiple_bookings']:
                return None
                
            # If they're marked as a repeated guest, they're not churned
            if row['is_repeated_guest'] == 1:
                return 0
            
            # If their gap between bookings is significantly shorter than the time since their last booking
            if row['avg_return_gap_days'] is not None and row['days_since_last_booking'] is not None:
                if row['days_since_last_booking'] > row['avg_return_gap_days'] * 1.5:
                    return 1
                    
            # Otherwise, not churned
            return 0
        
        customer_bookings['is_churned'] = customer_bookings.apply(is_churned, axis=1)
        
        # Merge back to original data
        churn_mapping = dict(zip(customer_bookings['customer_id'], customer_bookings['is_churned']))
        df['is_churned'] = df['customer_id'].map(churn_mapping)
        
        # For customers with only one booking or unclear churn status, use heuristics:
        # - Canceled bookings are more likely to churn
        # - Customers who made special requests are less likely to churn
        # - Customers who had room changes or booking issues might be more likely to churn
        
        def heuristic_churn(row):
            if pd.isnull(row['is_churned']):
                # Basic heuristic: canceled bookings tend to churn
                if row['is_canceled'] == 1:
                    return 1
                    
                # Customers with booking changes and room changes might be dissatisfied
                if row['booking_changes'] > 0 and row['room_changed']:
                    return 1
                    
                # Customers with special requests show engagement
                if row['total_of_special_requests'] > 1:
                    return 0
                    
                # Default to not churned for unclear cases
                return 0
            return row['is_churned']
        
        df['is_churned'] = df.apply(heuristic_churn, axis=1)
        
        # Add back to the original data
        self.data['is_churned'] = df['is_churned']
        
        # Print churn statistics
        churn_rate = self.data['is_churned'].mean() * 100
        print(f"Overall churn rate: {churn_rate:.2f}%")
        print(f"Churned customers: {self.data['is_churned'].sum()}")
        print(f"Non-churned customers: {len(self.data) - self.data['is_churned'].sum()}")
        
        return self.data
    
    def explore_churn_data(self):
        """Explore and visualize the churn data for insights"""
        if 'is_churned' not in self.data.columns:
            self.create_churn_label()
            
        print("Exploring churn data...")
        
        # Basic statistics
        print("\nData shape:", self.data.shape)
        print("\nChurn rate:", self.data['is_churned'].mean())
        
        # Create visualizations
        # 1. Churn rate by hotel type
        plt.figure(figsize=(10, 6))
        sns.countplot(x='hotel_name', hue='is_churned', data=self.data)
        plt.title('Churn Rate by Hotel Type')
        plt.savefig('churn_by_hotel.png')
        
        # 2. Churn rate by customer type
        plt.figure(figsize=(10, 6))
        sns.countplot(x='customer_type', hue='is_churned', data=self.data)
        plt.title('Churn Rate by Customer Type')
        plt.savefig('churn_by_customer_type.png')
        
        # 3. Churn rate by deposit type
        plt.figure(figsize=(10, 6))
        sns.countplot(x='deposit_type', hue='is_churned', data=self.data)
        plt.title('Churn Rate by Deposit Type')
        plt.savefig('churn_by_deposit.png')
        
        # 4. Churn rate by market segment
        plt.figure(figsize=(12, 6))
        sns.countplot(x='market_segment', hue='is_churned', data=self.data)
        plt.title('Churn Rate by Market Segment')
        plt.xticks(rotation=45)
        plt.savefig('churn_by_market.png')
        
        # 5. Feature correlation with churn
        numerical_features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 
                             'adults', 'children', 'adr', 'total_stay_nights', 'total_guests', 
                             'previous_cancellations', 'previous_bookings_not_canceled',
                             'booking_changes', 'required_car_parking_spaces', 
                             'total_of_special_requests', 'days_in_waiting_list']
        
        corr = self.data[numerical_features + ['is_churned']].corr()
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation with Churn')
        plt.savefig('churn_feature_correlation.png')
        
        # 6. Churn by season
        plt.figure(figsize=(10, 6))
        sns.countplot(x='season', hue='is_churned', data=self.data)
        plt.title('Churn Rate by Season')
        plt.savefig('churn_by_season.png')
        
        print("Churn data exploration completed, visualizations saved.")
        
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        if 'is_churned' not in self.data.columns:
            self.create_churn_label()
                
        print("Preprocessing data...")
        
        # Print available columns to debug
        print("Available columns:", self.data.columns.tolist())
        
        # Define features and target
        X = self.data.drop(['booking_id', 'is_churned'], axis=1) 
        if 'reservation_status_date' in X.columns:
            X = X.drop('reservation_status_date', axis=1)
        
        y = self.data['is_churned']
        
        # Handle missing values explicitly
        print("Handling missing values...")
        for col in X.columns:
            # Check if column has missing values
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                print(f"Column {col} has {missing_count} missing values ({missing_count/len(X)*100:.2f}%)")
                
                if X[col].dtype.kind in 'bifc':  # boolean, integer, float, complex
                    # Replace missing numerical values with median
                    X[col] = X[col].fillna(X[col].median())
                else:
                    # Replace missing categorical values with most frequent value
                    X[col] = X[col].fillna(X[col].mode()[0])
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, categorical_cols, numerical_cols

    def build_model(self):
        """Build and train the churn prediction model without grid search"""
        X_train, X_test, y_train, y_test, categorical_cols, numerical_cols = self.preprocess_data()
        
        print("Building model pipeline...")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        # Create model pipeline with Gradient Boosting classifier
        # Using reasonable default hyperparameters
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,         # Number of boosting stages
                learning_rate=0.05,       # Conservative learning rate
                max_depth=5,              # Moderate tree depth
                random_state=42           # For reproducibility
            ))
        ])
        
        print("Training model with optimized parameters...")
        import time
        start_time = time.time()
        
        # Train the model
        model_pipeline.fit(X_train, y_train)
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        
        # Save the best model
        self.model = model_pipeline
        
        # Evaluate on test set
        self.evaluate_model(X_test, y_test)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model on test data"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        print("Evaluating model on test data...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churned', 'Churned'],
                   yticklabels=['Not Churned', 'Churned'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for Churn Prediction')
        plt.savefig('churn_confusion_matrix.png')
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for Churn Prediction')
        plt.legend(loc="lower right")
        plt.savefig('churn_roc_curve.png')
        
        # Feature importance
        if hasattr(self.model['classifier'], 'feature_importances_'):
            try:
                # Get feature names after preprocessing
                feature_names = (
                    numerical_cols + 
                    self.model['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols).tolist()
                )
                importances = self.model['classifier'].feature_importances_
            except:
                print("Could not get feature names from model, skipping feature importance plot")
                return
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importances for Churn Prediction')
            plt.bar(range(min(20, len(indices))), importances[indices[:20]], align='center')
            plt.xticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]], rotation=90)
            plt.tight_layout()
            plt.savefig('churn_feature_importance.png')
            
    def save_model(self, filename='models/churn_model.pkl'):
        """Save the trained model to disk"""
        if self.model is None:
            print("No model to save!")
            return False
            
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
        return True
    
    def load_model(self, filename='churn_model.pkl'):
        """Load a trained model from disk"""
        try:
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict_churn(self, customer_data):
        """
        Predict churn probability for customers
        
        Parameters:
        -----------
        customer_data: pandas DataFrame
            Customer data with the same features as used in training
            
        Returns:
        --------
        DataFrame with original data and churn probabilities
        """
        if self.model is None:
            print("Model not loaded or trained!")
            return None
            
        # Make predictions
        try:
            churn_probs = self.model.predict_proba(customer_data)[:, 1]
            result = customer_data.copy()
            result['churn_probability'] = churn_probs
            result['high_risk'] = churn_probs > 0.5
            
            return result
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None
    
    def generate_loyalty_recommendations(self, customer_data=None):
        """
        Generate loyalty recommendations based on churn predictions
        
        Parameters:
        -----------
        customer_data: pandas DataFrame, optional
            Customer data with churn probabilities, or None to use the training data
            
        Returns:
        --------
        DataFrame with recommendations for different customer segments
        """
        # If no customer data provided, use the model to predict on training data
        if customer_data is None:
            if 'is_churned' not in self.data.columns:
                self.create_churn_label()
                    
            X = self.data.drop(['booking_id', 'is_churned'], axis=1)
            if 'reservation_status_date' in X.columns:
                X = X.drop('reservation_status_date', axis=1)
            
            # Handle missing values again (same as in preprocess_data)
            for col in X.columns:
                if X[col].isnull().any():
                    if X[col].dtype.kind in 'bifc':  # boolean, integer, float, complex
                        X[col] = X[col].fillna(X[col].median())
                    else:
                        X[col] = X[col].fillna(X[col].mode()[0])
                        
            churn_probs = self.model.predict_proba(X)[:, 1]
            self.data['churn_probability'] = churn_probs
            customer_data = self.data
        
        # Create customer segments based on churn risk and value
        customer_data['total_value'] = customer_data['adr'] * customer_data['total_stay_nights']
        
        def get_segment(row):
            if row['churn_probability'] > 0.7:
                if row['total_value'] > customer_data['total_value'].quantile(0.75):
                    return 'High-Value High-Risk'
                else:
                    return 'Low-Value High-Risk'
            elif row['churn_probability'] > 0.3:
                if row['total_value'] > customer_data['total_value'].quantile(0.75):
                    return 'High-Value Medium-Risk'
                else:
                    return 'Low-Value Medium-Risk'
            else:
                if row['total_value'] > customer_data['total_value'].quantile(0.75):
                    return 'High-Value Loyal'
                else:
                    return 'Low-Value Loyal'
        
        customer_data['segment'] = customer_data.apply(get_segment, axis=1)
        
        # Generate recommendations for each segment
        recommendations = {
            'High-Value High-Risk': [
                'Send personalized win-back offers',
                'Offer room upgrades on next stay',
                'Provide dedicated customer service',
                'Implement regular check-ins from hotel manager',
                'Offer loyalty program benefits without full membership'
            ],
            'Low-Value High-Risk': [
                'Send targeted promotional offers',
                'Offer discounts for off-peak periods',
                'Invite to join loyalty program with sign-up incentive',
                'Implement satisfaction surveys to address concerns',
                'Provide special amenities on next stay'
            ],
            'High-Value Medium-Risk': [
                'Implement preemptive loyalty rewards',
                'Offer early check-in/late check-out',
                'Send personalized communications about hotel updates',
                'Provide anniversary or birthday special offers',
                'Create exclusive VIP events'
            ],
            'Low-Value Medium-Risk': [
                'Offer upsell packages to increase value',
                'Implement automated follow-up emails',
                'Provide targeted promotions for longer stays',
                'Offer discounts for direct bookings',
                'Create budget-friendly loyalty incentives'
            ],
            'High-Value Loyal': [
                'Implement recognition programs',
                'Offer premium loyalty benefits',
                'Create exclusive experiences',
                'Provide surprise upgrades',
                'Implement personalized concierge service'
            ],
            'Low-Value Loyal': [
                'Offer gradual loyalty incentives',
                'Implement referral programs',
                'Provide small recognition gestures',
                'Create seasonal special offers',
                'Offer package deals to increase value'
            ]
        }
        
        # Count customers in each segment
        segment_counts = customer_data['segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Customer Count']
        
        # Add recommendations to each segment
        segment_counts['Recommendations'] = segment_counts['Segment'].map(lambda x: ', '.join(recommendations[x][:3]))
        
        return segment_counts
    
    def run_pipeline(self):
        """Run the complete churn prediction pipeline"""
        print("Starting customer churn prediction pipeline...")
        
        try:
            # Load data
            self.load_data()
            
            # Create churn labels
            self.create_churn_label()
            
            # Explore data
            self.explore_churn_data()
            
            # Build and train model
            self.build_model()
            
            # Generate loyalty recommendations
            recommendations = self.generate_loyalty_recommendations()
            print("\nCustomer Segments and Recommendations:")
            print(recommendations)
            
            # Save model
            self.save_model()
            
            print("Customer churn prediction pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error in churn prediction pipeline: {str(e)}")
            return False
        finally:
            if self.conn:
                self.conn.close()
                print("Database connection closed")


# Example usage
if __name__ == "__main__":
    churn_model = CustomerChurnPrediction(db_path)
    churn_model.run_pipeline()