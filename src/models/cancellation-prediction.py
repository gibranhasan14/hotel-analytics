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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import joblib

db_path = "data/hotel_dw.db" 

class CancellationPredictionModel:
    """
    Model to predict hotel booking cancellations
    """
    
    def __init__(self, db_path=db_path):
        """
        Initialize the cancellation prediction model
        
        Parameters:
        -----------
        db_path: str
            Path to the SQLite database for the data warehouse
        """
        self.db_path = db_path
        self.conn = None
        self.data = None
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
            
        # SQL query to join fact table with dimensions
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
            bf.revenue,
            
            dd.year,
            dd.month,
            dd.season,
            dd.is_weekend,
            
            hd.hotel_name,
            hd.meal_plan,
            hd.market_segment,
            hd.distribution_channel,
            
            cd.country,
            cd.customer_type,
            cd.deposit_type,
            
            rd.reserved_room_type,
            rd.assigned_room_type,
            rd.room_changed
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
        
        return self.data
        
    def explore_data(self):
        """Explore and visualize the data for insights"""
        if self.data is None:
            self.load_data()
            
        print("Exploring data...")
        
        # Basic statistics
        print("\nData shape:", self.data.shape)
        print("\nCancellation rate:", self.data['is_canceled'].mean())
        
        # Create visualizations
        # 1. Cancellation rate by hotel type
        plt.figure(figsize=(10, 6))
        sns.countplot(x='hotel_name', hue='is_canceled', data=self.data)
        plt.title('Cancellation Rate by Hotel Type')
        plt.savefig('cancellation_by_hotel.png')
        
        # 2. Cancellation rate by lead time buckets
        self.data['lead_time_bucket'] = pd.cut(
            self.data['lead_time'], 
            bins=[0, 7, 30, 90, 180, 365, float('inf')],
            labels=['0-7 days', '8-30 days', '31-90 days', '91-180 days', '181-365 days', '366+ days']
        )
        
        plt.figure(figsize=(12, 6))
        sns.countplot(x='lead_time_bucket', hue='is_canceled', data=self.data)
        plt.title('Cancellation Rate by Lead Time')
        plt.xticks(rotation=45)
        plt.savefig('cancellation_by_lead_time.png')
        
        # 3. Cancellation rate by deposit type
        plt.figure(figsize=(10, 6))
        sns.countplot(x='deposit_type', hue='is_canceled', data=self.data)
        plt.title('Cancellation Rate by Deposit Type')
        plt.savefig('cancellation_by_deposit.png')
        
        # 4. Cancellation rate by market segment
        plt.figure(figsize=(12, 6))
        sns.countplot(x='market_segment', hue='is_canceled', data=self.data)
        plt.title('Cancellation Rate by Market Segment')
        plt.xticks(rotation=45)
        plt.savefig('cancellation_by_market.png')
        
        # 5. Feature correlation with cancellation
        numerical_features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 
                             'adults', 'children', 'adr', 'total_stay_nights', 'total_guests', 
                             'previous_cancellations', 'previous_bookings_not_canceled',
                             'booking_changes', 'required_car_parking_spaces', 
                             'total_of_special_requests', 'days_in_waiting_list']
        
        corr = self.data[numerical_features + ['is_canceled']].corr()
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation with Cancellation')
        plt.savefig('feature_correlation.png')
        
        print("Data exploration completed, visualizations saved.")
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        if self.data is None:
            self.load_data()
            
        print("Preprocessing data...")
        print("Available columns:", self.data.columns.tolist())
        
        # Define features and target
        X = self.data.drop(['booking_id', 'is_canceled'], axis=1)
        y = self.data['is_canceled']
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Handle missing values
        print("Handling missing values...")
        for col in X.columns:
            # Check if column has missing values
            if X[col].isnull().any():
                print(f"Column {col} has {X[col].isnull().sum()} missing values")
                
                if col in numerical_cols:
                    # Replace missing numerical values with median
                    X[col] = X[col].fillna(X[col].median())
                else:
                    # Replace missing categorical values with most frequent value
                    X[col] = X[col].fillna(X[col].mode()[0])
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, categorical_cols, numerical_cols
    
    def build_model(self):
        """Build and train the prediction model"""
        X_train, X_test, y_train, y_test, categorical_cols, numerical_cols = self.preprocess_data()
        
        print("Building model pipeline...")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        # Create model pipeline with RandomForest classifier
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Define hyperparameter grid for tuning
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
        
        # Use cross-validation to find best hyperparameters
        print("Performing grid search for hyperparameter tuning...")
        grid_search = GridSearchCV(
            model_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Save the best model
        self.model = grid_search.best_estimator_
        
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
                   xticklabels=['Not Canceled', 'Canceled'],
                   yticklabels=['Not Canceled', 'Canceled'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        
        # Feature importance
        if hasattr(self.model['classifier'], 'feature_importances_'):
            # Get feature names after preprocessing
            try:
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
            plt.title('Feature Importances')
            plt.bar(range(min(20, len(indices))), importances[indices[:20]], align='center')
            plt.xticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]], rotation=90)
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            
    def save_model(self, filename='models/cancellation_model.pkl'):
        """Save the trained model to disk"""
        if self.model is None:
            print("No model to save!")
            return False
            
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
        return True
    
    def load_model(self, filename='cancellation_model.pkl'):
        """Load a trained model from disk"""
        try:
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict_cancellation(self, booking_data):
        """
        Predict cancellation probability for new bookings
        
        Parameters:
        -----------
        booking_data: pandas DataFrame
            New booking data with the same features as used in training
            
        Returns:
        --------
        DataFrame with original data and cancellation probabilities
        """
        if self.model is None:
            print("Model not loaded or trained!")
            return None
            
        # Make predictions
        try:
            cancellation_probs = self.model.predict_proba(booking_data)[:, 1]
            result = booking_data.copy()
            result['cancellation_probability'] = cancellation_probs
            result['high_risk'] = cancellation_probs > 0.5
            
            return result
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None
    
    def run_pipeline(self):
        """Run the complete modeling pipeline"""
        print("Starting cancellation prediction pipeline...")
        
        try:
            # Load data
            self.load_data()
            
            # Explore data
            self.explore_data()
            
            # Build and train model
            self.build_model()
            
            # Save model
            self.save_model()
            
            print("Cancellation prediction pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error in cancellation prediction pipeline: {str(e)}")
            return False
        finally:
            if self.conn:
                self.conn.close()
                print("Database connection closed")


# Example usage
if __name__ == "__main__":
    model = CancellationPredictionModel(db_path)
    model.run_pipeline()