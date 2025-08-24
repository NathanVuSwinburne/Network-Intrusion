#!/usr/bin/env python3
"""
Network Intrusion Detection - Prediction Service
Loads trained model and makes predictions on new CSV files
"""

import pandas as pd
import numpy as np
import pickle
import glob
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NetworkIntrusionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.selected_features = None
        self.attack_map = {
            'BENIGN': 'BENIGN',
            'DDoS': 'DDoS',
            'DoS Hulk': 'DoS',
            'DoS GoldenEye': 'DoS',
            'DoS slowloris': 'DoS',
            'DoS Slowhttptest': 'DoS',
            'PortScan': 'Port Scan',
            'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force',
            'Bot': 'Bot',
            'Web Attack – Brute Force': 'Web Attack',
            'Web Attack – XSS': 'Web Attack',
            'Web Attack – Sql Injection': 'Web Attack',
            'Infiltration': 'Infiltration',
            'Heartbleed': 'Heartbleed'
        }
        
    def load_model(self):
        """Load the latest trained model and preprocessing components"""
        try:
            # Find the latest model files
            model_files = glob.glob('model_checkpoint/trained_model_*.pkl')
            scaler_files = glob.glob('model_checkpoint/scaler_*.pkl')
            le_files = glob.glob('processed_data/label_encoder_*.pkl')
            feature_files = glob.glob('processed_data/selected_features_*.txt')
            
            if not model_files or not scaler_files or not le_files or not feature_files:
                raise FileNotFoundError("Model files not found. Please run training.py first.")
            
            # Get the latest files
            latest_model = max(model_files, key=os.path.getctime)
            latest_scaler = max(scaler_files, key=os.path.getctime)
            latest_le = max(le_files, key=os.path.getctime)
            latest_features = max(feature_files, key=os.path.getctime)
            
            print(f"Loading model: {latest_model}")
            print(f"Loading scaler: {latest_scaler}")
            print(f"Loading label encoder: {latest_le}")
            print(f"Loading features: {latest_features}")
            
            # Load the model
            with open(latest_model, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load the scaler
            with open(latest_scaler, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load the label encoder
            with open(latest_le, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load selected features
            with open(latest_features, 'r') as f:
                self.selected_features = [line.strip() for line in f.readlines()]
            
            print(f"Model loaded successfully!")
            print(f"Number of features: {len(self.selected_features)}")
            print(f"Attack types: {list(self.label_encoder.classes_)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_data(self, data):
        """Preprocess new data to match training data format"""
        try:
            print(f"Input data shape: {data.shape}")
            
            # Remove leading/trailing whitespace from column names
            data.columns = [col.strip() for col in data.columns]
            
            # Handle infinite values
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values with median
            for col in data.select_dtypes(include=[np.number]).columns:
                if data[col].isnull().sum() > 0:
                    median_val = data[col].median()
                    data[col] = data[col].fillna(median_val)
            
            # If there's a Label column, map it to Attack Type
            if 'Label' in data.columns:
                data['Attack Type'] = data['Label'].map(self.attack_map)
                data = data.drop('Label', axis=1)
            
            # Select only numeric columns (excluding target columns)
            drop_cols = {'Attack Type', 'y_multi'}
            numeric_cols = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]
            X = data[numeric_cols].copy()
            
            # Select only the features that were used in training
            missing_features = [f for f in self.selected_features if f not in X.columns]
            if missing_features:
                print(f"Warning: Missing features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
                # Add missing features with zeros
                for feature in missing_features:
                    X[feature] = 0
            
            # Select and reorder features to match training
            X = X[self.selected_features]
            
            # Convert to float32 for consistency
            X = X.astype(np.float32)
            
            print(f"Preprocessed data shape: {X.shape}")
            return X
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return None
    
    def predict(self, csv_file_path):
        """Make predictions on a CSV file"""
        try:
            if self.model is None:
                print("Model not loaded. Loading model...")
                if not self.load_model():
                    return None
            
            # Load the CSV file
            print(f"Loading data from: {csv_file_path}")
            data = pd.read_csv(csv_file_path)
            
            # Preprocess the data
            X = self.preprocess_data(data)
            if X is None:
                return None
            
            # Scale the data
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            print("Making predictions...")
            predictions = self.model.predict(X_scaled)
            prediction_proba = self.model.predict_proba(X_scaled)
            
            # Convert predictions to attack type names
            attack_types = self.label_encoder.inverse_transform(predictions)
            
            # Get confidence scores (max probability for each prediction)
            confidence_scores = np.max(prediction_proba, axis=1)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'Row_Index': range(len(predictions)),
                'Predicted_Attack_Type': attack_types,
                'Predicted_Class_Index': predictions,
                'Confidence_Score': confidence_scores
            })
            
            # Add probability scores for each class
            for i, class_name in enumerate(self.label_encoder.classes_):
                results[f'Prob_{class_name}'] = prediction_proba[:, i]
            
            print(f"Predictions completed for {len(results)} samples")
            
            # Print summary
            print("\nPrediction Summary:")
            print(results['Predicted_Attack_Type'].value_counts())
            print(f"\nAverage confidence: {confidence_scores.mean():.3f}")
            
            return results
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None
    
    def save_predictions(self, results, output_file=None):
        """Save predictions to CSV file"""
        try:
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f'predictions_{timestamp}.csv'
            
            results.to_csv(output_file, index=False)
            print(f"Predictions saved to: {os.path.abspath(output_file)}")
            return output_file
            
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
            return None

def main():
    """Main function for command line usage"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python prediction_service.py <csv_file_path>")
        return
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        return
    
    # Create predictor and make predictions
    predictor = NetworkIntrusionPredictor()
    results = predictor.predict(csv_file)
    
    if results is not None:
        # Save predictions
        output_file = predictor.save_predictions(results)
        print(f"\nPrediction completed successfully!")
        print(f"Results saved to: {output_file}")
    else:
        print("Prediction failed!")

if __name__ == "__main__":
    main()
