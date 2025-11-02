import os
import pandas as pd
import pickle
import glob

def load_latest_model(model_dir='models_checkpoint/binary'):
    """Load the most recent trained model and encoders."""
    model_files = glob.glob(os.path.join(model_dir, 'LogisticRegression_trained_model_*.pkl'))
    if not model_files:
        raise FileNotFoundError("No trained model found. Please train the model first.")
    
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model: {latest_model}")
    
    with open(latest_model, 'rb') as f:
        saved_data = pickle.load(f)
    
    return (
        saved_data['model'],
        saved_data['label_encoder'],
        saved_data['protocol_encoder'],
        saved_data['state_encoder']
    )

def load_prediction_data(data_path, model_features=None):
    """Load data for prediction and ensure correct feature order."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    
    # If the data includes labels, separate them
    y = None
    if 'label' in data.columns:
        y = data['label']
        X = data.drop('label', axis=1)
    else:
        X = data
    
    # Ensure the features match the model's expected features
    if model_features is not None:
        # Find missing and extra features
        missing_features = set(model_features) - set(X.columns)
        extra_features = set(X.columns) - set(model_features)
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing from input data")
            # Add missing features with default value of 0
            for feature in missing_features:
                X[feature] = 0
                
        if extra_features:
            print(f"Warning: Dropping {len(extra_features)} unexpected features")
            X = X.drop(columns=list(extra_features))
        
        # Reorder columns to match model's expected order
        X = X[model_features]
    
    return X, y

def make_predictions(model, X, label_encoder):
    """Make predictions using the loaded model."""
    print("\nMaking predictions...")
    
    # Get prediction probabilities
    proba = model.predict_proba(X)
    
    # Get class predictions
    y_pred = model.predict(X)
    
    # Create results DataFrame with explicit class labels
    results = pd.DataFrame({
        'predicted_prob_benign': proba[:, 0],
        'predicted_prob_attack': proba[:, 1],
        'predicted_label': y_pred,
        'predicted_class': ['BENIGN' if x == 0 else 'ATTACK' for x in y_pred]
    })
    
    return results

def main():
    try:
        print("=" * 60)
        print("BINARY CLASSIFICATION - LOGISTIC REGRESSION PREDICTION")
        print("=" * 60)
        
        # Load the latest trained model and encoders
        model, label_encoder, _, _ = load_latest_model()
        
        # Get the feature names the model was trained on
        model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        
        # Example: Load data for prediction
        data_path = 'data/processed_data_binary/X_test_scaled_binary_class_*.csv'
        latest_file = max(glob.glob(data_path), key=os.path.getctime)
        
        # Load and preprocess the data with feature validation
        X, y_true = load_prediction_data(latest_file, model_features=model_features)
        
        # Convert to numpy array to avoid feature name issues
        X_array = X.values if hasattr(X, 'values') else X
        
        # Make predictions
        predictions = make_predictions(model, X_array, label_encoder)
        
        # Display first few predictions
        print("\nSample predictions:")
        print(predictions.head())
        
        # If true labels are available, show accuracy
        if y_true is not None:
            accuracy = (predictions['predicted_label'] == y_true).mean()
            print(f"\nPrediction Accuracy: {accuracy:.2%}")
        
        # Save predictions
        output_file = 'predictions_results.csv'
        predictions.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {os.path.abspath(output_file)}")
        
        print("\n" + "=" * 60)
        print("PREDICTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
