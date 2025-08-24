#!/usr/bin/env python3
"""
Test script for the Network Intrusion Detection prediction system
"""

import pandas as pd
import numpy as np
from prediction_service import NetworkIntrusionPredictor
import os

def create_sample_data():
    """Create a sample CSV file with network traffic features for testing"""
    print("Creating sample test data...")
    
    # Create sample data with some of the common features from the dataset
    sample_data = {
        'Destination Port': [80, 443, 22, 21, 25, 53, 80, 443, 22, 80],
        'Flow Duration': [1000, 2000, 500, 1500, 800, 300, 1200, 1800, 600, 900],
        'Total Fwd Packets': [10, 15, 5, 8, 12, 3, 9, 14, 6, 11],
        'Total Backward Packets': [8, 12, 3, 6, 9, 2, 7, 11, 4, 8],
        'Total Length of Fwd Packets': [1500, 2200, 800, 1200, 1800, 450, 1350, 2100, 900, 1650],
        'Total Length of Bwd Packets': [1200, 1800, 600, 900, 1350, 300, 1050, 1650, 720, 1320],
        'Fwd Packet Length Max': [200, 300, 150, 180, 250, 100, 190, 280, 160, 220],
        'Fwd Packet Length Min': [50, 60, 40, 45, 55, 30, 48, 58, 42, 52],
        'Fwd Packet Length Mean': [150, 220, 120, 140, 180, 90, 145, 210, 125, 165],
        'Bwd Packet Length Max': [180, 250, 120, 160, 200, 80, 170, 240, 130, 190],
        'Bwd Packet Length Min': [40, 50, 30, 35, 45, 25, 38, 48, 32, 42],
        'Bwd Packet Length Mean': [120, 180, 90, 110, 150, 70, 115, 175, 95, 135],
        'Flow Bytes/s': [2500, 4000, 1400, 2100, 3200, 750, 2400, 3900, 1520, 2750],
        'Flow Packets/s': [18, 27, 8, 14, 21, 5, 16, 25, 10, 19],
        'Flow IAT Mean': [100, 150, 60, 90, 120, 40, 95, 145, 65, 105],
        'Flow IAT Std': [20, 30, 15, 18, 25, 10, 19, 28, 16, 22],
        'Flow IAT Max': [200, 300, 120, 180, 250, 80, 190, 290, 130, 210],
        'Flow IAT Min': [10, 15, 5, 8, 12, 3, 9, 14, 6, 11],
        'Fwd IAT Total': [900, 1350, 450, 675, 1080, 270, 855, 1305, 495, 945],
        'Fwd IAT Mean': [90, 135, 45, 68, 108, 27, 86, 131, 50, 95],
        'Fwd IAT Std': [18, 27, 12, 15, 22, 8, 17, 26, 13, 19],
        'Fwd IAT Max': [180, 270, 90, 135, 216, 54, 171, 261, 99, 189],
        'Fwd IAT Min': [9, 14, 4, 7, 11, 3, 8, 13, 5, 10],
        'Bwd IAT Total': [720, 1080, 360, 540, 864, 216, 684, 1044, 396, 756],
        'Bwd IAT Mean': [72, 108, 36, 54, 86, 22, 68, 104, 40, 76],
        'Bwd IAT Std': [14, 22, 9, 12, 17, 6, 14, 21, 10, 15],
        'Bwd IAT Max': [144, 216, 72, 108, 173, 43, 137, 209, 79, 151],
        'Bwd IAT Min': [7, 11, 4, 5, 9, 2, 7, 10, 4, 8],
        'Fwd PSH Flags': [1, 2, 0, 1, 1, 0, 1, 2, 0, 1],
        'Bwd PSH Flags': [1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
        'Fwd URG Flags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Bwd URG Flags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Fwd Header Length': [200, 300, 100, 160, 240, 60, 180, 280, 120, 220],
        'Bwd Header Length': [160, 240, 80, 120, 180, 48, 144, 224, 96, 176],
        'Fwd Packets/s': [10, 15, 5, 8, 12, 3, 9, 14, 6, 11],
        'Bwd Packets/s': [8, 12, 3, 6, 9, 2, 7, 11, 4, 8],
        'Min Packet Length': [40, 50, 30, 35, 45, 25, 38, 48, 32, 42],
        'Max Packet Length': [200, 300, 150, 180, 250, 100, 190, 280, 160, 220],
        'Packet Length Mean': [135, 200, 105, 125, 165, 85, 130, 195, 110, 150],
        'Packet Length Std': [25, 40, 18, 22, 32, 12, 24, 38, 20, 28],
        'Packet Length Variance': [625, 1600, 324, 484, 1024, 144, 576, 1444, 400, 784],
        'FIN Flag Count': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'SYN Flag Count': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'RST Flag Count': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'PSH Flag Count': [2, 3, 0, 1, 2, 0, 2, 3, 0, 2],
        'ACK Flag Count': [8, 12, 3, 6, 9, 2, 7, 11, 4, 8],
        'URG Flag Count': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'CWE Flag Count': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ECE Flag Count': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Down/Up Ratio': [0.8, 0.8, 0.6, 0.75, 0.75, 0.67, 0.78, 0.79, 0.67, 0.73],
        'Average Packet Size': [135, 200, 105, 125, 165, 85, 130, 195, 110, 150],
        'Avg Fwd Segment Size': [150, 220, 120, 140, 180, 90, 145, 210, 125, 165],
        'Avg Bwd Segment Size': [120, 180, 90, 110, 150, 70, 115, 175, 95, 135],
        'Fwd Header Length.1': [200, 300, 100, 160, 240, 60, 180, 280, 120, 220],
        'Fwd Avg Bytes/Bulk': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Fwd Avg Packets/Bulk': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Fwd Avg Bulk Rate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Bwd Avg Bytes/Bulk': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Bwd Avg Packets/Bulk': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Bwd Avg Bulk Rate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Subflow Fwd Packets': [10, 15, 5, 8, 12, 3, 9, 14, 6, 11],
        'Subflow Fwd Bytes': [1500, 2200, 800, 1200, 1800, 450, 1350, 2100, 900, 1650],
        'Subflow Bwd Packets': [8, 12, 3, 6, 9, 2, 7, 11, 4, 8],
        'Subflow Bwd Bytes': [1200, 1800, 600, 900, 1350, 300, 1050, 1650, 720, 1320],
        'Init_Win_bytes_forward': [8192, 16384, 4096, 8192, 12288, 2048, 8192, 16384, 4096, 10240],
        'Init_Win_bytes_backward': [8192, 16384, 4096, 8192, 12288, 2048, 8192, 16384, 4096, 10240],
        'act_data_pkt_fwd': [9, 14, 4, 7, 11, 2, 8, 13, 5, 10],
        'min_seg_size_forward': [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
        'Active Mean': [500, 750, 250, 375, 600, 150, 475, 725, 275, 525],
        'Active Std': [100, 150, 50, 75, 120, 30, 95, 145, 55, 105],
        'Active Max': [800, 1200, 400, 600, 960, 240, 760, 1160, 440, 840],
        'Active Min': [200, 300, 100, 150, 240, 60, 190, 290, 110, 210],
        'Idle Mean': [100, 150, 50, 75, 120, 30, 95, 145, 55, 105],
        'Idle Std': [20, 30, 10, 15, 24, 6, 19, 29, 11, 21],
        'Idle Max': [200, 300, 100, 150, 240, 60, 190, 290, 110, 210],
        'Idle Min': [50, 75, 25, 38, 60, 15, 48, 73, 28, 53]
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save to CSV
    test_file = 'sample_data/sample_submission.csv'
    df.to_csv(test_file, index=False)
    print(f"Sample data saved to: {test_file}")
    return test_file

def test_prediction_service():
    """Test the prediction service"""
    print("Testing Network Intrusion Detection Prediction Service")
    print("=" * 60)
    
    # Create sample data
    test_file = create_sample_data()
    
    # Initialize predictor
    predictor = NetworkIntrusionPredictor()
    
    # Test prediction
    print("\nTesting predictions...")
    results = predictor.predict(test_file)
    
    if results is not None:
        print("\n✅ Prediction successful!")
        print(f"Predictions shape: {results.shape}")
        print("\nFirst 5 predictions:")
        print(results[['Predicted_Attack_Type', 'Confidence_Score']].head())
        
        # Save results
        output_file = predictor.save_predictions(results)
        print(f"\nResults saved to: {output_file}")
        
        # Clean up
        os.remove(test_file)
        print(f"Cleaned up test file: {test_file}")
        
        return True
    else:
        print("❌ Prediction failed!")
        return False

if __name__ == "__main__":

    success = test_prediction_service()
    if success:
        print("\n🎉 Test completed successfully!")
        print("\nYou can now:")
        print("1. Run 'python web_app.py' to start the web interface")
        print("2. Use 'python prediction_service.py <csv_file>' for command line predictions")
    else:
        print("\n❌ Test failed. Please check if the model is trained first.")
        print("Run 'python training.py' to train the model.")
