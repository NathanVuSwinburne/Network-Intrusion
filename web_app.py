#!/usr/bin/env python3
"""
Network Intrusion Detection - Web Application
Flask web interface for uploading CSV files and getting predictions
"""

from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import pandas as pd
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from prediction_service import NetworkIntrusionPredictor
import tempfile

app = Flask(__name__)
app.secret_key = 'network_intrusion_detection_2024'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global predictor instance
predictor = NetworkIntrusionPredictor()

@app.route('/')
def index():
    """Main page with file upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and make predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, filename)
        file.save(temp_file_path)
        
        # Make predictions
        results = predictor.predict(temp_file_path)
        
        if results is None:
            return jsonify({'error': 'Failed to make predictions. Please check your CSV format.'}), 500
        
        # Clean up temp file
        os.remove(temp_file_path)
        os.rmdir(temp_dir)
        
        # Convert results to JSON-serializable format
        results_dict = results.to_dict('records')
        
        # Calculate summary statistics
        summary = {
            'total_samples': len(results),
            'attack_distribution': results['Predicted_Attack_Type'].value_counts().to_dict(),
            'average_confidence': float(results['Confidence_Score'].mean()),
            'high_confidence_count': int((results['Confidence_Score'] > 0.8).sum()),
            'low_confidence_count': int((results['Confidence_Score'] < 0.5).sum())
        }
        
        return jsonify({
            'success': True,
            'predictions': results_dict,
            'summary': summary,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/download_results', methods=['POST'])
def download_results():
    """Download prediction results as CSV"""
    try:
        data = request.get_json()
        predictions = data.get('predictions', [])
        
        if not predictions:
            return jsonify({'error': 'No predictions to download'}), 400
        
        # Convert back to DataFrame
        df = pd.DataFrame(predictions)
        
        # Save to temporary file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = f'predictions_{timestamp}.csv'
        df.to_csv(temp_file, index=False)
        
        return send_file(temp_file, as_attachment=True, download_name=temp_file)
        
    except Exception as e:
        return jsonify({'error': f'Error downloading results: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        if predictor.model is None:
            predictor.load_model()
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': predictor.model is not None,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    if predictor.load_model():
        print("Model loaded successfully!")
    else:
        print("Warning: Model not loaded. Please run training.py first.")
    
    print("Starting web application...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
