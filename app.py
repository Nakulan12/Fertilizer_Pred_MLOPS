from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
from datetime import datetime
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'agrismart-secret-key-2024'

# Initialize predictor
predictor = None
try:
    from src.Fertilizer_Pred.pipeline.prediction_pipeline import FertilizerPredictor
    predictor = FertilizerPredictor()
    logger.info("✅ ML Models loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading ML models: {e}")
    traceback.print_exc()

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict-form')
def predict_form():
    """Prediction form page"""
    return render_template('predict.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if not predictor:
            flash('ML models not available. Please contact support.', 'error')
            return redirect(url_for('predict_form'))

        # Extract form data
        try:
            input_data = {
                'temperature': float(request.form['temperature']),
                'humidity': float(request.form['humidity']),
                'soil_type': request.form['soil_type'].strip(),
                'nitrogen': float(request.form['nitrogen']),
                'potassium': float(request.form['potassium']),
                'phosphorous': float(request.form['phosphorous']),
                'ph': float(request.form['ph']),
                'rainfall': float(request.form['rainfall'])
            }
            
            # Validate ranges
            if not (0 <= input_data['humidity'] <= 100):
                raise ValueError("Humidity must be between 0 and 100%")
            if not (3 <= input_data['ph'] <= 10):
                raise ValueError("pH must be between 3 and 10")
            if input_data['temperature'] < -10 or input_data['temperature'] > 50:
                raise ValueError("Temperature must be between -10 and 50°C")
                
        except (ValueError, KeyError) as e:
            flash(f'Invalid input: {str(e)}', 'error')
            return redirect(url_for('predict_form'))

        # Get predictions
        logger.info(f"Making prediction with data: {input_data}")
        predictions = predictor.predict(input_data)
        logger.info(f"Predictions received: {predictions}")
        
        if not predictions:
            flash('Unable to generate predictions. Please check your input data.', 'error')
            return redirect(url_for('predict_form'))
        
        flash('Predictions generated successfully!', 'success')
        return render_template('results.html', 
                             predictions=predictions, 
                             input_data=input_data,
                             timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    except Exception as e:
        logger.error(f'Prediction error: {e}')
        traceback.print_exc()
        flash('Something went wrong during prediction. Please try again.', 'error')
        return redirect(url_for('predict_form'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if not predictor:
            return jsonify({
                'success': False,
                'error': 'ML models not available'
            }), 500

        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # Validate required fields
        required_fields = ['temperature', 'humidity', 'soil_type', 'nitrogen', 
                          'potassium', 'phosphorous', 'ph', 'rainfall']
        
        missing_fields = [field for field in required_fields if field not in data or data[field] == '']
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        # Prepare input data
        try:
            input_data = {
                'temperature': float(data['temperature']),
                'humidity': float(data['humidity']),
                'soil_type': str(data['soil_type']).strip(),
                'nitrogen': float(data['nitrogen']),
                'potassium': float(data['potassium']),
                'phosphorous': float(data['phosphorous']),
                'ph': float(data['ph']),
                'rainfall': float(data['rainfall'])
            }
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid input data: {str(e)}'
            }), 400

        # Get predictions
        predictions = predictor.predict(input_data)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'input_data': input_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"API Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred during prediction'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'ml_models': 'loaded' if predictor else 'not_loaded'
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    print("🌱 Starting AgriSmart Fertilizer Recommendation System")
    print(f"🔧 ML Models: {'✅ Loaded' if predictor else '❌ Not Loaded'}")
    print("🚀 Server starting on http://localhost:8080")
    
    app.run(host="0.0.0.0", port=8080, debug=True)
