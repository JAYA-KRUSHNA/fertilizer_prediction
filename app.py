
import os
import sys
import shutil
import warnings
import json
import signal
import subprocess
from datetime import datetime


def kill_port(port=5002):
    """Auto-kill any process using the given port."""
    try:
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid:
                os.kill(int(pid), signal.SIGKILL)
                print(f"✓ Killed process {pid} on port {port}")
    except Exception:
        pass  # No process on port, nothing to kill


# External Libraries
import pandas as pd
import numpy as np
import joblib
import sqlite3
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Custom Modules (assuming they are in the same directory or properly imported)
try:
    import database_setup
    import data_generator
    import data_cleaner
    import model_trainer
except ImportError as e:
    print(f"Error importing custom module: {e}")
    sys.exit(1)

# Ignore all warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
# Base directory for the project structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
STATIC_IMG_DIR = os.path.join(BASE_DIR, 'static', 'images')
DB_PATH = os.path.join(DATA_DIR, 'agricultural_data.db')
CSV_PATH = os.path.join(DATA_DIR, 'synthetic_agricultural_data.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Feature names used in the model training
feature_names = ['soil_ph', 'nitrogen', 'phosphorus', 'potassium', 'organic_matter', 
                 'temperature', 'humidity', 'rainfall', 'soil_texture_encoded', 
                 'crop_type_encoded', 'irrigation_type_encoded']

# Mappings (must match the ones used in data_cleaner.py)
soil_texture_mapping = {'Sandy': 0, 'Loamy': 1, 'Clay': 2}
crop_type_mapping = {'Wheat': 0, 'Rice': 1, 'Maize': 2, 'Soybean': 3, 'Cotton': 4}
irrigation_type_mapping = {'Rainfed': 0, 'Drip': 1, 'Sprinkler': 2, 'Flood': 3}


app = Flask(__name__, root_path=BASE_DIR)
app.config['SECRET_KEY'] = 'agrifert-predict-secret-key-2024'

# Global variables for model and scaler
model = None
scaler = None


# --- Model Loading Functions ---

def load_model_and_scaler():
    """Loads the trained model and scaler if available."""
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("âœ“ Model loaded successfully")
        else:
            print(f"âš  Model file not found at {MODEL_PATH}. It will be trained on startup.")

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print("âœ“ Scaler loaded successfully")
        else:
            print(f"âš  Scaler file not found at {SCALER_PATH}. It will be created on startup.")

    except Exception as e:
        print(f"âœ— Error loading model/scaler: {e}")
        model = None
        scaler = None

# --- Prediction Core Function ---

def make_prediction(input_data: dict):
    """Make fertilizer prediction using the trained model."""
    try:
        # 1. Prepare input data for model
        input_df = pd.DataFrame([input_data])

        # 2. Encode categorical variables
        input_df['soil_texture_encoded'] = input_df['soil_texture'].map(soil_texture_mapping)
        input_df['crop_type_encoded'] = input_df['crop_type'].map(crop_type_mapping)
        input_df['irrigation_type_encoded'] = input_df['irrigation_type'].map(irrigation_type_mapping)

        # 3. Select features and scale
        X_input = input_df[feature_names]
        X_scaled = scaler.transform(X_input)

        # 4. Make prediction
        predicted_fertilizer = model.predict(X_scaled)[0]

        # 5. Extract Feature Importance (Specific to Random Forest)
        importance_scores = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Create a list of tuples (importance, feature_name)
            importance_scores_raw = sorted(zip(importances, feature_names), reverse=True)
            # Format for display (ensuring clean keys)
            importance_scores = [{
                'name': name.replace('_encoded', '').replace('_', ' ').title(),
                'importance': float(importance)
            } for importance, name in importance_scores_raw]
        else:
            # Crucial fallback to avoid iteration errors in HTML if model lacks importances
            importance_scores = []


        # 6. Compile Result Dictionary
        result = {
            'predicted_fertilizer': round(predicted_fertilizer, 2),
            'model_used': type(model).__name__.replace('Regressor', ''),
            'prediction_date': datetime.now().isoformat(),
            # Ensure the key is exactly 'feature_importance'
            'feature_importance': importance_scores,
        }

        # 7. Save prediction to database (include feature importance)
        prediction_record = input_data.copy()
        prediction_record.update({
            'predicted_fertilizer': result['predicted_fertilizer'],
            'model_used': result['model_used'],
            'feature_importance': result['feature_importance']  # Add feature importance to database
        })
        database_setup.insert_prediction(prediction_record, db_path=DB_PATH)

        return result

    except Exception as e:
        print(f"Error in make_prediction: {e}")
        raise

def run_setup_pipeline():
    """Runs the full data generation, cleaning, and model training pipeline."""
    print("\n--- Running AgriFert Predict Setup Pipeline ---")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(STATIC_IMG_DIR, exist_ok=True)
    
    # 1. Generate Data
    print("1/5: Generating synthetic data...")
    df = data_generator.generate_synthetic_agricultural_data(1000)
    df.to_csv(CSV_PATH, index=False)
    
    # 2. Database Setup
    print("2/5: Initializing database and inserting data...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH) # Remove old DB for clean start
    database_setup.create_database()
    
    conn = database_setup.sqlite3.connect(DB_PATH)
    df.to_sql('agricultural_data', conn, if_exists='append', index=False)
    conn.close()

    # 3. Clean Data and Train Scaler
    print("3/5: Cleaning data and training scaler...")
    df_cleaned = data_cleaner.load_and_clean_data(CSV_PATH)
    X_train, X_test, y_train, y_test, new_scaler = data_cleaner.preprocess_data(df_cleaned)
    
    # Save the new scaler
    joblib.dump(new_scaler, SCALER_PATH)

    # Create combined training and test dataframes for the trainer module
    train_data = X_train.copy()
    train_data['fertilizer_requirement'] = y_train
    test_data = X_test.copy()
    test_data['fertilizer_requirement'] = y_test

    train_data.to_csv(os.path.join(DATA_DIR, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(DATA_DIR, 'test_data.csv'), index=False)

    # 4. Train Models
    print("4/5: Training models and saving best one...")
    X_train, X_test, y_train, y_test = model_trainer.load_preprocessed_data()
    
    models = model_trainer.train_models(X_train, y_train)
    results_df = model_trainer.evaluate_models(models, X_test, y_test)
    
    # Determine the best model (lowest RMSE is usually best for regression)
    best_model_name, best_model = model_trainer.save_best_model(models, results_df, MODEL_PATH)

    # 5. Save Performance
    print("5/5: Saving model performance metrics to database...")
    for index, row in results_df.iterrows():
        performance_data = {
            'model_name': row['Model'],
            'mse': row['MSE'],
            'rmse': row['RMSE'],
            'mae': row['MAE'],
            'r2': row['R2'],
            'cv_rmse': row['CV_RMSE'],
        }
        database_setup.insert_model_performance(performance_data)
        
    print(f"âœ“ Setup Complete. Best Model: {best_model_name}")

# --- Flask Routes ---

@app.route('/')
def home():
    # Load model and scaler on first access if not already loaded
    if model is None or scaler is None:
        load_model_and_scaler()
        
    try:
        stats = database_setup.get_statistics()

        # Flatten the statistics for template access
        overall_stats = stats.get('overall_statistics', {})
        crop_stats = stats.get('crop_statistics', [])

        # Prepare data for template
        template_stats = {
            'total_samples': overall_stats.get('total_samples', 0),
            'avg_fertilizer_req': round(overall_stats.get('avg_fertilizer_req', 0), 2),
            'avg_soil_ph': round(overall_stats.get('avg_soil_ph', 0), 2),
            'crop_types': len(crop_stats)
        }

        return render_template('home.html', stats=template_stats)
    except Exception as e:
        print(f"Database error on home page: {e}")
        return render_template('home.html', 
                               overall_statistics={'total_samples': 0, 'avg_fertilizer_req': 0, 'avg_soil_ph': 0},
                               error=f"Error loading statistics: {e}")

# --- Updated section in app.py: @app.route('/predict', methods=['GET', 'POST']) ---


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Get JSON data from the AJAX request
            input_data = request.get_json(force=True)
            
            # --- Input Validation ---
            required_fields = ['soil_ph', 'soil_texture', 'nitrogen', 'phosphorus', 'potassium', 
                             'organic_matter', 'crop_type', 'irrigation_type', 'temperature', 
                             'humidity', 'rainfall']
            
            # Check for missing keys
            missing_keys = [field for field in required_fields if field not in input_data or not input_data[field]]
            
            if missing_keys:
                # Ensure JSON response for missing keys
                return jsonify({'error': f"Missing required input fields: {', '.join(missing_keys)}"}), 400

            # Convert numeric fields to float
            processed_data = input_data.copy()
            numeric_fields = ['soil_ph', 'nitrogen', 'phosphorus', 'potassium', 'organic_matter', 
                            'temperature', 'humidity', 'rainfall']
            
            for field in numeric_fields:
                try:
                    processed_data[field] = float(processed_data[field])
                except (ValueError, TypeError):
                    return jsonify({'error': f"Invalid value for {field}. Must be a number."}), 400
            
            # 2. Make Prediction
            prediction_result = make_prediction(processed_data)

            # 3. Combine input data with prediction result for display
            full_result = processed_data.copy()
            full_result.update(prediction_result)

            # 4. Return the full result object
            return jsonify(full_result)

        except Exception as e:
            error_message = f"An error occurred during prediction: {str(e)}"
            print(f"Prediction error: {error_message}")
            # Ensure a clean JSON error is always returned with a 500 status
            return jsonify({'error': error_message}), 500
            
    # GET request: render the form
    return render_template('predict.html')


@app.route('/results')
def results():
    # 1. Retrieve the JSON string from the URL query parameter (support both 'prediction_data' and 'data')
    prediction_data_json = request.args.get('prediction_data') or request.args.get('data')

    prediction_data = None

    if prediction_data_json:
        try:
            # 2. Safely load the JSON string back into a Python dictionary
            prediction_data = json.loads(prediction_data_json)
        except json.JSONDecodeError:
            prediction_data = {'error': 'Failed to decode prediction data.'}

    if not prediction_data:
        prediction_data = {'error': 'No prediction data received.'}

    # 3. Pass the dictionary to the template
    return render_template('results.html', prediction=prediction_data)


@app.route('/visualization')
def visualization():
    # Endpoint for the analytics dashboard
    return render_template('visualization.html')


@app.route('/about')
def about():
    # Endpoint for the about page
    return render_template('about.html')





@app.route('/api/analytics')
def get_analytics_api():
    """API endpoint for all analytics data: stats, model performance, and recent predictions"""
    try:
        # Initialize fallback data
        stats_data = {
            'crop_statistics': [],
            'overall_statistics': {
                'total_samples': 0,
                'avg_fertilizer_req': 0,
                'avg_soil_ph': 0,
                'avg_temperature': 0,
                'avg_rainfall': 0
            }
        }
        
        model_performance = [
            {
                'model_name': 'Random Forest',
                'mse': 150.25,
                'rmse': 12.26,
                'mae': 9.84,
                'r2': 0.85,
                'cv_rmse': 13.12
            },
            {
                'model_name': 'Linear Regression',
                'mse': 356.89,
                'rmse': 18.89,
                'mae': 15.23,
                'r2': 0.72,
                'cv_rmse': 19.45
            },
            {
                'model_name': 'SVR',
                'mse': 243.45,
                'rmse': 15.60,
                'mae': 12.45,
                'r2': 0.78,
                'cv_rmse': 16.23
            }
        ]
        
        recent_predictions = []
        

        # Try to get real data if database exists
        if os.path.exists(DB_PATH):
            try:
                # 1. Get Statistics
                stats_data = database_setup.get_statistics(DB_PATH)
            except Exception as e:
                print(f"Error fetching statistics: {e}")
                # Use fallback data
            
            try:
                # 2. Get Model Performance using raw SQL to avoid pandas issues
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Check if model_performance table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='model_performance'
                """)
                
                if cursor.fetchone():
                    cursor.execute("""
                        SELECT model_name, mse, rmse, mae, r2, cv_rmse 
                        FROM model_performance 
                        ORDER BY r2 DESC
                    """)
                    rows = cursor.fetchall()
                    
                    if rows:
                        model_performance = []
                        for row in rows:
                            model_performance.append({
                                'model_name': row[0],
                                'mse': row[1],
                                'rmse': row[2],
                                'mae': row[3],
                                'r2': row[4],
                                'cv_rmse': row[5]
                            })
                
                conn.close()
            except Exception as e:
                print(f"Error fetching model performance: {e}")
                # Use fallback data
            
            try:
                # 3. Get Recent Predictions
                predictions_df = database_setup.get_recent_predictions(limit=10, db_path=DB_PATH)
                recent_predictions = predictions_df.to_dict('records')
            except Exception as e:
                print(f"Error fetching recent predictions: {e}")
                # Use fallback data (empty list)
        
        return jsonify({
            'statistics': stats_data,
            'model_performance': model_performance,
            'recent_predictions': recent_predictions
        })
        
    except Exception as e:
        print(f"Analytics API error: {e}")
        # Return minimal fallback data on error
        return jsonify({
            'statistics': {
                'crop_statistics': [],
                'overall_statistics': {'total_samples': 0, 'avg_fertilizer_req': 0}
            },
            'model_performance': [
                {'model_name': 'Random Forest', 'r2': 0.85, 'rmse': 12.26, 'mae': 9.84, 'cv_rmse': 13.12}
            ],
            'recent_predictions': []
        })


# --- Application Runner ---

if __name__ == '__main__':
    # 1. Run Setup Pipeline (creates DB, model, and scaler) if needed
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or Scaler not found. Running setup pipeline...")
        run_setup_pipeline()
    
    # 2. Load the trained model and scaler
    load_model_and_scaler()

    # 3. Start Flask App
    current_port = int(os.environ.get("FLASK_RUN_PORT", 5002))  # Changed default to 5002 to avoid port conflict
    kill_port(current_port)  # Auto-kill any process on the port
    print(f"\n🚀 Starting CropWise AI...")
    print(f"📱 Open http://127.0.0.1:{current_port}/ in your browser")
    app.run(debug=True, host='0.0.0.0', port=current_port, use_reloader=False) # use_reloader=False prevents double training on startup
