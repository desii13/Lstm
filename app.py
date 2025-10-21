from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import joblib
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained model and scaler
print("Loading model and scaler...")
model = load_model('lstm_air_quality_model.h5')
scaler = joblib.load('scaler.pkl')
print("Model and scaler loaded successfully!")

# Load and preprocess the original data for visualization
def load_and_preprocess_data():
    df = pd.read_csv('AirQuality.csv', sep=';')
    
    # Remove unnamed columns
    df = df.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1)
    
    # Clean and convert data types
    numeric_columns = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    
    other_numeric = ['PT08.S1(CO)', 'NMHC(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 
                     'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
    for col in other_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert Date and Time columns
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
    df = df.set_index('DateTime')
    df = df.drop(['Date', 'Time'], axis=1)
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Filter out -200 values
    target_col = 'CO(GT)'
    feature_cols = ['PT08.S1(CO)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']
    
    df_clean = df[(df[target_col] != -200) & (df[target_col] > 0)].copy()
    for col in feature_cols:
        df_clean = df_clean[(df_clean[col] != -200) & (df_clean[col] > 0)]
    
    return df_clean

# Load data
df_clean = load_and_preprocess_data()

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :-1])  # All features except target
        y.append(data[i, -1])  # Target variable (CO levels)
    return np.array(X), np.array(y)

def make_prediction(features):
    """Make prediction using the trained LSTM model"""
    # Scale the input features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Reshape for LSTM (samples, timesteps, features)
    features_reshaped = features_scaled.reshape(1, 1, len(features))
    
    # Make prediction
    prediction_scaled = model.predict(features_reshaped, verbose=0)
    
    # Inverse transform the prediction
    dummy_array = np.zeros((1, len(features) + 1))
    dummy_array[0, -1] = prediction_scaled[0, 0]
    prediction = scaler.inverse_transform(dummy_array)[0, -1]
    
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form (accept comma or dot)
        def to_float(x: str) -> float:
            return float((x or '').replace(',', '.'))

        pt08_s1_co = to_float(request.form.get('pt08_s1_co', ''))
        nox_gt = to_float(request.form.get('nox_gt', ''))
        no2_gt = to_float(request.form.get('no2_gt', ''))
        temperature = to_float(request.form.get('temperature', ''))
        humidity = to_float(request.form.get('humidity', ''))
        ah = to_float(request.form.get('ah', ''))

        # Basic range validation
        if pt08_s1_co < 0 or nox_gt < 0 or no2_gt < 0 or ah < 0 or not (0 <= humidity <= 100):
            return jsonify({
                'success': False,
                'error': 'Invalid value ranges. Ensure PT08/NOx/NO2/AH >= 0 and RH between 0-100%.'
            }), 400
        
        # Create feature array
        features = np.array([pt08_s1_co, nox_gt, no2_gt, temperature, humidity, ah])
        
        # Make prediction
        prediction = make_prediction(features)
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'message': f'Predicted CO Level: {round(prediction, 2)} mg/m³'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/historical_data')
def historical_data():
    """Generate historical data visualization"""
    try:
        # Get the last 500 data points for visualization
        recent_data = df_clean.tail(500)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot CO levels over time
        plt.subplot(2, 1, 1)
        plt.plot(recent_data.index, recent_data['CO(GT)'], 'b-', linewidth=1, alpha=0.7)
        plt.title('Historical CO Levels Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('CO Levels (mg/m³)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Plot temperature
        plt.subplot(2, 1, 2)
        plt.plot(recent_data.index, recent_data['T'], 'r-', linewidth=1, alpha=0.7)
        plt.title('Historical Temperature Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'image': img_base64
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/model_performance')
def model_performance():
    """Generate model performance visualization"""
    try:
        # Prepare data for performance evaluation
        target_col = 'CO(GT)'
        feature_cols = ['PT08.S1(CO)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']
        
        data = df_clean[feature_cols + [target_col]].values
        data_scaled = scaler.fit_transform(data)
        
        # Create sequences
        seq_length = 24
        X, y = create_sequences(data_scaled, seq_length)
        
        # Split data (use last 20% for testing)
        test_size = int(len(X) * 0.2)
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        
        # Make predictions
        y_pred = model.predict(X_test, verbose=0)
        
        # Inverse transform predictions and actual values
        dummy_array_pred = np.zeros((len(y_pred), len(feature_cols) + 1))
        dummy_array_pred[:, -1] = y_pred.flatten()
        y_pred_inv = scaler.inverse_transform(dummy_array_pred)[:, -1]
        
        dummy_array_actual = np.zeros((len(y_test), len(feature_cols) + 1))
        dummy_array_actual[:, -1] = y_test
        y_test_inv = scaler.inverse_transform(dummy_array_actual)[:, -1]
        
        # Create performance plots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Predictions vs Actual (first 100 samples)
        plt.subplot(2, 2, 1)
        n_plot = min(100, len(y_test_inv))
        plt.plot(y_test_inv[:n_plot], label='Actual CO Levels', alpha=0.7, linewidth=1)
        plt.plot(y_pred_inv[:n_plot], label='Predicted CO Levels', alpha=0.7, linewidth=1)
        plt.title('Predictions vs Actual (First 100 samples)', fontsize=12, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('CO Levels (mg/m³)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        plt.subplot(2, 2, 2)
        plt.scatter(y_test_inv, y_pred_inv, alpha=0.5, s=20)
        min_val = min(y_test_inv.min(), y_pred_inv.min())
        max_val = max(y_test_inv.max(), y_pred_inv.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('Actual CO Levels (mg/m³)')
        plt.ylabel('Predicted CO Levels (mg/m³)')
        plt.title('Actual vs Predicted CO Levels', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        plt.subplot(2, 2, 3)
        residuals = y_test_inv - y_pred_inv
        plt.scatter(y_pred_inv, residuals, alpha=0.5, s=20)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted CO Levels (mg/m³)')
        plt.ylabel('Residuals (mg/m³)')
        plt.title('Residual Plot', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Distribution of errors
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals (mg/m³)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'metrics': {
                'rmse': round(rmse, 4),
                'mae': round(mae, 4)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
