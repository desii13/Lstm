# Air Quality Prediction System using LSTM

## Overview
This project implements a deep learning system for predicting air quality (specifically CO levels) using Long Short-Term Memory (LSTM) neural networks. The system includes a complete web application built with Flask that allows users to input environmental parameters and receive real-time CO level predictions.

## Features
- **LSTM Deep Learning Model**: Trained on historical air quality data
- **Real-time Prediction**: Web interface for instant CO level predictions
- **Data Visualization**: Historical data and model performance charts
- **Interactive Dashboard**: Modern, responsive web interface
- **Model Performance Metrics**: RMSE and MAE evaluation metrics

## Dataset
The system uses the Air Quality dataset containing:
- **Time Period**: March 2004 to April 2005
- **Frequency**: Hourly measurements
- **Features**: CO levels, temperature, humidity, NOx, NO2, and other environmental parameters
- **Target Variable**: CO(GT) - Carbon Monoxide levels

## Model Architecture
- **Type**: LSTM (Long Short-Term Memory) Neural Network
- **Architecture**: 
  - 3 LSTM layers (50 units each)
  - Dropout layers (0.2) for regularization
  - Dense output layer
- **Sequence Length**: 24 hours (using past 24 hours to predict next hour)
- **Features**: 6 environmental parameters
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone or download the project files
2. Install required packages:
```bash
pip install -r requirements.txt
```

### Files Structure
```
├── AirQuality.csv              # Original dataset
├── app.py                      # Flask web application
├── lstm_model.py              # LSTM model training script
├── data_analysis.py           # Data exploration script
├── templates/
│   └── index.html             # Web interface template
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── lstm_air_quality_model.h5 # Trained LSTM model
└── scaler.pkl                 # Data scaler for preprocessing
```

## Usage

### 1. Training the Model
Run the LSTM training script:
```bash
python lstm_model.py
```
This will:
- Load and preprocess the air quality data
- Train the LSTM model
- Save the trained model and scaler
- Generate performance visualizations

### 2. Running the Web Application
Start the Flask application:
```bash
python app.py
```
The application will be available at: `http://localhost:5000`

### 3. Using the Web Interface
1. Open your web browser and navigate to `http://localhost:5000`
2. Fill in the environmental parameters:
   - PT08.S1(CO): CO sensor reading
   - NOx(GT): Nitrogen oxides level
   - NO2(GT): Nitrogen dioxide level
   - Temperature: Environmental temperature (°C)
   - Relative Humidity: Humidity percentage
   - Absolute Humidity: Absolute humidity value
3. Click "Predict CO Level" to get the prediction
4. View historical data and model performance charts

## Model Performance
The trained LSTM model achieves:
- **RMSE**: ~0.79 mg/m³
- **MAE**: ~0.65 mg/m³

## Technical Details

### Data Preprocessing
- **Missing Value Handling**: Removed rows with -200 values (missing data indicator)
- **Data Scaling**: MinMaxScaler normalization
- **Sequence Creation**: 24-hour sliding window for time series prediction
- **Train/Validation/Test Split**: 60%/20%/20%

### Model Training
- **Epochs**: 50
- **Batch Size**: 32
- **Validation**: 20% of training data
- **Early Stopping**: Monitored validation loss

### Web Application Features
- **Real-time Prediction**: Instant CO level predictions
- **Historical Data Visualization**: Time series plots of past data
- **Model Performance Analysis**: Prediction vs actual comparison
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Charts**: Dynamic loading of visualizations

## API Endpoints

### POST /predict
Predict CO levels based on input parameters.
**Parameters:**
- pt08_s1_co: CO sensor reading
- nox_gt: NOx level
- no2_gt: NO2 level
- temperature: Temperature in °C
- humidity: Relative humidity %
- ah: Absolute humidity

**Response:**
```json
{
    "success": true,
    "prediction": 2.45,
    "message": "Predicted CO Level: 2.45 mg/m³"
}
```

### GET /historical_data
Get historical data visualization.

### GET /model_performance
Get model performance analysis and metrics.

## Future Enhancements
- Multi-step ahead prediction
- Additional air quality parameters (PM2.5, O3, etc.)
- Real-time data integration
- Mobile app development
- Advanced visualization features

## Dependencies
- TensorFlow 2.15.0
- Scikit-learn 1.3.0
- Pandas 2.0.3
- NumPy 1.24.3
- Matplotlib 3.7.2
- Flask 2.3.3
- Joblib 1.3.2

## License
This project is for educational purposes as part of the Final Assignment (TA-04) for Deep Learning course.

## Author
Student - Tugas Akhir 04 (TA-04): Prediksi Data Sekuensial Menggunakan LSTM
