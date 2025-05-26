# 🛩️ Turbofan Engine RUL Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive web application for predicting Remaining Useful Life (RUL) of turbofan engines using a CNN-LSTM deep learning model. This system provides predictive maintenance capabilities for aerospace applications.

![Demo Screenshot](https://via.placeholder.com/800x400/1f77b4/ffffff?text=Turbofan+Engine+RUL+Prediction+System)

## 🎯 Live Demo

- **Demo Version** (No TensorFlow required): [Try it now!](http://localhost:8502)
- **Full Version** (With CNN-LSTM model): Requires local setup with TensorFlow

## 🌟 Features

- **Single Engine Prediction**: Analyze individual engine sensor data
- **Batch Processing**: Process multiple engines simultaneously
- **Interactive Visualizations**: Real-time charts and gauges
- **Multiple Input Methods**: Manual input, CSV upload, or sample data
- **Model Information**: Detailed architecture and performance metrics
- **Export Results**: Download predictions as CSV files

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- The trained model file: `best_cnn_lstm_model.h5`

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd turbofan-rul-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file is present**
   Make sure `best_cnn_lstm_model.h5` is in the same directory as `app.py`

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   The application will automatically open at `http://localhost:8501`

## 📊 Model Overview

### Architecture
- **Type**: CNN-LSTM Hybrid Neural Network
- **Input**: 30 time steps × 16 sensor features
- **Output**: Binary classification (Healthy vs Near Failure)
- **Performance**: 97.37% validation accuracy

### Key Components
- **Conv1D Layer**: 128 filters, kernel size 3
- **MaxPooling1D**: Pool size 2
- **LSTM Layer**: 128 units
- **Dropout**: 38% rate
- **Dense Output**: Sigmoid activation

## 🔧 Input Features

The model uses 16 key sensor measurements:

| Feature | Description | Type |
|---------|-------------|------|
| `op_setting_3` | Operational Setting 3 | Operational |
| `T2_fan_inlet_temp` | Fan Inlet Temperature | Temperature |
| `T24_LPC_outlet_temp` | LPC Outlet Temperature | Temperature |
| `T50_LPT_outlet_temp` | LPT Outlet Temperature | Temperature |
| `P2_fan_inlet_pressure` | Fan Inlet Pressure | Pressure |
| `P15_bypass_duct_pressure` | Bypass Duct Pressure | Pressure |
| `P30_HPC_outlet_pressure` | HPC Outlet Pressure | Pressure |
| `Nf_fan_speed` | Fan Speed | Speed |
| `Nc_core_speed` | Core Speed | Speed |
| `Ps30_HPC_static_pressure` | HPC Static Pressure | Pressure |
| `NRf_corrected_fan_speed` | Corrected Fan Speed | Speed |
| `NRc_corrected_core_speed` | Corrected Core Speed | Speed |
| `BPR_bypass_ratio` | Bypass Ratio | Other |
| `farB_burner_fuel_air_ratio` | Burner Fuel-Air Ratio | Other |
| `htBleed_bleed_enthalpy` | Bleed Enthalpy | Other |
| `W32_LPT_coolant_bleed` | LPT Coolant Bleed | Other |

## 📈 Usage Guide

### Single Engine Prediction

1. **Navigate to "Single Prediction"** in the sidebar
2. **Choose input method**:
   - **Manual Input**: Enter sensor values for multiple cycles
   - **Upload CSV**: Upload a file with sensor readings
   - **Use Sample Data**: Test with pre-generated healthy engine data
3. **Click "Predict RUL"** to get results
4. **View results**: Health status, failure probability, and visualizations

### Batch Processing

1. **Navigate to "Batch Prediction"** in the sidebar
2. **Upload CSV file** with columns:
   - `engine_id`: Unique identifier for each engine
   - All 16 sensor feature columns (see table above)
3. **Click "Run Batch Prediction"**
4. **Download results** as CSV file

### CSV File Format

For batch processing, your CSV should look like:

```csv
engine_id,op_setting_3,T2_fan_inlet_temp,T24_LPC_outlet_temp,...
1,0.5,0.2,0.3,...
1,0.6,0.25,0.35,...
2,0.4,0.15,0.28,...
```

## 🎯 Interpretation

### Prediction Results

- **Healthy (Class 0)**: RUL > 40 cycles, continue normal operation
- **Near Failure (Class 1)**: RUL ≤ 40 cycles, schedule maintenance

### Confidence Levels

- **0-25%**: Very low failure risk (Green)
- **25-50%**: Low failure risk (Yellow)
- **50-75%**: Moderate failure risk (Orange)
- **75-100%**: High failure risk (Red)

## 🔬 Technical Details

### Data Preprocessing

- All sensor values are normalized to [0, 1] range using MinMaxScaler
- Sequences of 30 consecutive time steps are used for prediction
- Missing data is handled by padding with the last available values

### Model Training

- **Dataset**: NASA Turbofan Engine Degradation Simulation (FD004)
- **Samples**: 61,249 sensor readings from 249 engines
- **Optimization**: Optuna hyperparameter tuning
- **Class Balancing**: Weighted loss function for imbalanced data

### Performance Metrics

| Metric | Healthy Engines | Near Failure | Overall |
|--------|----------------|--------------|---------|
| Precision | 99% | 90% | - |
| Recall | 97% | 97% | - |
| F1-Score | 98% | 93% | 97% |
| Accuracy | - | - | 97% |

## 🛠️ Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure `best_cnn_lstm_model.h5` is in the same directory as `app.py`

2. **Import errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **CSV upload issues**
   - Verify all required columns are present
   - Check for proper column naming (case-sensitive)
   - Ensure numeric data types for sensor values

4. **Memory issues**
   - For large batch files, process in smaller chunks
   - Close other applications to free up RAM

### Performance Tips

- Use sample data to test the application quickly
- For batch processing, limit to 1000 engines at a time
- Ensure stable internet connection for initial model loading

## 📚 References

- NASA Turbofan Engine Degradation Simulation Dataset
- TensorFlow/Keras Documentation
- Streamlit Documentation
- Plotly Visualization Library

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your input data format
3. Ensure all dependencies are properly installed

## 📄 License

This project is for educational and research purposes. Please ensure compliance with your organization's policies when using with real engine data.

---

**Built with ❤️ using Streamlit, TensorFlow, and Plotly** 