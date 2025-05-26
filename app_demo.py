import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pickle
import os

# Configure page
st.set_page_config(
    page_title="Turbofan Engine RUL Prediction",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-healthy {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .prediction-failure {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    .demo-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Feature names used in the model
FEATURES = [
    'op_setting_3', 'T2_fan_inlet_temp', 'T24_LPC_outlet_temp', 'T50_LPT_outlet_temp',
    'P2_fan_inlet_pressure', 'P15_bypass_duct_pressure', 'P30_HPC_outlet_pressure',
    'Nf_fan_speed', 'Nc_core_speed', 'Ps30_HPC_static_pressure', 'NRf_corrected_fan_speed',
    'NRc_corrected_core_speed', 'BPR_bypass_ratio', 'farB_burner_fuel_air_ratio',
    'htBleed_bleed_enthalpy', 'W32_LPT_coolant_bleed'
]

# Feature descriptions for better user understanding
FEATURE_DESCRIPTIONS = {
    'op_setting_3': 'Operational Setting 3',
    'T2_fan_inlet_temp': 'Fan Inlet Temperature (T2)',
    'T24_LPC_outlet_temp': 'LPC Outlet Temperature (T24)',
    'T50_LPT_outlet_temp': 'LPT Outlet Temperature (T50)',
    'P2_fan_inlet_pressure': 'Fan Inlet Pressure (P2)',
    'P15_bypass_duct_pressure': 'Bypass Duct Pressure (P15)',
    'P30_HPC_outlet_pressure': 'HPC Outlet Pressure (P30)',
    'Nf_fan_speed': 'Fan Speed (Nf)',
    'Nc_core_speed': 'Core Speed (Nc)',
    'Ps30_HPC_static_pressure': 'HPC Static Pressure (Ps30)',
    'NRf_corrected_fan_speed': 'Corrected Fan Speed (NRf)',
    'NRc_corrected_core_speed': 'Corrected Core Speed (NRc)',
    'BPR_bypass_ratio': 'Bypass Ratio (BPR)',
    'farB_burner_fuel_air_ratio': 'Burner Fuel-Air Ratio (farB)',
    'htBleed_bleed_enthalpy': 'Bleed Enthalpy (htBleed)',
    'W32_LPT_coolant_bleed': 'LPT Coolant Bleed (W32)'
}

# Typical ranges for each feature (for validation and default values)
FEATURE_RANGES = {
    'op_setting_3': (0.0, 1.0),
    'T2_fan_inlet_temp': (0.0, 1.0),
    'T24_LPC_outlet_temp': (0.0, 1.0),
    'T50_LPT_outlet_temp': (0.0, 1.0),
    'P2_fan_inlet_pressure': (0.0, 1.0),
    'P15_bypass_duct_pressure': (0.0, 1.0),
    'P30_HPC_outlet_pressure': (0.0, 1.0),
    'Nf_fan_speed': (0.0, 1.0),
    'Nc_core_speed': (0.0, 1.0),
    'Ps30_HPC_static_pressure': (0.0, 1.0),
    'NRf_corrected_fan_speed': (0.0, 1.0),
    'NRc_corrected_core_speed': (0.0, 1.0),
    'BPR_bypass_ratio': (0.0, 1.0),
    'farB_burner_fuel_air_ratio': (0.0, 1.0),
    'htBleed_bleed_enthalpy': (0.0, 1.0),
    'W32_LPT_coolant_bleed': (0.0, 1.0)
}

def mock_predict_rul(sensor_data):
    """
    Mock prediction function that simulates the CNN-LSTM model behavior
    This is for demonstration purposes when TensorFlow is not available
    """
    # Convert sensor data to DataFrame if it's a list
    if isinstance(sensor_data, list):
        df = pd.DataFrame(sensor_data)
    else:
        df = sensor_data
    
    # Simple heuristic based on sensor values to simulate realistic predictions
    # Higher values in temperature and pressure sensors indicate more stress
    temp_features = ['T2_fan_inlet_temp', 'T24_LPC_outlet_temp', 'T50_LPT_outlet_temp']
    pressure_features = ['P2_fan_inlet_pressure', 'P15_bypass_duct_pressure', 'P30_HPC_outlet_pressure']
    
    # Calculate stress indicators
    temp_stress = df[temp_features].mean().mean()
    pressure_stress = df[pressure_features].mean().mean()
    overall_stress = (temp_stress + pressure_stress) / 2
    
    # Add some randomness but keep it realistic
    np.random.seed(int(overall_stress * 1000) % 100)
    noise = np.random.normal(0, 0.1)
    
    # Calculate failure probability (higher stress = higher probability)
    probability = min(max(overall_stress + noise, 0.0), 1.0)
    
    # Adjust probability based on realistic patterns
    if overall_stress < 0.3:
        probability = np.random.uniform(0.05, 0.25)  # Healthy range
    elif overall_stress < 0.6:
        probability = np.random.uniform(0.2, 0.5)   # Moderate range
    else:
        probability = np.random.uniform(0.6, 0.95)  # High stress range
    
    prediction_class = int(probability > 0.5)
    
    return probability, prediction_class

def create_sequences(data, seq_len=30):
    """Create sequences for the CNN-LSTM model (mock version)"""
    if len(data) < seq_len:
        # If we don't have enough data, pad with the last available values
        padding_needed = seq_len - len(data)
        last_row = data.iloc[-1:].values
        padding = np.repeat(last_row, padding_needed, axis=0)
        padded_data = np.vstack([padding, data.values])
        return padded_data.reshape(1, seq_len, len(FEATURES))
    else:
        # Use the last seq_len rows
        return data.iloc[-seq_len:].values.reshape(1, seq_len, len(FEATURES))

def main():
    # Demo warning
    st.markdown('''
    <div class="demo-warning">
        <h4>🚧 Demo Mode</h4>
        <p>This is a demonstration version running without TensorFlow. The predictions are generated using a mock algorithm for interface demonstration. To use the actual CNN-LSTM model, please install TensorFlow in a Python 3.8-3.11 environment.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🛩️ Turbofan Engine RUL Prediction System</h1>', unsafe_allow_html=True)
    
    # st.success("✅ Demo mode loaded successfully!")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Single Prediction", "Batch Prediction", "Model Information"])
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    else:
        model_info_page()

def single_prediction_page():
    st.markdown('<h2 class="sub-header">Single Engine Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Sensor Readings Input")
        
        # Create input method selection
        input_method = st.radio("Choose input method:", ["Manual Input", "Upload CSV", "Use Sample Data"])
        
        sensor_data = []
        
        if input_method == "Manual Input":
            st.info("Enter sensor readings for the last 30 cycles (or as many as available)")
            
            # Number of cycles to input
            num_cycles = st.slider("Number of cycles to input:", min_value=1, max_value=30, value=5)
            
            # Create tabs for each cycle
            tabs = st.tabs([f"Cycle {i+1}" for i in range(num_cycles)])
            
            for i, tab in enumerate(tabs):
                with tab:
                    cycle_data = {}
                    cols = st.columns(2)
                    
                    for j, feature in enumerate(FEATURES):
                        col_idx = j % 2
                        with cols[col_idx]:
                            min_val, max_val = FEATURE_RANGES[feature]
                            default_val = (min_val + max_val) / 2
                            
                            cycle_data[feature] = st.number_input(
                                f"{FEATURE_DESCRIPTIONS[feature]}",
                                min_value=min_val,
                                max_value=max_val,
                                value=default_val,
                                step=0.01,
                                key=f"{feature}_cycle_{i}"
                            )
                    
                    sensor_data.append(cycle_data)
        
        elif input_method == "Upload CSV":
            st.info("Upload a CSV file with sensor readings. The file should contain columns for each sensor.")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Check if required features are present
                    missing_features = [f for f in FEATURES if f not in df.columns]
                    if missing_features:
                        st.error(f"Missing required features: {missing_features}")
                    else:
                        st.success("✅ CSV file loaded successfully!")
                        st.dataframe(df[FEATURES].head())
                        
                        # Convert to list of dictionaries
                        sensor_data = df[FEATURES].to_dict('records')
                        
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
        
        else:  # Use Sample Data
            st.info("Using sample healthy engine data")
            # Create sample data (healthy engine)
            np.random.seed(42)
            sample_data = []
            for i in range(10):
                cycle_data = {}
                for feature in FEATURES:
                    # Generate realistic sample values (more towards healthy range)
                    if 'temp' in feature.lower():
                        cycle_data[feature] = np.random.uniform(0.1, 0.4)
                    elif 'pressure' in feature.lower():
                        cycle_data[feature] = np.random.uniform(0.2, 0.6)
                    elif 'speed' in feature.lower():
                        cycle_data[feature] = np.random.uniform(0.3, 0.7)
                    else:
                        cycle_data[feature] = np.random.uniform(0.1, 0.8)
                sample_data.append(cycle_data)
            sensor_data = sample_data
    
    with col2:
        st.subheader("🔮 Prediction Results")
        
        if sensor_data and st.button("🚀 Predict RUL", type="primary"):
            # Make prediction using mock function
            probability, prediction_class = mock_predict_rul(sensor_data)
            
            if probability is not None:
                # Display results
                if prediction_class == 0:
                    st.markdown(f'''
                    <div class="prediction-healthy">
                        <h3>✅ Engine Status: HEALTHY</h3>
                        <p><strong>Failure Probability:</strong> {probability:.1%}</p>
                        <p><strong>Remaining Useful Life:</strong> > 40 cycles</p>
                        <p>The engine is operating within normal parameters.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="prediction-failure">
                        <h3>⚠️ Engine Status: NEAR FAILURE</h3>
                        <p><strong>Failure Probability:</strong> {probability:.1%}</p>
                        <p><strong>Remaining Useful Life:</strong> ≤ 40 cycles</p>
                        <p>Immediate maintenance recommended!</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Failure Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance visualization
                st.subheader("📈 Sensor Readings Analysis")
                
                if len(sensor_data) > 1:
                    # Show trend of last few readings
                    df_plot = pd.DataFrame(sensor_data)
                    df_plot['cycle'] = range(1, len(df_plot) + 1)
                    
                    # Select top 6 most variable features for plotting
                    feature_vars = df_plot[FEATURES].var().sort_values(ascending=False)
                    top_features = feature_vars.head(6).index.tolist()
                    
                    fig = px.line(df_plot, x='cycle', y=top_features, 
                                title="Sensor Trends (Top 6 Most Variable)")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Current readings table
                st.subheader("📋 Current Sensor Readings")
                current_readings = pd.DataFrame([sensor_data[-1]]).T
                current_readings.columns = ['Value']
                current_readings.index = [FEATURE_DESCRIPTIONS[idx] for idx in current_readings.index]
                st.dataframe(current_readings, use_container_width=True)

def batch_prediction_page():
    st.markdown('<h2 class="sub-header">Batch Engine Prediction</h2>', unsafe_allow_html=True)
    
    st.info("Upload a CSV file containing sensor data for batch prediction.")
    st.markdown("""
    **📋 CSV Format Requirements:**
    - **Required**: All 16 sensor feature columns (see Model Information page for details)
    - **Optional**: `engine_id` column (will be added automatically if missing)
    - **Note**: If your CSV contains data from multiple engines, include an `engine_id` column to identify each engine
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file for batch prediction", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            
            # Check required sensor columns (engine_id is optional)
            missing_features = [col for col in FEATURES if col not in df.columns]
            
            if missing_features:
                st.error(f"Missing required sensor features: {missing_features}")
                st.info("💡 **Required sensor features**: " + ", ".join(FEATURES))
                st.info("📝 **Note**: The 'engine_id' column is optional and will be added automatically if missing.")
                return
            
            # Handle missing engine_id column (add it if not present)
            if 'engine_id' not in df.columns:
                st.info("ℹ️ No 'engine_id' column found. Adding default engine ID for batch processing...")
                df.insert(0, 'engine_id', 1)  # Add engine_id column with default value 1
                st.success("✅ Added default engine_id column")
            
            st.dataframe(df.head())
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                unique_engines = df['engine_id'].unique()
                
                for i, engine_id in enumerate(unique_engines):
                    engine_data = df[df['engine_id'] == engine_id][FEATURES]
                    
                    if len(engine_data) > 0:
                        probability, prediction_class = mock_predict_rul(engine_data)
                        
                        if probability is not None:
                            results.append({
                                'Engine ID': engine_id,
                                'Status': 'Healthy' if prediction_class == 0 else 'Near Failure',
                                'Failure Probability': f"{probability:.1%}",
                                'Recommended Action': 'Continue Operation' if prediction_class == 0 else 'Schedule Maintenance'
                            })
                    
                    progress_bar.progress((i + 1) / len(unique_engines))
                
                # Display results with color coding
                results_df = pd.DataFrame(results)
                st.subheader("📊 Batch Prediction Results")
                
                # Apply color styling to the dataframe
                def color_status(val):
                    if val == 'Healthy':
                        return 'background-color: #d4edda; color: #155724; font-weight: bold'
                    elif val == 'Near Failure':
                        return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                    return ''
                
                def color_probability(val):
                    # Extract percentage value
                    prob_val = float(val.strip('%')) / 100
                    if prob_val <= 0.3:
                        return 'background-color: #d4edda; color: #155724'
                    elif prob_val <= 0.5:
                        return 'background-color: #fff3cd; color: #856404'
                    else:
                        return 'background-color: #f8d7da; color: #721c24'
                
                def color_action(val):
                    if val == 'Continue Operation':
                        return 'background-color: #cce5ff; color: #004085; font-weight: bold'
                    elif val == 'Schedule Maintenance':
                        return 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
                    return ''
                
                # Style the dataframe
                styled_df = results_df.style.applymap(color_status, subset=['Status']) \
                                           .applymap(color_probability, subset=['Failure Probability']) \
                                           .applymap(color_action, subset=['Recommended Action'])
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    healthy_count = len(results_df[results_df['Status'] == 'Healthy'])
                    st.metric("Healthy Engines", healthy_count)
                
                with col2:
                    failure_count = len(results_df[results_df['Status'] == 'Near Failure'])
                    st.metric("Engines Near Failure", failure_count)
                
                with col3:
                    total_engines = len(results_df)
                    failure_rate = (failure_count / total_engines * 100) if total_engines > 0 else 0
                    st.metric("Failure Rate", f"{failure_rate:.1f}%")
                
                # Visualization with custom colors
                status_counts = results_df['Status'].value_counts()
                
                # Define colors matching our theme
                colors = []
                for status in status_counts.index:
                    if status == 'Healthy':
                        colors.append('#28a745')  # Green for healthy
                    elif status == 'Near Failure':
                        colors.append('#dc3545')  # Red for failure
                    else:
                        colors.append('#6c757d')  # Gray for unknown
                
                fig = px.pie(values=status_counts.values, names=status_counts.index, 
                           title="Engine Health Distribution",
                           color_discrete_sequence=colors)
                
                # Update layout for better appearance
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"engine_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def model_info_page():
    st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Architecture")
        st.markdown("""
        **CNN-LSTM Hybrid Model**
        - **Input Layer**: 30 time steps × 16 features
        - **Conv1D Layer**: 128 filters, kernel size 3, ReLU activation
        - **MaxPooling1D**: Pool size 2
        - **LSTM Layer**: 128 units
        - **Dropout**: 38% dropout rate
        - **Dense Output**: 1 unit, sigmoid activation
        
        **Training Details**:
        - **Optimizer**: Adam (learning rate: 0.00325)
        - **Loss Function**: Binary crossentropy
        - **Batch Size**: 32
        - **Epochs**: 20
        - **Validation Accuracy**: 97.37%
        """)
    
    with col2:
        st.subheader("📊 Model Performance")
        st.markdown("""
        **Classification Report**:
        - **Healthy Engines (Class 0)**:
          - Precision: 99%
          - Recall: 97%
          - F1-Score: 98%
        
        - **Near Failure (Class 1)**:
          - Precision: 90%
          - Recall: 97%
          - F1-Score: 93%
        
        - **Overall Accuracy**: 97%
        - **Weighted F1-Score**: 97%
        """)
    
    st.subheader("🔧 Sensor Features")
    
    # Create a nice table of features
    feature_info = []
    for feature in FEATURES:
        feature_info.append({
            'Feature Code': feature,
            'Description': FEATURE_DESCRIPTIONS[feature],
            'Type': 'Operational' if 'op_setting' in feature else 
                   'Temperature' if any(temp in feature for temp in ['T2', 'T24', 'T50']) else
                   'Pressure' if any(press in feature for press in ['P2', 'P15', 'P30', 'Ps30']) else
                   'Speed' if any(speed in feature for speed in ['Nf', 'Nc', 'NRf', 'NRc']) else
                   'Other'
        })
    
    feature_df = pd.DataFrame(feature_info)
    st.dataframe(feature_df, use_container_width=True)
    
    st.subheader("ℹ️ About the Dataset")
    st.markdown("""
    This model was trained on the **NASA Turbofan Engine Degradation Simulation Dataset (FD004)**:
    
    - **Total Samples**: 61,249 sensor readings
    - **Engines**: 249 different turbofan engines
    - **Features**: 16 selected sensor measurements
    - **Target**: Binary classification (Healthy vs Near Failure)
    - **Threshold**: RUL ≤ 40 cycles indicates near failure
    
    The dataset simulates realistic engine degradation patterns and is widely used for 
    predictive maintenance research in aerospace applications.
    """)
    
    st.subheader("🎯 Use Cases")
    st.markdown("""
    **Primary Applications**:
    - **Predictive Maintenance**: Schedule maintenance before failure occurs
    - **Fleet Management**: Monitor multiple engines simultaneously  
    - **Cost Optimization**: Reduce unplanned downtime and maintenance costs
    - **Safety Enhancement**: Prevent catastrophic failures through early detection
    
    **Industries**:
    - Commercial Aviation
    - Military Aircraft
    - Industrial Gas Turbines
    - Power Generation
    """)
    
    st.subheader("🚧 Demo Mode Information")
    st.markdown("""
    **Current Status**: Running in demo mode without TensorFlow
    
    **To use the actual CNN-LSTM model**:
    1. Install Python 3.8-3.11 (TensorFlow doesn't support 3.13 yet)
    2. Install TensorFlow: `pip install tensorflow`
    3. Run the original `app.py` file
    
    **Demo Prediction Algorithm**:
    - Uses sensor stress indicators to simulate realistic predictions
    - Temperature and pressure sensors are weighted more heavily
    - Includes randomness to demonstrate various scenarios
    - Results are representative but not from the actual trained model
    """)

if __name__ == "__main__":
    main() 