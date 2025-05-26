# 🔧 TensorFlow Setup Guide

This guide will help you set up the full CNN-LSTM model with TensorFlow support.

## 🚨 Current Issue

You're currently running **Python 3.13.2**, but TensorFlow doesn't support Python 3.13 yet. TensorFlow currently supports Python 3.8-3.11.

## 🛠️ Solutions

### Option 1: Use pyenv to install Python 3.11 (Recommended)

1. **Install pyenv** (if not already installed):
   ```bash
   # On macOS with Homebrew
   brew install pyenv
   
   # Add to your shell profile (~/.zshrc or ~/.bash_profile)
   echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.zshrc
   echo 'eval "$(pyenv init -)"' >> ~/.zshrc
   source ~/.zshrc
   ```

2. **Install Python 3.11**:
   ```bash
   pyenv install 3.11.7
   pyenv local 3.11.7
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv_tensorflow
   source venv_tensorflow/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the full application**:
   ```bash
   streamlit run app.py
   ```

### Option 2: Use conda/miniconda

1. **Install miniconda** (if not already installed):
   ```bash
   # Download and install from https://docs.conda.io/en/latest/miniconda.html
   ```

2. **Create environment with Python 3.11**:
   ```bash
   conda create -n tensorflow_env python=3.11
   conda activate tensorflow_env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

### Option 3: Use Docker

1. **Create a Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
   ```

2. **Build and run**:
   ```bash
   docker build -t turbofan-rul .
   docker run -p 8501:8501 turbofan-rul
   ```

## 🧪 Testing the Setup

Once you have TensorFlow installed, test the model loading:

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Test model loading
model = tf.keras.models.load_model('best_cnn_lstm_model.h5')
print("Model loaded successfully!")
print("Model summary:")
model.summary()
```

## 📋 Requirements Verification

Make sure these packages are installed:

```bash
pip list | grep -E "(tensorflow|streamlit|numpy|pandas|plotly|scikit-learn)"
```

Expected output:
```
numpy                    1.24.0+
pandas                   2.0.0+
plotly                   5.15.0+
scikit-learn             1.3.0+
streamlit                1.28.0+
tensorflow               2.13.0+
```

## 🔄 Switching Between Versions

- **Demo version** (current): `streamlit run app_demo.py`
- **Full version** (with TensorFlow): `streamlit run app.py`

## 🐛 Troubleshooting

### TensorFlow Installation Issues

1. **macOS with Apple Silicon (M1/M2/M3)**:
   ```bash
   # Try installing with conda
   conda install -c apple tensorflow-deps
   pip install tensorflow-macos
   ```

2. **Memory issues**:
   ```bash
   # Set memory growth for GPU (if applicable)
   export TF_FORCE_GPU_ALLOW_GROWTH=true
   ```

3. **Version conflicts**:
   ```bash
   # Clean install
   pip uninstall tensorflow
   pip cache purge
   pip install tensorflow==2.13.0
   ```

### Model Loading Issues

1. **File not found**:
   - Ensure `best_cnn_lstm_model.h5` is in the same directory as `app.py`
   - Check file permissions

2. **Compatibility issues**:
   - The model was saved with TensorFlow 2.x
   - Ensure you're using TensorFlow 2.13.0 or later

## 🎯 Next Steps

1. Choose one of the setup options above
2. Install the required Python version
3. Install dependencies
4. Test the model loading
5. Run the full application with `streamlit run app.py`

## 📞 Support

If you encounter issues:
1. Check the Python version: `python --version`
2. Check TensorFlow installation: `python -c "import tensorflow; print(tensorflow.__version__)"`
3. Verify model file exists: `ls -la best_cnn_lstm_model.h5`

The demo version (`app_demo.py`) will continue to work with your current Python 3.13 setup for interface testing and development. 