# Contributing to Turbofan Engine RUL Prediction System

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/turbofan-rul-prediction.git
   cd turbofan-rul-prediction
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements_demo.txt  # For demo version
   # OR
   pip install -r requirements.txt       # For full version (requires Python 3.8-3.11)
   ```

## 🛠️ Development Setup

### Demo Version (Recommended for development)
```bash
streamlit run app_demo.py
```

### Full Version (Requires TensorFlow)
```bash
streamlit run app.py
```

## 📝 How to Contribute

### Reporting Bugs
- Use the GitHub issue tracker
- Include detailed steps to reproduce
- Provide system information (Python version, OS, etc.)

### Suggesting Features
- Open an issue with the "enhancement" label
- Describe the feature and its use case
- Discuss implementation approach

### Code Contributions

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow PEP 8 style guidelines
   - Add docstrings to functions
   - Include comments for complex logic

3. **Test your changes**:
   - Test both demo and full versions
   - Verify UI functionality
   - Check with sample data

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: your descriptive commit message"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**:
   - Provide a clear description
   - Reference any related issues
   - Include screenshots for UI changes

## 🎨 Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add type hints where appropriate
- Keep functions focused and small
- Use docstrings for all functions and classes

### Example:
```python
def predict_rul(sensor_data: pd.DataFrame) -> tuple[float, int]:
    """
    Predict remaining useful life for engine sensor data.
    
    Args:
        sensor_data: DataFrame containing sensor readings
        
    Returns:
        tuple: (probability, prediction_class)
    """
    # Implementation here
    pass
```

## 📁 Project Structure

```
turbofan-rul-prediction/
├── app.py                    # Main Streamlit app (with TensorFlow)
├── app_demo.py              # Demo version (without TensorFlow)
├── requirements.txt         # Full version dependencies
├── requirements_demo.txt    # Demo version dependencies
├── best_cnn_lstm_model.h5   # Trained model file
├── sample_engine_data.csv   # Sample data for testing
├── README.md               # Main documentation
├── SETUP_TENSORFLOW.md     # TensorFlow setup guide
├── CONTRIBUTING.md         # This file
├── LICENSE                 # MIT License
└── .gitignore             # Git ignore rules
```

## 🧪 Testing Guidelines

### Manual Testing Checklist
- [ ] Demo app loads without errors
- [ ] All three input methods work (Manual, CSV, Sample)
- [ ] Predictions generate realistic results
- [ ] Batch processing works with sample CSV
- [ ] Visualizations render correctly
- [ ] Download functionality works
- [ ] Model information page displays correctly

### Test Data
Use the provided `sample_engine_data.csv` for testing batch functionality.

## 🔧 Areas for Contribution

### High Priority
- [ ] Add unit tests
- [ ] Improve error handling
- [ ] Add data validation
- [ ] Performance optimization
- [ ] Mobile responsiveness

### Medium Priority
- [ ] Add more visualization options
- [ ] Implement data export formats (Excel, JSON)
- [ ] Add model comparison features
- [ ] Improve documentation
- [ ] Add internationalization

### Low Priority
- [ ] Dark mode theme
- [ ] Advanced filtering options
- [ ] Real-time data streaming
- [ ] API endpoints
- [ ] Docker deployment

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Plotly Documentation](https://plotly.com/python/)
- [NASA Turbofan Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

## 🤝 Community

- Be respectful and inclusive
- Help others learn and grow
- Share knowledge and best practices
- Provide constructive feedback

## 📞 Questions?

If you have questions about contributing, please:
1. Check existing issues and discussions
2. Open a new issue with the "question" label
3. Be specific about what you need help with

Thank you for contributing to the Turbofan Engine RUL Prediction System! 🚀 