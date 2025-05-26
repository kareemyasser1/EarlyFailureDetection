# Use Python 3.11 slim image for TensorFlow compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements_demo.txt .

# Install Python dependencies
# Try full requirements first, fallback to demo if TensorFlow fails
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir -r requirements_demo.txt

# Copy application files
COPY . .

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Default command (can be overridden)
CMD ["streamlit", "run", "app_demo.py", "--server.address", "0.0.0.0", "--server.port", "8501"] 