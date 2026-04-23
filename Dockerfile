FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required directories
RUN mkdir -p artifacts/raw artifacts/processed \
             artifacts/models artifacts/opendata logs

# Expose API port
EXPOSE 5001

# Environment variables
ENV PYTHONPATH=/app

# Start Flask API
CMD ["python3", "-m", "api.app"]