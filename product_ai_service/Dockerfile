# Start with a standard Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first for better caching
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# --- CRITICAL: Copy all necessary files into the container ---
# This copies your main script
COPY main.py .
# This copies your trained model
COPY product_model.joblib .
# This copies your model's features file
COPY model_features.json .

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]