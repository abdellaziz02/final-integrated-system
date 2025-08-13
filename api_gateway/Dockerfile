# Start with an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install the 'curl' command for health checks if needed
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# The command to run when the application starts, formatted as a valid JSON array
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]