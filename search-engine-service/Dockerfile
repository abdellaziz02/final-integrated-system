# 1. Start from a standard, lightweight Python image
FROM python:3.11-slim

# 2. Install system dependencies needed for Selenium/Chrome to run headlessly
# This is a critical step that was missing before.
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    --no-install-recommends

# 3. Set the working directory
WORKDIR /app

# 4. Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all the project files (app.py, .joblib models, etc.)
COPY . .

# 6. Expose the port the Flask app will run on
EXPOSE 5000

# 7. The command to run the Flask application using the Gunicorn production server
# We give it a long timeout because scraping can be slow.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "8", "--timeout", "300", "app:app"]