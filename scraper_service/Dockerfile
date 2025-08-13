# /scraper_service/Dockerfile (Final Corrected Version)

# 1. Start from a standard, lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. --- THE FINAL FIX: Modern and Correct Chrome Installation ---
# This is the new, recommended way to install Chrome on Debian-based systems.
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    --no-install-recommends \
    && curl -sS -o - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome-archive-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y \
       google-chrome-stable \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application's code
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8001

# 7. The command to run the FastAPI/Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]