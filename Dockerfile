# Use a lightweight Python image
FROM python:3.11-slim

# Install Docker Compose dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install docker-compose plugin (Docker-in-Docker approach)
RUN curl -SL https://github.com/docker/compose/releases/download/v2.27.1/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose \
    && chmod +x /usr/local/bin/docker-compose

# Set working directory
WORKDIR /app

# Copy entire project
COPY . .


# Expose api_gateway port
EXPOSE 8080

# Start all services using docker-compose
CMD ["docker-compose", "up", "--build"]
