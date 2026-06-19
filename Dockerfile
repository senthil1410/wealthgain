# Base image — Python 3.11 slim (stable, widely supported)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — reinstall only if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Environment variables (secrets injected at runtime — not baked in)
ENV PYTHONUNBUFFERED=1

# Start command
CMD ["streamlit", "run", "wealthgain_chat.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]