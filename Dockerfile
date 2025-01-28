# Step 1: Use a base image
FROM python:3.8-slim

# Step 2: Set environment variables
ENV PYTHONUNBUFFERED=1

# Step 3: Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsm6 \
    libxext6 \
    && apt-get clean

# Step 4: Create working directory
WORKDIR /app

# Step 5: Copy application code
COPY . /app

# Step 6: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 7: Expose the application port
EXPOSE 8000

# Step 8: Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
