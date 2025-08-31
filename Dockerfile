FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV (headless)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install ultralytics fastapi uvicorn python-multipart opencv-python-headless

# Copy files
COPY trained.pt /app/
COPY app.py /app/

# Create output directory
RUN mkdir -p runs/detect/predict

# Set OpenCV to headless mode
ENV OPENCV_IO_ENABLE_OPENEXR=1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]