FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install ultralytics fastapi uvicorn python-multipart

# Copy your trained model
COPY trained.pt /app/

# Copy API code
COPY app.py /app/

# Create directories for YOLO outputs
RUN mkdir -p runs/detect/predict

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]