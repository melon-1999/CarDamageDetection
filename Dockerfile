FROM ultralytics/ultralytics:latest

WORKDIR /app

# Install FastAPI and web dependencies
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy your files
COPY trained.pt /app/
COPY app.py /app/

# Create output directory
RUN mkdir -p runs/detect/predict

# Set environment for headless operation
ENV OPENCV_IO_ENABLE_OPENEXR=1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]