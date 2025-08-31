from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil
import os
from pathlib import Path
import uuid
import cv2

# Set OpenCV to headless mode
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

app = FastAPI()

# Load model once at startup
model = YOLO("trained.pt")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    input_path = f"input_{unique_id}_{file.filename}"
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run YOLO prediction with save=True
    results = model(input_path, save=True)
    
    # Find output image
    output_dir = Path("runs/detect/predict")
    output_files = list(output_dir.glob("*"))
    
    if output_files:
        latest_file = max(output_files, key=os.path.getctime)
        os.remove(input_path)
        
        return FileResponse(
            path=latest_file,
            media_type="image/jpeg",
            filename=f"processed_{file.filename}"
        )
    
    os.remove(input_path)
    return {"error": "No predictions found"}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}