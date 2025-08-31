from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil
import os
from pathlib import Path
import uuid

app = FastAPI()

# Load model once at startup
model = YOLO("trained.pt")  # Dein trainiertes Modell

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Unique filename to avoid conflicts
    unique_id = str(uuid.uuid4())
    input_path = f"input_{unique_id}_{file.filename}"
    
    # Save uploaded image
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run YOLO prediction with save=True
    results = model(input_path, save=True)
    
    # YOLO saves results in runs/detect/predict/
    # Find the output image
    output_dir = Path("runs/detect/predict")
    output_files = list(output_dir.glob("*"))
    
    if output_files:
        # Get the latest created file (your processed image)
        latest_file = max(output_files, key=os.path.getctime)
        
        # Clean up input
        os.remove(input_path)
        
        # Return the processed image
        return FileResponse(
            path=latest_file,
            media_type="image/jpeg",
            filename=f"processed_{file.filename}"
        )
    
    # Cleanup if no output
    os.remove(input_path)
    return {"error": "No predictions found"}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}