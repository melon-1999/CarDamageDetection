from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil
import os
from pathlib import Path
import uuid

app = FastAPI()

# Load model once at startup
model = YOLO("trained.pt")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    input_path = f"input_{unique_id}_{file.filename}"
    
    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run YOLO prediction with save=True
    results = model(input_path, save=True)
    
    # Find the output image in runs/detect/predict/
    output_dir = Path("/ultralytics/runs/detect/predict")
    if output_dir.exists():
        output_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        
        if output_files:
            # Get the most recent file
            latest_file = max(output_files, key=os.path.getctime)
            
            # Clean up input
            os.remove(input_path)
            
            return FileResponse(
                path=str(latest_file),
                media_type="image/jpeg",
                filename=f"processed_{file.filename}"
            )
    
    # Clean up if no output found
    if os.path.exists(input_path):
        os.remove(input_path)
    
    return {"error": "No predictions found"}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}