from fastapi import FastAPI, UploadFile, File, Form
from smart_video import SmartVideoSearch
from pathlib import Path
import shutil

app = FastAPI(title="Smart Video Search API")

# Load once
model = SmartVideoSearch(
    qdrant_url="https://99a1ea80-e6d2-4464-8a71-ca580856c2b8.eu-west-2-0.aws.cloud.qdrant.io:6333",
    qdrant_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.sh06y7PCOwJRzOR72sxtAHFNGh9RD6ke0nA_4_1WacU",
    temp_frame_dir="temp_frames",
    yolo_model_path="yolov8x.pt"
)

@app.get("/device-status")
def device_status():
    return model.get_device_status()

@app.post("/generate-embeddings")
async def generate_embeddings(video: UploadFile = File(...)):
    video_path = Path("temp_frames") / video.filename
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    model.ensure_embeddings(str(video_path))
    return {"message": f"Embeddings generated for {video.filename}"}

@app.post("/search")
async def search(text_query: str = Form(...), top_k: int = Form(5)):
    results = model.search(text_query, top_k=top_k)
    return {"results": results}
