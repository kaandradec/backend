from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from i3d_model import predict_fn
from utils import procesar_video
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pon tu frontend local aquí si quieres restringirlo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Cargar etiquetas
with open("label_map.txt") as f:
    labels = [line.strip() for line in f]

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        tensor = procesar_video(video_path)
        output = predict_fn(tf.constant(tensor, dtype=tf.float32))
        logits = output["default"][0]
        probs = tf.nn.softmax(logits).numpy()

        top_k = 5
        indices = probs.argsort()[-top_k:][::-1]
        predictions = [{"label": labels[i], "prob": float(probs[i])} for i in indices]
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/")
async def root():
    return {"message": "API de predicción de acciones con I3D. Usa /docs para ver la documentación."}
