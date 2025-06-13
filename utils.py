import cv2
import numpy as np

def procesar_video(path, num_frames=64, size=224):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size))
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise ValueError("No se pudieron extraer frames del video")

    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]

    video_tensor = np.array(frames, dtype=np.float32) / 255.0
    return np.expand_dims(video_tensor, axis=0)  # [1, T, H, W, 3]
