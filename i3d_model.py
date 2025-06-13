import tensorflow as tf
import tensorflow_hub as hub

print("Cargando modelo I3D...")
model_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
model = hub.load(model_url)
predict_fn = model.signatures['default']
print("Modelo cargado.")