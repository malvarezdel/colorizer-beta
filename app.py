import os
import uuid
import time
import threading
import requests
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 # 5 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# URLs de los modelos (Zhang et al. - Rama Caffe)
MODEL_URLS = {
      "proto": "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
      "model": "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel",
      "pts": "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy"
}

def download_models():
      for name, url in MODEL_URLS.items():
                ext = url.split('.')[-1]
                filename = f"colorization_{name}.{ext}" if name != "pts" else "pts_in_hull.npy"
                path = os.path.join(app.config['MODEL_FOLDER'], filename)

          if os.path.exists(path):
                        # Verificar si el archivo es valido (si es < 1KB probablemente es un error de login/404)
                        if os.path.getsize(path) < 1024:
                                          os.remove(path)
                                  if not os.path.exists(path):
                                                print(f"Descargando {filename}...")
                                                headers = {"User-Agent": "Mozilla/5.0"}
                                                r = requests.get(url, stream=True, headers=headers)
                                                with open(path, 'wb') as f:
                                                                  for chunk in r.iter_content(chunk_size=8192):
                                                                                        f.write(chunk)
                                                                                print(f"{filename} descargado.")

                                    # Descargar modelos al iniciar
                                    download_models()

# Cargar el modelo en memoria global (OpenCV DNN es eficiente)
print("Cargando modelo en memoria...")
PROTO_PATH = os.path.join(app.config['MODEL_FOLDER'], "colorization_proto.prototxt")
MODEL_PATH = os.path.join(app.config['MODEL_FOLDER'], "colorization_model.caffemodel")
PTS_PATH = os.path.join(app.config['MODEL_FOLDER'], "pts_in_hull.npy")

net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
pts = np.load(PTS_PATH)

# Anadir centros de cluster como convoluciones 1x1
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def process_image(input_path, output_path):
      image = cv2.imread(input_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
                                        colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    cv2.imwrite(output_path, colorized)

@app.route('/')
def index():
      return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
      if 'file' not in request.files:
                return jsonify({'error': 'No se envio ningun archivo'}), 400
    file = request.files['file']
    if file.filename == '':
              return jsonify({'error': 'No se selecciono ningun archivo'}), 400
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], "colorized_" + filename)
    try:
              file.save(filepath)
        # Procesar con OpenCV
        process_image(filepath, result_path)
        return send_file(
                      result_path,
                      mimetype='image/jpeg',
                      as_attachment=True,
                      download_name="colorized_" + file.filename.rsplit('.', 1)[0] + '.jpg'
        )
except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Error al procesar la imagen', 'details': str(e)}), 500
finally:
        def cleanup():
                      time.sleep(60) # Dar tiempo para la descarga
            for p in [filepath, result_path]:
                              try:
                                                    if p and os.path.exists(p):
                                                                              os.remove(p)
                                                                      except:
                                                    pass
                      threading.Thread(target=cleanup, daemon=True).start()

@app.route('/health')
def health():
      return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5000, debug=False)
