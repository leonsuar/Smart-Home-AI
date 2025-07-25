# ml_server.py (ubicado en ~/Smart-Home-AI/ml_server/ml_server.py)

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

model = None

@app.before_request
def log_request_info():
    logging.info(f"Petición entrante: {request.method} {request.url}")

@app.after_request
def log_response_info(response):
    logging.info(f"Petición saliente: {request.method} {request.url} - Status: {response.status_code}")
    return response

def load_model():
    global model
    if model is None:
        logging.info("Cargando modelo SentenceTransformer: paraphrase-multilingual-MiniLM-L12-v2...")
        try:
            # Asegúrate de que el modelo se descarga en un directorio persistente si es necesario
            # Para Docker, se descargará en el contenedor si no está en caché
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logging.info("Modelo SentenceTransformer cargado exitosamente.")
            logging.info(f"Usando dispositivo: {model.device}")
        except Exception as e:
            logging.error(f"Error al cargar el modelo SentenceTransformer: {e}")
            model = None # Asegurarse de que el modelo es None si falla la carga

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    if model is None:
        return jsonify({"error": "Modelo no cargado. Intenta de nuevo más tarde."}), 503
    
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({"error": "No se proporcionó texto."}), 400

    try:
        logging.info(f"Generando embedding para texto: '{text[:50]}...'")
        embedding = model.encode(text).tolist()
        logging.info(f"Embedding generado para texto: '{text[:50]}...'")
        return jsonify({"embedding": embedding})
    except Exception as e:
        logging.error(f"Error al generar embedding: {e}")
        return jsonify({"error": f"Error al generar embedding: {e}"}), 500

if __name__ == '__main__':
    # Cargar el modelo cuando la aplicación Flask se inicie
    load_model()
    # Asegúrate de que Flask escuche en 0.0.0.0 para ser accesible desde otros contenedores
    app.run(host='0.0.0.0', port=5001, debug=True)
