from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import logging
import torch # Importar torch para verificar la disponibilidad de CUDA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Cargar el modelo de Sentence-Transformers
# Usamos un modelo multilingüe que es bueno para embeddings
# Se descarga la primera vez, lo que puede tardar.
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
model = None # Inicializar a None

def load_model():
    """Carga el modelo de SentenceTransformer de forma segura."""
    global model
    try:
        logging.info(f"Cargando modelo SentenceTransformer: {model_name}...")
        # Intentar usar CUDA si está disponible, de lo contrario usar CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Usando dispositivo: {device}")
        model = SentenceTransformer(model_name, device=device)
        logging.info("Modelo SentenceTransformer cargado exitosamente.")
    except Exception as e:
        logging.error(f"Error al cargar el modelo SentenceTransformer: {e}")
        model = None # Asegurarse de que el modelo sea None si falla la carga

# Cargar el modelo al iniciar la aplicación Flask
# Esto asegura que el modelo se cargue una vez cuando la aplicación se inicializa.
with app.app_context():
    load_model()

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    """
    Endpoint para obtener el embedding de un texto.
    Espera un JSON con la clave 'text'.
    """
    if model is None:
        logging.error("Solicitud de embedding rechazada: Modelo no cargado.")
        return jsonify({"error": "Modelo de embedding no cargado"}), 500

    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({"error": "Falta el campo 'text' en la solicitud"}), 400

    try:
        # Generar el embedding
        embedding = model.encode(text).tolist() # Convertir a lista para JSON
        logging.info(f"Embedding generado para texto: '{text[:50]}...'")
        return jsonify({"embedding": embedding})
    except Exception as e:
        logging.error(f"Error al generar embedding para '{text[:50]}...': {e}")
        return jsonify({"error": f"Error al generar embedding: {e}"}), 500

if __name__ == '__main__':
    # Iniciar el servidor Flask
    # host='0.0.0.0' permite que sea accesible desde fuera del contenedor
    # port=5001 es el puerto que expone el servicio ML
    # debug=True es útil para desarrollo, pero desactívalo en producción
    app.run(host='0.0.0.0', port=5001, debug=True)
