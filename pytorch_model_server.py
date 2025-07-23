from flask import Flask, request, jsonify
from transformers import pipeline # Aunque no se usa directamente, se mantiene por si acaso
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import logging
import os
import traceback # Importar traceback para imprimir la pila de errores
import httpx # Para descargar el modelo si no está presente
import asyncio # Para manejar operaciones asíncronas

app = Flask(__name__)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ruta al modelo TinyLlama GGUF dentro del contenedor
# Esto estará dentro del volumen nombrado /app/models
TINYLLAMA_MODEL_PATH = "/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
TINYLLAMA_DOWNLOAD_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Asegurar que el directorio de modelos exista dentro del volumen
MODEL_DIR = os.path.dirname(TINYLLAMA_MODEL_PATH)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    logging.info(f"Creado directorio de modelos: {MODEL_DIR}")

# Cargar el modelo de generación de texto (TinyLlama)
generator = None
async def download_tinyllama_model():
    if not os.path.exists(TINYLLAMA_MODEL_PATH):
        logging.info(f"Modelo no encontrado en {TINYLLAMA_MODEL_PATH}. Descargando de {TINYLLAMA_DOWNLOAD_URL}...")
        try:
            async with httpx.AsyncClient(timeout=None) as client: # Usar un timeout largo para archivos grandes
                response = await client.get(TINYLLAMA_DOWNLOAD_URL, follow_redirects=True)
                response.raise_for_status()
                with open(TINYLLAMA_MODEL_PATH, "wb") as f:
                    f.write(response.content)
            logging.info("Modelo TinyLlama descargado exitosamente.")
        except httpx.RequestError as e:
            logging.error(f"Error al descargar el modelo TinyLlama: {e}")
            logging.error(traceback.format_exc())
            return False
        except Exception as e:
            logging.error(f"Error inesperado durante la descarga del modelo TinyLlama: {e}")
            logging.error(traceback.format_exc())
            return False
    else:
        logging.info(f"Modelo TinyLlama ya existe en {TINYLLAMA_MODEL_PATH}. Saltando descarga.")
    return True

async def load_generator_model():
    global generator
    if await download_tinyllama_model(): # Asegurar que el modelo esté presente antes de cargarlo
        try:
            # Hardcodeamos n_threads a 4 (núcleos lógicos de tu CPU) para evitar posibles problemas de detección
            generator = Llama(
                model_path=TINYLLAMA_MODEL_PATH,
                n_ctx=2048,
                n_gpu_layers=0, # Aseguramos que no se usen capas de GPU
                n_threads=4,    # Hardcodeado a 4 hilos
                verbose=False
            )
            logging.info(f"Modelo TinyLlama cargado exitosamente desde '{TINYLLAMA_MODEL_PATH}' para generación de texto.")
        except Exception as e:
            logging.error(f"Error al cargar el modelo TinyLlama para generación de texto: {e}")
            logging.error(traceback.format_exc()) # Imprime la pila de errores completa
            generator = None
    else:
        logging.error("No se pudo descargar el modelo TinyLlama. El generador no estará disponible.")

# Cargar el modelo de Sentence Transformers para embeddings
embedding_model = None
async def load_embedding_model():
    global embedding_model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("Modelo 'all-MiniLM-L6-v2' (SentenceTransformer) cargado exitosamente para embeddings.")
    except Exception as e:
        logging.error(f"Error al cargar el modelo de embeddings: {e}")
        logging.error(traceback.format_exc()) # Imprime la pila de errores completa
        embedding_model = None

# Función de inicialización asíncrona para ambos modelos
async def initialize_models():
    await load_generator_model()
    await load_embedding_model()

@app.route('/generate_text', methods=['POST'])
def generate_text():
    if generator is None:
        logging.error("Intento de generación de texto sin modelo TinyLlama cargado.")
        return jsonify({"error": "Modelo de generación de texto no disponible. Intente nuevamente en unos momentos."}), 500

    data = request.get_json()
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.95)

    if not prompt:
        return jsonify({"error": "Se requiere un 'prompt' para la generación de texto."}), 400

    logging.info(f"Recibida solicitud de generación de texto para el prompt: '{prompt}'")
    try:
        output = generator.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["\n", "User:", "###"],
            echo=False
        )
        
        generated_text = output['choices'][0]['text']
        
        logging.info(f"Texto generado: '{generated_text}'")
        return jsonify({"generated_text": generated_text}), 200
    except Exception as e:
        logging.error(f"Error durante la generación de texto con TinyLlama: {e}")
        logging.error(traceback.format_exc()) # Imprime la pila de errores completa
        return jsonify({"error": f"Error interno al generar texto: {e}"}), 500

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    if embedding_model is None:
        logging.error("Intento de obtener embedding sin modelo cargado.")
        return jsonify({"error": "Modelo de embeddings no disponible. Intente nuevamente en unos momentos."}), 500

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "Se requiere un 'text' para obtener el embedding."}), 400

    logging.info(f"Recibida solicitud de embedding para el texto: '{text}'")
    try:
        embedding = embedding_model.encode(text).tolist()
        logging.info(f"Embedding generado (primeros 5 valores): {embedding[:5]}...")
        return jsonify({"embedding": embedding}), 200
    except Exception as e:
        logging.error(f"Error durante la generación del embedding: {e}")
        logging.error(traceback.format_exc()) # Imprime la pila de errores completa
        return jsonify({"error": f"Error interno al generar embedding: {e}"}), 500

if __name__ == '__main__':
    # Ejecutar la inicialización de los modelos de forma asíncrona antes de iniciar el servidor Flask
    asyncio.run(initialize_models())
    app.run(host='0.0.0.0', port=5001, debug=True)
