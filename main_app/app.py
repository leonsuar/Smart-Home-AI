from flask import Flask, request, jsonify, render_template
import asyncio
import os
import time
import logging # Importar logging

# Configurar logging básico para la aplicación
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Importar los módulos con rutas absolutas
from core_logic.neuron_network import RedNeuronal
from core_logic.utils import normalize_text
from core_logic.mqtt_client import MQTTClient
from core_logic.home_assistant_api import HomeAssistantAPI

app = Flask(__name__)

# Configuración MQTT (¡Asegúrate de que estos valores coincidan con tu configuración de Home Assistant!)
# Estos valores se obtienen de las variables de entorno definidas en docker-compose.yml
MQTT_BROKER_ADDRESS = os.getenv("MQTT_BROKER_ADDRESS", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "homeassistant_mqtt_user")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "homeassistant_mqtt_password")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Variables globales para MQTT y la red neuronal
mqtt_client_global = None
home_assistant_api_global = None
red_neuronal_global = None

# Almacenamiento temporal para confirmaciones de guardado pendientes
pending_saves = {}

# Función de inicialización asíncrona
async def initialize_system_async(): # Renombrado para claridad, no es un decorador
    """
    Inicializa el cliente MQTT, la API de Home Assistant y la red neuronal.
    """
    global mqtt_client_global, home_assistant_api_global, red_neuronal_global
    
    logging.info("Initializing MQTT client...")
    mqtt_client_global = MQTTClient(
        broker_address=MQTT_BROKER_ADDRESS,
        port=MQTT_BROKER_PORT,
        username=MQTT_USERNAME,
        password=MQTT_PASSWORD
    )
    mqtt_client_global.connect()

    logging.info("Initializing Home Assistant API...")
    home_assistant_api_global = HomeAssistantAPI(mqtt_client_global)

    # Opcional: Suscribirse a tópicos de Home Assistant para recibir eventos
    # mqtt_client_global.set_message_callback(handle_ha_event)
    # home_assistant_api_global.subscribe_to_all_ha_states()
    # home_assistant_api_global.subscribe_to_state_changes("motion_sensor_garden", domain="binary_sensor")

    logging.info("Initializing Neuron Network...")
    red_neuronal_global = RedNeuronal(home_assistant_api=home_assistant_api_global)
    await red_neuronal_global.initialize_network_automatically()
    logging.info("System initialization complete.")


@app.route('/')
def index():
    """
    Sirve el archivo HTML principal de la interfaz de usuario.
    """
    return render_template('index.html')

@app.route('/obtener_log')
async def obtener_log():
    """
    Devuelve los mensajes de log actuales y el estado de la red.
    """
    if red_neuronal_global is None:
        return jsonify(log=[{"tiempo": time.strftime("%H:%M:%S"), "mensaje": "El sistema aún se está inicializando. Por favor, espera un momento.", "tipo": "warning"}], estado_red=[]), 503
    return jsonify({
        'log': red_neuronal_global.mensajes,
        'estado_red': red_neuronal_global.obtener_estado_red()
    })

@app.route('/enviar_comando', methods=['POST'])
async def enviar_comando():
    """
    Procesa los comandos enviados por el usuario.
    """
    data = request.json
    comando = data.get('comando', '').strip()
    session_id = request.remote_addr

    if red_neuronal_global is None:
        return jsonify(log=[{"tiempo": time.strftime("%H:%M:%S"), "mensaje": "El sistema aún se está inicializando. Por favor, espera un momento.", "tipo": "warning"}], estado_red=[]), 503

    red_neuronal_global.log_mensaje(f"Tú: {comando}", tipo="comando")

    response_text = ""
    should_offer_to_save = False

    llm_response_text, is_new_knowledge_candidate = await red_neuronal_global.get_local_or_llm_response(comando)
    response_text = llm_response_text
    should_offer_to_save = is_new_knowledge_candidate

    red_neuronal_global.log_mensaje(f"IA: {response_text}", tipo="info")

    if should_offer_to_save:
        pending_saves[session_id] = {
            "prompt": comando,
            "response": response_text
        }

    return jsonify({
        'log': red_neuronal_global.mensajes,
        'estado_red': red_neuronal_global.obtener_estado_red(),
        'response_text': response_text,
        'should_offer_to_save': should_offer_to_save
    })

@app.route('/confirm_save', methods=['POST'])
async def confirm_save():
    """
    Endpoint para que el usuario confirme si desea guardar una respuesta aprendida.
    """
    data = request.json
    session_id = request.remote_addr
    confirm_choice = data.get('choice')

    if session_id not in pending_saves:
        red_neuronal_global.log_mensaje("Error: No hay una respuesta pendiente para guardar.", tipo="error")
        return jsonify({'status': 'error', 'message': 'No hay una respuesta pendiente para guardar.'})

    pending_data = pending_saves.pop(session_id)
    prompt_to_save = pending_data['prompt']
    response_to_save = pending_data['response']

    if confirm_choice == 'yes':
        embedding = await red_neuronal_global.get_embedding_from_pytorch(prompt_to_save)
        if embedding is not None:
            await red_neuronal_global.train_with_feedback(prompt_to_save, response_to_save, embedding=embedding, save_memory=True)
            red_neuronal_global.log_mensaje(f"Guardado: '{prompt_to_save}' -> '{response_to_save[:50]}...' en memoria.", tipo="info")
            message = "¡Respuesta guardada en la memoria!"
        else:
            red_neuronal_global.log_mensaje(f"Error: No se pudo generar embedding para guardar la respuesta.", tipo="error")
            message = "Error al guardar la respuesta (no se pudo generar el embedding)."
    else:
        red_neuronal_global.log_mensaje(f"Descartado: '{prompt_to_save}' -> '{response_to_save[:50]}...'", tipo="info")
        message = "Respuesta descartada."

    return jsonify({
        'status': 'success',
        'message': message,
        'log': red_neuronal_global.mensajes,
        'estado_red': red_neuronal_global.obtener_estado_red()
    })

# Manejador de eventos MQTT entrantes (ejemplo)
def handle_ha_event(topic, payload):
    if "homeassistant/binary_sensor/motion_garden/state" in topic and payload == "ON":
        red_neuronal_global.log_mensaje("¡Alerta! Movimiento detectado en el jardín.", tipo="warning")
    elif "homeassistant/sensor/temperature_living_room/state" in topic:
        red_neuronal_global.log_mensaje(f"Temperatura en sala: {payload}°C", tipo="info")

if __name__ == '__main__':
    if not os.path.exists('./knowledge'):
        os.makedirs('./knowledge')
    
    # Ejecutar la inicialización asíncrona del sistema antes de iniciar la aplicación Flask
    asyncio.run(initialize_system_async())

    app.run(host='0.0.0.0', port=5000, debug=True)
