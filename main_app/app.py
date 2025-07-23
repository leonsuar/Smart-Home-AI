from flask import Flask, request, jsonify, render_template
import asyncio
import os
import time # Importar time para time.strftime

# Importar los módulos refactorizados
from core_logic.neuron_network import RedNeuronal
from core_logic.utils import normalize_text
from core_logic.mqtt_client import MQTTClient # ¡NUEVO!
from core_logic.home_assistant_api import HomeAssistantAPI # ¡NUEVO!

app = Flask(__name__)

# Configuración MQTT (¡Asegúrate de que estos valores coincidan con tu configuración de Home Assistant!)
MQTT_BROKER_ADDRESS = os.getenv("MQTT_BROKER_ADDRESS", "localhost") # O la IP de tu HA
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "homeassistant_mqtt_user") # Tu usuario MQTT de HA
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "homeassistant_mqtt_password") # Tu contraseña MQTT de HA

# Variables globales para MQTT y la red neuronal
mqtt_client_global = None
home_assistant_api_global = None
red_neuronal_global = None

# Almacenamiento temporal para confirmaciones de guardado pendientes
pending_saves = {}

@app.before_serving
async def initialize_system():
    """
    Inicializa el cliente MQTT, la API de Home Assistant y la red neuronal
    automáticamente al iniciar la aplicación.
    """
    global mqtt_client_global, home_assistant_api_global, red_neuronal_global
    
    print("INFO: Initializing MQTT client...")
    mqtt_client_global = MQTTClient(
        broker_address=MQTT_BROKER_ADDRESS,
        port=MQTT_BROKER_PORT,
        username=MQTT_USERNAME,
        password=MQTT_PASSWORD
    )
    mqtt_client_global.connect()

    print("INFO: Initializing Home Assistant API...")
    home_assistant_api_global = HomeAssistantAPI(mqtt_client_global)

    # Opcional: Suscribirse a tópicos de Home Assistant para recibir eventos
    # Por ejemplo, para monitorear cambios de estado de sensores de movimiento
    # mqtt_client_global.set_message_callback(handle_ha_event)
    # home_assistant_api_global.subscribe_to_all_ha_states() # Suscribirse a todos los estados (puede ser ruidoso)
    # home_assistant_api_global.subscribe_to_state_changes("motion_sensor_garden", domain="binary_sensor")

    print("INFO: Initializing Neuron Network...")
    red_neuronal_global = RedNeuronal(home_assistant_api=home_assistant_api_global) # ¡Pasa la instancia de HA API!
    await red_neuronal_global.initialize_network_automatically()
    print("INFO: System initialization complete.")


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
    session_id = request.remote_addr # Usar la IP remota como un ID de sesión simple

    if red_neuronal_global is None:
        return jsonify(log=[{"tiempo": time.strftime("%H:%M:%S"), "mensaje": "El sistema aún se está inicializando. Por favor, espera un momento.", "tipo": "warning"}], estado_red=[]), 503

    red_neuronal_global.log_mensaje(f"Tú: {comando}", tipo="comando")

    response_text = ""
    should_offer_to_save = False # Bandera para indicar si se debe ofrecer guardar la respuesta

    # La lógica de manejo de comandos internos y de domótica ahora está en neuron_network.py
    # get_local_or_llm_response ahora devuelve la respuesta y una bandera
    llm_response_text, is_new_knowledge_candidate = await red_neuronal_global.get_local_or_llm_response(comando)
    response_text = llm_response_text
    should_offer_to_save = is_new_knowledge_candidate

    red_neuronal_global.log_mensaje(f"IA: {response_text}", tipo="info")

    if should_offer_to_save:
        # Almacena el prompt y la respuesta para una posible confirmación de guardado
        pending_saves[session_id] = {
            "prompt": comando,
            "response": response_text
        }

    return jsonify({
        'log': red_neuronal_global.mensajes,
        'estado_red': red_neuronal_global.obtener_estado_red(),
        'response_text': response_text, # La respuesta final para mostrar en el chat
        'should_offer_to_save': should_offer_to_save # Indicador para el frontend
    })

@app.route('/confirm_save', methods=['POST'])
async def confirm_save():
    """
    Endpoint para que el usuario confirme si desea guardar una respuesta aprendida.
    """
    data = request.json
    session_id = request.remote_addr
    confirm_choice = data.get('choice') # 'yes' o 'no'

    if session_id not in pending_saves:
        red_neuronal_global.log_mensaje("Error: No hay una respuesta pendiente para guardar.", tipo="error")
        return jsonify({'status': 'error', 'message': 'No hay una respuesta pendiente para guardar.'})

    pending_data = pending_saves.pop(session_id) # Elimina la entrada pendiente
    prompt_to_save = pending_data['prompt']
    response_to_save = pending_data['response']

    if confirm_choice == 'yes':
        # Generar el embedding justo antes de guardar (para asegurar que sea el más reciente)
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
# Esta función sería llamada por el mqtt_client_global cuando reciba un mensaje
def handle_ha_event(topic, payload):
    # Aquí puedes procesar los eventos de Home Assistant y pasarlos a la red neuronal
    # Por ejemplo:
    if "homeassistant/binary_sensor/motion_garden/state" in topic and payload == "ON":
        red_neuronal_global.log_mensaje("¡Alerta! Movimiento detectado en el jardín.", tipo="warning")
        # Aquí podrías hacer que la red neuronal responda o tome una acción
        # red_neuronal_global.process_security_event("motion_garden_detected")
    elif "homeassistant/sensor/temperature_living_room/state" in topic:
        red_neuronal_global.log_mensaje(f"Temperatura en sala: {payload}°C", tipo="info")

if __name__ == '__main__':
    # Asegúrate de que el directorio para los archivos de conocimiento exista
    if not os.path.exists('./knowledge'): # Cambiado a 'knowledge' según la estructura
        os.makedirs('./knowledge')
    
    # Flask 2.x+ con funciones async requiere un servidor ASGI como Gunicorn con un worker de Uvicorn.
    # Para desarrollo, Flask puede ejecutarlo directamente, pero es posible que veas advertencias
    # sobre el uso de `asyncio.run` o `app.before_serving` con el servidor de desarrollo predeterminado.
    # Para producción, se recomienda:
    # gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5000 app:app
    app.run(host='0.0.0.0', port=5000, debug=True)

