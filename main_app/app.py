from flask import Flask, request, jsonify, render_template
import asyncio
import os
import time
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from core_logic.neuron_network import RedNeuronal
from core_logic.utils import normalize_text
from core_logic.mqtt_client import MQTTClient
from core_logic.home_assistant_api import HomeAssistantAPI
from core_logic.llm_service import HA_COMMAND_SCHEMA

app = Flask(__name__)

CONFIG_FILE_PATH = './knowledge/config.json'

current_config = {
    "mqtt_broker_address": "localhost",
    "mqtt_broker_port": 1883,
    "mqtt_username": "homeassistant_mqtt_user",
    "mqtt_password": "homeassistant_mqtt_password",
    "ml_server_ip": "127.0.0.1",
    "gemini_api_key": ""
}

mqtt_client_global = None
home_assistant_api_global = None
red_neuronal_global = None

pending_saves = {}

def load_configuration():
    global current_config
    
    if not os.path.exists('./knowledge'):
        os.makedirs('./knowledge')
        logging.info("Directorio './knowledge' creado para configuración.")

    config_from_file = {}
    if os.path.exists(CONFIG_FILE_PATH) and os.path.getsize(CONFIG_FILE_PATH) > 0:
        try:
            with open(CONFIG_FILE_PATH, 'r') as f:
                config_from_file = json.load(f)
                logging.info(f"Configuración cargada desde '{CONFIG_FILE_PATH}'.")
        except json.JSONDecodeError as e:
            logging.error(f"Error al decodificar JSON de configuración '{CONFIG_FILE_PATH}': {e}. Se ignorará el archivo corrupto.")
            os.rename(CONFIG_FILE_PATH, CONFIG_FILE_PATH + ".bak")
    else:
        logging.info(f"No se encontró '{CONFIG_FILE_PATH}' o está vacío. Se usarán valores por defecto y variables de entorno.")

    current_config.update(config_from_file)

    current_config["mqtt_broker_address"] = os.getenv("MQTT_BROKER_ADDRESS", current_config.get("mqtt_broker_address", "localhost"))
    current_config["mqtt_broker_port"] = int(os.getenv("MQTT_BROKER_PORT", current_config.get("mqtt_broker_port", 1883)))
    current_config["mqtt_username"] = os.getenv("MQTT_USERNAME", current_config.get("mqtt_username", "homeassistant_mqtt_user"))
    current_config["mqtt_password"] = os.getenv("MQTT_PASSWORD", current_config.get("mqtt_password", "homeassistant_mqtt_password"))
    current_config["ml_server_ip"] = os.getenv("ML_SERVER_INTERNAL_IP", current_config.get("ml_server_ip", "127.0.0.1"))
    current_config["gemini_api_key"] = os.getenv("GEMINI_API_KEY", current_config.get("gemini_api_key", ""))

    logging.info(f"Configuración final: {current_config}")


def save_configuration():
    """Guarda la configuración actual en el archivo JSON persistente."""
    if not os.path.exists('./knowledge'):
        os.makedirs('./knowledge')
    config_to_save = {k: v for k, v in current_config.items() if k != "gemini_api_key"}
    with open(CONFIG_FILE_PATH, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    logging.info(f"Configuración guardada en '{CONFIG_FILE_PATH}'.")


async def initialize_system_async():
    global mqtt_client_global, home_assistant_api_global, red_neuronal_global
    
    load_configuration()

    logging.info("Esperando 10 segundos para asegurar que ML Server se inicie completamente...")
    await asyncio.sleep(10)

    logging.info("Initializing MQTT client...")
    mqtt_client_global = MQTTClient(
        broker_address=current_config["mqtt_broker_address"],
        port=current_config["mqtt_broker_port"],
        username=current_config["mqtt_username"],
        password=current_config["mqtt_password"]
    )
    mqtt_client_global.connect()

    logging.info("Initializing Home Assistant API...")
    home_assistant_api_global = HomeAssistantAPI(mqtt_client_global)

    logging.info("Initializing Neuron Network...")
    red_neuronal_global = RedNeuronal(
        home_assistant_api=home_assistant_api_global,
        ml_server_ip=current_config["ml_server_ip"],
        gemini_api_key=current_config["gemini_api_key"]
    )
    await red_neuronal_global.initialize_network_automatically()
    logging.info("System initialization complete.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/config')
def config():
    return render_template('config.html', config=current_config)

@app.route('/guardar_configuracion', methods=['POST'])
def guardar_configuracion():
    global current_config
    data = request.json

    try:
        current_config["mqtt_broker_address"] = data.get("mqtt_broker_address", current_config["mqtt_broker_address"])
        current_config["mqtt_broker_port"] = int(data.get("mqtt_broker_port", current_config["mqtt_broker_port"]))
        current_config["mqtt_username"] = data.get("mqtt_username", current_config["mqtt_username"])
        current_config["mqtt_password"] = data.get("mqtt_password", current_config["mqtt_password"])
        current_config["ml_server_ip"] = data.get("ml_server_ip", current_config["ml_server_ip"])
        
        save_configuration()

        logging.info("Configuración actualizada y guardada exitosamente.")
        return jsonify({"status": "success", "message": "Configuración guardada exitosamente. Reinicia el contenedor para aplicar los cambios."}), 200
    except ValueError as e:
        logging.error(f"Error de validación al guardar configuración: {e}")
        return jsonify({"status": "error", "message": f"Error de validación: {e}"}), 400
    except Exception as e:
        logging.error(f"Error al guardar configuración: {e}")
        return jsonify({"status": "error", "message": f"Error interno al guardar la configuración: {e}"}), 500


@app.route('/obtener_log')
async def obtener_log():
    if red_neuronal_global is None:
        return jsonify(log=[{"tiempo": time.strftime("%H:%M:%S"), "mensaje": "El sistema aún se está inicializando. Por favor, espera un momento.", "tipo": "warning"}], estado_red=[]), 503
    return jsonify({
        'log': red_neuronal_global.mensajes,
        'estado_red': red_neuronal_global.obtener_estado_red()
    })

@app.route('/enviar_comando', methods=['POST'])
async def enviar_comando():
    data = request.json
    comando = data.get('comando', '').strip()
    session_id = request.remote_addr

    if red_neuronal_global is None:
        return jsonify(log=[{"tiempo": time.strftime("%H:%M:%S"), "mensaje": "El sistema aún se está inicializando. Por favor, espera un momento.", "tipo": "warning"}], estado_red=[]), 503

    red_neuronal_global.log_mensaje(f"Tú: {comando}", tipo="comando")

    response_text = ""
    should_offer_to_save = False
    ha_command_to_execute = None

    llm_response_text, is_new_knowledge_candidate, ha_command = await red_neuronal_global.get_local_or_llm_response(comando)
    
    response_text = llm_response_text
    should_offer_to_save = is_new_knowledge_candidate
    ha_command_to_execute = ha_command

    if ha_command_to_execute:
        domain = ha_command_to_execute.get("domain")
        service = ha_command_to_execute.get("service")
        entity_id = ha_command_to_execute.get("entity_id")
        
        # Parsear el payload si existe y es una cadena JSON
        payload_str = ha_command_to_execute.get("payload", "{}") # Asegurarse de que sea una cadena, por defecto un objeto vacío
        parsed_payload = {}
        if isinstance(payload_str, str): # Solo intentar parsear si es una cadena
            try:
                parsed_payload = json.loads(payload_str)
            except json.JSONDecodeError as e:
                red_neuronal_global.log_mensaje(f"HA Error: No se pudo parsear el payload JSON: {e}", tipo="error")
                parsed_payload = {} # Usar un payload vacío si hay error de parseo
        elif isinstance(payload_str, dict): # Si por alguna razón ya es un dict (ej. de memoria local)
            parsed_payload = payload_str


        if domain and service and entity_id:
            success, ha_message = home_assistant_api_global.send_ha_command(domain, service, entity_id, parsed_payload) # Usar parsed_payload
            if success:
                red_neuronal_global.log_mensaje(f"HA: {ha_message}", tipo="info")
            else:
                red_neuronal_global.log_mensaje(f"HA Error: {ha_message}", tipo="error")
        else:
            red_neuronal_global.log_mensaje("HA Error: Comando HA incompleto recibido de la IA.", tipo="error")


    red_neuronal_global.log_mensaje(f"IA: {response_text}", tipo="ia")

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

def handle_ha_event(topic, payload):
    if "homeassistant/binary_sensor/motion_garden/state" in topic and payload == "ON":
        red_neuronal_global.log_mensaje("¡Alerta! Movimiento detectado en el jardín.", tipo="warning")
    elif "homeassistant/sensor/temperature_living_room/state" in topic:
        red_neuronal_global.log_mensaje(f"Temperatura en sala: {payload}°C", tipo="info")

if __name__ == '__main__':
    load_configuration()
    
    if not os.path.exists('./knowledge'):
        os.makedirs('./knowledge')
    
    asyncio.run(initialize_system_async())

    app.run(host='0.0.0.0', port=5000, debug=True)
