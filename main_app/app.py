import asyncio
import os
import uuid
import logging
import time
import requests
import json
from flask import Flask, render_template, request, jsonify

from core_logic.mqtt_client import MQTTClient
from core_logic.home_assistant_api import HomeAssistantAPI
from core_logic.neuron_network import RedNeuronal 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Variables globales para las instancias
mqtt_client_global = None
home_assistant_api_global = None
neuron_network_global = None
config_global = {}

# Lista para almacenar los logs
system_logs = []

# Función para añadir mensajes al log del sistema
def add_log_entry(message, level='info', source='System'):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"tiempo": timestamp, "tipo": level, "fuente": source, "mensaje": message}
    system_logs.append(log_entry)

    python_log_level = logging.INFO 
    if level == 'error':
        python_log_level = logging.ERROR
    elif level == 'warning':
        python_log_level = logging.WARNING

    logging.log(python_log_level, message)

def load_config():
    global config_global
    config_path = './knowledge/config.json'
    
    default_config = {
        "mqtt_broker_address": "localhost",
        "mqtt_broker_port": 1883,
        "mqtt_username": "",
        "mqtt_password": "",
        "ml_server_ip": "ml_server", # <-- ¡CORREGIDO! Valor por defecto para comunicación entre contenedores
        "gemini_api_key": "" 
    }

    file_config = {}
    try:
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        add_log_entry("Configuración cargada desde './knowledge/config.json'.", 'info')
    except FileNotFoundError:
        add_log_entry(f"Error: Archivo de configuración no encontrado en '{config_path}'. Usando valores por defecto.", 'error')
    except json.JSONDecodeError:
        add_log_entry(f"Error: El archivo de configuración '{config_path}' no es un JSON válido. Usando valores por defecto.", 'error')
    
    # Fusionar la configuración por defecto, luego la del archivo
    config_global = {**default_config, **file_config}

    # Sobrescribir con variables de entorno si existen (estas tienen la máxima prioridad)
    config_global['mqtt_broker_address'] = os.environ.get('MQTT_BROKER_ADDRESS', config_global['mqtt_broker_address'])
    config_global['mqtt_broker_port'] = int(os.environ.get('MQTT_BROKER_PORT', config_global['mqtt_broker_port'])) 
    config_global['mqtt_username'] = os.environ.get('MQTT_USERNAME', config_global['mqtt_username'])
    config_global['mqtt_password'] = os.environ.get('MQTT_PASSWORD', config_global['mqtt_password'])
    # Asegurarse de que ML_SERVER_INTERNAL_IP sobrescriba si está presente
    config_global['ml_server_ip'] = os.environ.get('ML_SERVER_INTERNAL_IP', config_global['ml_server_ip'])
    config_global['gemini_api_key'] = os.environ.get('GEMINI_API_KEY', config_global['gemini_api_key'])

    add_log_entry(f"Configuración final: {config_global}", 'info')


async def initialize_system_async():
    global mqtt_client_global, home_assistant_api_global, neuron_network_global

    add_log_entry("Esperando 30 segundos para asegurar que ML Server se inicie completamente...", 'info')
    await asyncio.sleep(30) 

    add_log_entry("Initializing MQTT client...", 'info')
    mqtt_client_global = MQTTClient(
        broker_address=config_global["mqtt_broker_address"],
        broker_port=config_global["mqtt_broker_port"],
        username=config_global["mqtt_username"],
        password=config_global["mqtt_password"],
        client_id=f"smart_home_ai_client-{uuid.uuid4().hex[:8]}",
        message_callback=None 
    )
    mqtt_client_global.connect()
    mqtt_client_global.loop_start()

    add_log_entry("Initializing Home Assistant API...", 'info')
    home_assistant_api_global = HomeAssistantAPI(mqtt_client=mqtt_client_global)
    mqtt_client_global.message_callback = home_assistant_api_global.process_mqtt_message
    
    mqtt_client_global.subscribe_to_all_ha_topics("homeassistant") 

    add_log_entry("Initializing Neuron Network...", 'info')
    neuron_network_global = RedNeuronal(
        ml_server_ip=config_global["ml_server_ip"],
        gemini_api_key=config_global["gemini_api_key"],
        home_assistant_api=home_assistant_api_global
    )
    test_embedding_text = "test..."
    for i in range(1, 6):
        add_log_entry(f"Solicitando embedding para '{test_embedding_text}' al ML Server en http://{config_global['ml_server_ip']}:5001/get_embedding (Intento {i}/5)", 'info')
        try:
            test_embedding = await neuron_network_global.get_embedding(test_embedding_text)
            if test_embedding:
                add_log_entry("Embedding recibido exitosamente del ML Server.", 'info')
                break
        except requests.exceptions.ConnectionError as e:
            add_log_entry(f"Error de conexión con ML Server: {e}", 'error')
            if i == 5:
                add_log_entry("Máximo de reintentos alcanzado para ML Server.", 'error')
                add_log_entry("Fallo al conectar con ML Server. La IA puede no funcionar correctamente.", 'error')
                add_log_entry("No se pudo establecer conexión con ML Server. La IA no funcionará correctamente.", 'error')
            await asyncio.sleep(5) 
        except Exception as e:
            add_log_entry(f"Error inesperado al conectar con ML Server: {e}", 'error')
            if i == 5:
                add_log_entry("Fallo al conectar con ML Server. La IA puede no funcionar correctamente.", 'error')
            await asyncio.sleep(5) 

    add_log_entry("System initialization complete.", 'info')

load_config()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/obtener_log')
def obtener_log():
    try:
        import psutil 
        system_stats = [
            {"tipo": "Sistema: RAM Disponible (MB)", "valor": round(psutil.virtual_memory().available / (1024 * 1024), 2)},
            {"tipo": "Sistema: Núcleos CPU", "valor": psutil.cpu_count()},
            {"tipo": "Sistema: Uso Disco (%)", "valor": psutil.disk_usage('/').percent}
        ]
    except ImportError:
        system_stats = [{"tipo": "Sistema: Estadísticas no disponibles", "valor": "psutil no instalado"}]

    return jsonify({
        "log": system_logs[-100:], 
        "estado_red": system_stats,
        "discovered_entities": home_assistant_api_global.ha_entity_info,
        "tasmota_map": home_assistant_api_global.tasmota_command_map
    })

@app.route('/get_config_data')
def get_config_data():
    return jsonify(config_global)

@app.route('/enviar_comando', methods=['POST'])
async def enviar_comando():
    data = request.json
    comando_usuario = data.get('comando', '').strip()
    
    if not comando_usuario:
        return jsonify({"status": "error", "message": "Comando vacío.", "should_offer_to_save": False})

    add_log_entry(f"Tú: {comando_usuario}", 'comando', 'User') 

    response_from_ia = await neuron_network_global.process_command(comando_usuario)
    add_log_entry(f"IA: {response_from_ia['response_text']}", 'ia', 'AI') 

    # Determinar si se debe ofrecer guardar la interacción
    # Solo ofrecer guardar si la IA generó una respuesta de texto y no un comando HA
    should_offer_to_save = response_from_ia.get("action_type") == "text_response" and \
                           neuron_network_global.last_interaction is not None

    return jsonify({
        "status": "success",
        "response_text": response_from_ia["response_text"],
        "should_offer_to_save": should_offer_to_save
    })

@app.route('/confirm_save', methods=['POST'])
async def confirm_save():
    data = request.json
    choice = data.get('choice')

    if choice == 'yes':
        await neuron_network_global.save_last_interaction()
        add_log_entry("Interacción guardada en la memoria de la IA.", 'info', 'System')
        return jsonify({"status": "success", "message": "Interacción guardada."})
    else:
        neuron_network_global.discard_last_interaction()
        add_log_entry("Interacción descartada.", 'info', 'System')
        return jsonify({"status": "success", "message": "Interacción descartada."})

@app.route('/config')
def config_page():
    # Pasa el objeto de configuración global a la plantilla
    return render_template('config.html', config=config_global) 

@app.route('/guardar_configuracion', methods=['POST'])
def save_configuration():
    global config_global
    new_config = request.json
    config_path = './knowledge/config.json'

    try:
        # Actualizar solo los campos proporcionados, manteniendo los demás
        for key, value in new_config.items():
            config_global[key] = value
        
        with open(config_path, 'w') as f:
            json.dump(config_global, f, indent=4)
        add_log_entry("Configuración guardada exitosamente en './knowledge/config.json'.", 'info', 'Config')
        return jsonify({"status": "success", "message": "Configuración guardada exitosamente."})
    except Exception as e:
        add_log_entry(f"Error al guardar la configuración: {e}", 'error', 'Config')
        return jsonify({"status": "error", "message": f"Error al guardar la configuración: {e}"}), 500


if __name__ == '__main__':
    asyncio.run(initialize_system_async())
    app.run(host='0.0.0.0', port=5000, debug=True)
