import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HomeAssistantAPI:
    def __init__(self, mqtt_client):
        self.mqtt_client = mqtt_client
        self.base_topic = "homeassistant"
        self.ha_entity_info = {} # Almacena la información de las entidades descubiertas por HA
        self.tasmota_command_map = {} # Mapeo de nombres amigables a comandos Tasmota
        logging.info(f"HomeAssistantAPI inicializada con tópico base: {self.base_topic}")

    def process_mqtt_message(self, topic, payload):
        # Lógica de descubrimiento de Home Assistant
        if topic.startswith(f"{self.base_topic}/"):
            try:
                if topic.endswith("/config"):
                    data = json.loads(payload)
                    entity_id = self._get_entity_id_from_ha_config_topic(topic, data)
                    if entity_id:
                        self.ha_entity_info[entity_id] = {
                            "name": data.get("name", entity_id.split('.')[-1]),
                            "domain": entity_id.split('.')[0],
                            "command_topic": data.get("command_topic"), # Tópico de comando para HA Discovery
                            "state_topic": data.get("state_topic"),
                            "payload_on": data.get("payload_on"),
                            "payload_off": data.get("payload_off"),
                            "device": data.get("device", {}),
                            "raw_config": data # Guardar la configuración completa
                        }
                        logging.info(f"Dispositivo Home Assistant descubierto y almacenado: {entity_id} (Nombre: {self.ha_entity_info[entity_id]['name']})")
            except json.JSONDecodeError:
                pass 
            except Exception as e:
                logging.error(f"Error al procesar mensaje MQTT de Home Assistant para tópico {topic}: {e}")

        # Lógica para el descubrimiento nativo de Tasmota (si no usa HA Discovery)
        # Esta sección se encarga de poblar ha_entity_info y tasmota_command_map
        elif topic.startswith("tasmota/discovery/") and topic.endswith("/config"):
            try:
                data = json.loads(payload)
                device_name = data.get("hn") 
                if not device_name:
                    device_name = data.get("dn", topic.split('/')[-2]) 

                functions = data.get("fn", ["Main"]) 
                
                for i, func_name in enumerate(functions):
                    cmnd_topic_base = f"{data.get('ft', '%prefix%/%topic%/').replace('%prefix%', 'cmnd').replace('%topic%', data.get('t', device_name))}"
                    stat_topic_base = f"{data.get('ft', '%prefix%/%topic%/').replace('%prefix%', 'stat').replace('%topic%', data.get('t', device_name))}"
                    tele_topic_base = f"{data.get('ft', '%prefix%/%topic%/').replace('%prefix%', 'tele').replace('%topic%', data.get('t', device_name))}"

                    # Tópicos específicos de POWER para Tasmota
                    tasmota_power_command_topic = f"{cmnd_topic_base}/POWER{i+1}" if len(functions) > 1 else f"{cmnd_topic_base}/POWER"
                    tasmota_state_topic = f"{stat_topic_base}/POWER{i+1}" if len(functions) > 1 else f"{stat_topic_base}/POWER"

                    if func_name: 
                        entity_id = f"light.{device_name.lower().replace('-', '_').replace(' ', '_')}"
                        if len(functions) > 1: 
                            entity_id = f"light.{device_name.lower().replace('-', '_').replace(' ', '_')}_{i+1}"

                        self.ha_entity_info[entity_id] = {
                            "name": func_name, 
                            "domain": "light", # Asumimos 'light' para Tasmota POWER
                            "command_topic": tasmota_power_command_topic, # Este es el tópico cmnd real de Tasmota
                            "state_topic": tasmota_state_topic,
                            "tele_state_topic": f"{tele_topic_base}/STATE", 
                            "raw_config": data 
                        }
                        self.tasmota_command_map[func_name.lower()] = entity_id
                        logging.info(f"Dispositivo Tasmota nativo descubierto y almacenado: {entity_id} (Nombre: {func_name})")
            except json.JSONDecodeError:
                pass 
            except Exception as e:
                logging.error(f"Error al procesar mensaje MQTT de Tasmota para tópico {topic}: {e}")
        
        elif topic.startswith("tele/") and topic.endswith("/STATE"):
            pass 
        elif topic.startswith("stat/") and topic.endswith("/POWER"):
            pass


    def _get_entity_id_from_ha_config_topic(self, topic, config_payload):
        parts = topic.split('/')
        if len(parts) >= 4 and parts[0] == self.base_topic and parts[-1] == "config":
            domain = parts[1]
            node_id = parts[2]
            object_id = parts[3]
            if "object_id" in config_payload:
                return f"{domain}.{config_payload['object_id']}"
            else:
                return f"{domain}.{node_id}_{object_id}"
        return None

    def send_ha_command(self, domain, service, entity_id, payload=None):
        """
        Envía un comando a Home Assistant a través de su tópico de servicios MQTT.
        Esto asume que una instancia de Home Assistant está escuchando este tópico.
        """
        if not self.mqtt_client:
            logging.error("Cliente MQTT no inicializado.")
            return False, "Cliente MQTT no inicializado."

        if payload is None or payload == "":
            json_payload = {}
        elif isinstance(payload, str):
            try:
                json_payload = json.loads(payload)
            except json.JSONDecodeError:
                logging.error(f"Payload no es un JSON válido: {payload}")
                return False, "Payload de comando no es un JSON válido."
        else:
            json_payload = payload

        service_topic = f"{self.base_topic}/services/{domain}/{service}"
        
        ha_command_payload = {
            "entity_id": entity_id,
            **json_payload 
        }

        try:
            self.mqtt_client.publish(service_topic, json.dumps(ha_command_payload))
            logging.info(f"Comando HA de servicio enviado: Tópico='{service_topic}', Payload='{json.dumps(ha_command_payload)}'")
            return True, f"Comando '{service}' enviado a '{entity_id}' a través de Home Assistant MQTT."
        except Exception as e:
            logging.error(f"Error al enviar comando HA de servicio: {e}")
            return False, f"Error al enviar comando HA de servicio: {e}"

    def send_tasmota_command(self, entity_id, state):
        """
        Envía un comando directo a un dispositivo Tasmota.
        Asume que entity_id es un dispositivo Tasmota descubierto.
        'state' debe ser 'ON' o 'OFF'.
        """
        if not self.mqtt_client:
            logging.error("Cliente MQTT no inicializado.")
            return False, "Cliente MQTT no inicializado."

        entity_info = self.ha_entity_info.get(entity_id)
        if not entity_info or not entity_info.get("command_topic"):
            logging.error(f"No se encontró información de comando Tasmota para la entidad: {entity_id}")
            return False, f"No se encontró información de comando Tasmota para la entidad: {entity_id}"
        
        command_topic = entity_info["command_topic"]
        payload = state.upper() # Tasmota espera "ON" o "OFF"

        try:
            self.mqtt_client.publish(command_topic, payload)
            logging.info(f"Comando Tasmota directo enviado: Tópico='{command_topic}', Payload='{payload}'")
            return True, f"Comando '{state}' enviado directamente a '{entity_info['name']}' (Tasmota)."
        except Exception as e:
            logging.error(f"Error al enviar comando Tasmota directo: {e}")
            return False, f"Error al enviar comando Tasmota directo: {e}"


    def get_discovered_entities(self):
        return self.ha_entity_info

    def get_tasmota_command_map(self):
        return self.tasmota_command_map
