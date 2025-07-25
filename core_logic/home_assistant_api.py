import json
import logging
from core_logic.mqtt_client import MQTTClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HomeAssistantAPI:
    def __init__(self, mqtt_client: MQTTClient, base_topic="homeassistant"):
        self.mqtt_client = mqtt_client
        self.base_topic = base_topic
        logging.info(f"HomeAssistantAPI inicializada con tópico base: {self.base_topic}")

        # Diccionario para almacenar información de entidades HA descubiertas automáticamente
        # Clave: entity_id (ej. "light.luz_azotea_1")
        # Valor: Diccionario con detalles como 'command_topic', 'state_topic', 'domain', etc.
        self.ha_entity_info = {}

        # Mapeo de entity_id de HA a información de comando de Tasmota.
        # Esto es para dispositivos Tasmota que no usan el descubrimiento estándar de HA para comandos
        # o cuando necesitamos un mapeo más específico (ej. Power1 vs POWER).
        # Se puede poblar dinámicamente o mantener para overrides manuales.
        self.tasmota_command_map = {
            # Ejemplo manual (se puede eliminar si todo es por descubrimiento)
            "light.tasmotafondo_tasmota": {
                "command_topic_prefix": "cmnd/tasmota_80DDC9", # Asegúrate que este sea el Topic de Tasmota
                "power_topic_suffix": "Power1" # O "POWER" si solo tiene un relé
            }
        }

    def process_mqtt_message(self, topic: str, payload_str: str):
        """
        Procesa los mensajes MQTT entrantes, especialmente los de descubrimiento de Home Assistant.
        """
        # Mensajes de descubrimiento de Home Assistant
        if topic.startswith(f"{self.base_topic}/") and topic.endswith("/config"):
            try:
                config = json.loads(payload_str)
                domain = topic.split('/')[1] # light, switch, sensor, etc.
                object_id = topic.split('/')[2] # ID del objeto en el tópico de descubrimiento

                # Construir el entity_id estándar de Home Assistant
                entity_id = f"{domain}.{object_id}"
                
                # Extraer información relevante del payload de configuración
                command_topic = config.get("command_topic")
                state_topic = config.get("state_topic")
                name = config.get("name", object_id) # Usar el nombre si está disponible, sino el object_id
                
                self.ha_entity_info[entity_id] = {
                    "domain": domain,
                    "name": name,
                    "command_topic": command_topic,
                    "state_topic": state_topic,
                    "config_payload": config # Guardar el payload completo para referencia
                }
                logging.info(f"Dispositivo HA descubierto: {entity_id} (Nombre: {name}, Comando: {command_topic}, Estado: {state_topic})")

                # Lógica específica para Tasmota basada en el descubrimiento
                # Tasmota a menudo incluye "device" con "identifiers" que contienen el prefijo MQTT
                if "device" in config and "identifiers" in config["device"]:
                    for identifier in config["device"]["identifiers"]:
                        # Los identificadores de Tasmota suelen ser algo como "tasmota_XXXXXX"
                        if identifier.startswith("tasmota_"):
                            tasmota_topic_base = identifier
                            # Tasmota discovery a veces incluye cmd_t y stat_t directamente
                            tasmota_cmd_topic = config.get("cmd_t")
                            tasmota_stat_topic = config.get("stat_t")

                            # Si no se proporciona cmd_t/stat_t directamente, podemos inferirlos
                            if not tasmota_cmd_topic:
                                # Asumimos el formato cmnd/<TOPIC>/POWER o cmnd/<TOPIC>/PowerX
                                # Esto requiere que el dispositivo Tasmota tenga su Topic configurado.
                                # Por simplicidad, usaremos el topic base de Tasmota como la base del comando.
                                # Home Assistant normalmente mapea esto, pero si necesitamos control directo:
                                tasmota_cmd_topic = f"cmnd/{tasmota_topic_base}/POWER" # Default POWER
                                # Si tiene múltiples relés, podría ser Power1, Power2, etc.
                                # Para una detección más robusta, necesitaríamos más lógica o configuración manual.

                            if tasmota_cmd_topic:
                                # Extraer el sufijo de Power (ej. POWER, Power1)
                                power_suffix = "POWER"
                                if "/POWER" in tasmota_cmd_topic:
                                    power_suffix = "POWER"
                                elif "/Power1" in tasmota_cmd_topic:
                                    power_suffix = "Power1"
                                # ... y así para otros PowerX si existen

                                self.tasmota_command_map[entity_id] = {
                                    "command_topic_prefix": f"cmnd/{tasmota_topic_base}",
                                    "power_topic_suffix": power_suffix # Esto es una inferencia, puede necesitar ajuste
                                }
                                logging.info(f"Dispositivo Tasmota detectado y mapeado: {entity_id} -> {self.tasmota_command_map[entity_id]}")
                            break # Salir después de encontrar un identificador Tasmota
            except json.JSONDecodeError as e:
                logging.error(f"Error al decodificar el mensaje de descubrimiento MQTT: {e} - Tópico: {topic}, Payload: {payload_str}")
            except Exception as e:
                logging.error(f"Error inesperado al procesar mensaje de descubrimiento: {e} - Tópico: {topic}")
        
        # Puedes añadir lógica aquí para procesar mensajes de estado si quieres que la IA los use
        # (ej. homeassistant/light/mylight/state)
        # if topic.startswith(f"{self.base_topic}/") and topic.endswith("/state"):
        #     # Actualizar un estado interno de la IA si es necesario
        #     pass


    def send_ha_command(self, domain: str, service: str, entity_id: str, payload: dict = None):
        """
        Envía un comando de servicio a Home Assistant a través de MQTT.
        Prioriza el mapeo de Tasmota, luego la información de descubrimiento de HA,
        y finalmente el tópico de servicio genérico de HA.
        """
        
        # 1. Intentar con el mapeo específico de Tasmota (manual o descubierto)
        if entity_id in self.tasmota_command_map and domain == "light": # Asumiendo que Tasmota es para luces/switches
            tasmota_info = self.tasmota_command_map[entity_id]
            tasmota_topic = f"{tasmota_info['command_topic_prefix']}/{tasmota_info['power_topic_suffix']}"
            
            tasmota_payload = ""
            if service == "turn_on":
                tasmota_payload = "ON"
            elif service == "turn_off":
                tasmota_payload = "OFF"
            elif service == "toggle":
                tasmota_payload = "TOGGLE"
            else:
                logging.warning(f"Servicio Tasmota '{service}' no mapeado para {entity_id}. Enviando payload vacío.")
                tasmota_payload = ""

            try:
                self.mqtt_client.publish(tasmota_topic, tasmota_payload)
                logging.info(f"Comando Tasmota publicado: Tópico='{tasmota_topic}', Payload='{tasmota_payload}'")
                return True, f"Comando Tasmota '{service}' para '{entity_id}' enviado."
            except Exception as e:
                logging.error(f"Error al publicar comando Tasmota '{domain}.{service}' para '{entity_id}': {e}")
                return False, f"Error al enviar comando Tasmota: {e}"
        
        # 2. Si no es Tasmota o el mapeo de Tasmota no aplica, intentar con la información de descubrimiento de HA
        if entity_id in self.ha_entity_info and self.ha_entity_info[entity_id].get("command_topic"):
            command_topic = self.ha_entity_info[entity_id]["command_topic"]
            message_payload = {"entity_id": entity_id} # Algunos dispositivos HA esperan entity_id en el payload
            if payload:
                message_payload.update(payload)
            
            try:
                self.mqtt_client.publish(command_topic, json.dumps(message_payload))
                logging.info(f"Comando HA (descubierto) publicado: Tópico='{command_topic}', Payload='{message_payload}'")
                return True, f"Comando '{service}' para '{entity_id}' enviado (vía tópico descubierto)."
            except Exception as e:
                logging.error(f"Error al publicar comando HA (descubierto) '{domain}.{service}' para '{entity_id}': {e}")
                return False, f"Error al enviar comando HA (descubierto): {e}"

        # 3. Fallback a la lógica original para comandos de servicio de Home Assistant (si no se encuentra mapeo específico)
        topic = f"{self.base_topic}/services/{domain}/{service}"
        message_payload = {"entity_id": entity_id}
        if payload:
            message_payload.update(payload)
        
        try:
            self.mqtt_client.publish(topic, json.dumps(message_payload))
            logging.info(f"Comando HA (genérico) publicado: Tópico='{topic}', Payload='{message_payload}'")
            return True, f"Comando '{service}' para '{entity_id}' enviado (vía tópico genérico)."
        except Exception as e:
            logging.error(f"Error al publicar comando HA (genérico) '{domain}.{service}' para '{entity_id}': {e}")
            return False, f"Error al enviar comando HA (genérico): {e}"

    def get_state(self, entity_id: str):
        logging.warning(f"Solicitud de estado para {entity_id} - En una integración real, se esperaría una actualización de estado.")
        return None

    def subscribe_to_state_changes(self, entity_id: str, domain: str = None):
        if domain:
            topic = f"{self.base_topic}/{domain}/{entity_id}/state"
        else:
            topic = f"{self.base_topic}/#/{entity_id}/state" 
        self.mqtt_client.subscribe(topic)
        logging.info(f"Suscrito a cambios de estado de HA para '{entity_id}' en tópico: '{topic}'")

    def subscribe_to_all_ha_topics(self):
        """
        Utiliza el cliente MQTT para suscribirse a todos los tópicos bajo el tópico base de Home Assistant.
        Esto incluye tópicos de descubrimiento y de estado.
        """
        self.mqtt_client.subscribe_to_all_ha_topics(self.base_topic)

