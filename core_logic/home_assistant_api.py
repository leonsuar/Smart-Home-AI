import json
import logging
from core_logic.mqtt_client import MQTTClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HomeAssistantAPI:
    def __init__(self, mqtt_client: MQTTClient, base_topic="homeassistant"):
        self.mqtt_client = mqtt_client
        self.base_topic = base_topic
        logging.info(f"HomeAssistantAPI inicializada con tópico base: {self.base_topic}")

    def send_ha_command(self, domain: str, service: str, entity_id: str, payload: dict = None):
        """
        Envía un comando de servicio a Home Assistant a través de MQTT.
        :param domain: El dominio del servicio (ej. 'light', 'switch').
        :param service: El servicio a llamar (ej. 'turn_on', 'turn_off').
        :param entity_id: El ID de la entidad (ej. 'light.sala_de_estar').
        :param payload: Un diccionario con datos adicionales para el servicio (opcional).
        """
        topic = f"{self.base_topic}/services/{domain}/{service}"
        message_payload = {"entity_id": entity_id}
        if payload:
            message_payload.update(payload)
        
        try:
            self.mqtt_client.publish(topic, json.dumps(message_payload))
            logging.info(f"Comando HA publicado: Tópico='{topic}', Payload='{message_payload}'")
            return True, f"Comando '{service}' para '{entity_id}' enviado."
        except Exception as e:
            logging.error(f"Error al publicar comando HA '{domain}.{service}' para '{entity_id}': {e}")
            return False, f"Error al enviar comando HA: {e}"

    def get_state(self, entity_id: str):
        """
        Solicita el estado actual de una entidad de Home Assistant.
        Nota: Esto es una simplificación. En una integración real,
        necesitarías un mecanismo para recibir y procesar la respuesta del estado.
        """
        logging.warning(f"Solicitud de estado para {entity_id} - En una integración real, se esperaría una actualización de estado.")
        return None

    def subscribe_to_state_changes(self, entity_id: str, domain: str = None):
        """
        Se suscribe a los cambios de estado de una entidad específica de Home Assistant.
        """
        if domain:
            topic = f"{self.base_topic}/{domain}/{entity_id}/state"
        else:
            topic = f"{self.base_topic}/#/{entity_id}/state" 
        self.mqtt_client.subscribe(topic)
        logging.info(f"Suscrito a cambios de estado de HA para '{entity_id}' en tópico: '{topic}'")

    def subscribe_to_all_ha_states(self):
        """
        Se suscribe a todos los mensajes de estado de Home Assistant.
        ¡ADVERTENCIA! Esto puede generar mucho tráfico MQTT.
        """
        topic = f"{self.base_topic}/#/#"
        self.mqtt_client.subscribe(topic)
        logging.info(f"Suscrito a todos los estados de Home Assistant en tópico: '{topic}'")

