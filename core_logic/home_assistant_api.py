import json
import logging
from core_logic.mqtt_client import MQTTClient # ¡Asegúrate de que esta importación sea correcta!

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HomeAssistantAPI:
    def __init__(self, mqtt_client: MQTTClient, base_topic="homeassistant"):
        self.mqtt_client = mqtt_client
        self.base_topic = base_topic
        logging.info(f"HomeAssistantAPI inicializada con tópico base: {self.base_topic}")

    def publish_command(self, domain: str, service: str, entity_id: str, payload: dict = None):
        """
        Publica un comando a Home Assistant.
        Ejemplo: domain='light', service='turn_on', entity_id='light.living_room'
        """
        topic = f"{self.base_topic}/services/{domain}/{service}"
        message = {"entity_id": entity_id}
        if payload:
            message.update(payload)
        
        self.mqtt_client.publish(topic, json.dumps(message))
        logging.info(f"Comando HA publicado: Tópico='{topic}', Payload='{message}'")

    def get_state(self, entity_id: str):
        """
        Solicita el estado actual de una entidad de Home Assistant.
        Nota: Esto es una simplificación. En una integración real,
        necesitarías un mecanismo para recibir y procesar la respuesta del estado.
        """
        # Home Assistant no tiene un tópico directo para "pedir estado" de esta manera
        # Lo que se hace es suscribirse a los tópicos de estado y esperar actualizaciones.
        # Para fines de este simulador, si necesitas un estado "simulado", puedes implementarlo aquí.
        logging.warning(f"Solicitud de estado para {entity_id} - En una integración real, se esperaría una actualización de estado.")
        # Podrías devolver un estado mock o None, o implementar un mecanismo de callback.
        return None

    def subscribe_to_state_changes(self, entity_id: str, domain: str = None):
        """
        Se suscribe a los cambios de estado de una entidad específica de Home Assistant.
        """
        if domain:
            topic = f"{self.base_topic}/{domain}/{entity_id}/state"
        else:
            # Si el dominio no se especifica, se asume un tópico genérico para la entidad
            topic = f"{self.base_topic}/#/{entity_id}/state" # Esto podría ser demasiado amplio
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

