import logging
from .mqtt_client import MQTTClient # Importar el cliente MQTT
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HomeAssistantAPI:
    def __init__(self, mqtt_client: MQTTClient, base_topic="homeassistant"):
        """
        Inicializa la interfaz con Home Assistant usando un cliente MQTT.
        :param mqtt_client: Una instancia de MQTTClient.
        :param base_topic: El tópico base para los mensajes de Home Assistant (ej. "homeassistant").
        """
        self.mqtt_client = mqtt_client
        self.base_topic = base_topic
        logging.info(f"HomeAssistantAPI inicializada con tópico base: {self.base_topic}")

    def _publish_command(self, domain, entity_id, command_payload):
        """
        Función interna para publicar un comando a Home Assistant vía MQTT.
        Home Assistant espera comandos en tópicos específicos, a menudo con un payload JSON.
        Ejemplo: homeassistant/light/mylight/set
        """
        topic = f"{self.base_topic}/{domain}/{entity_id}/set"
        self.mqtt_client.publish(topic, command_payload)
        logging.info(f"Comando enviado a Home Assistant: {domain}.{entity_id} con payload {command_payload}")

    def turn_on_light(self, entity_id, brightness=None, color=None):
        """
        Enciende una luz en Home Assistant.
        :param entity_id: ID de la entidad de la luz (ej. "living_room_light").
        :param brightness: Opcional, brillo (0-255).
        :param color: Opcional, color en formato RGB o nombre.
        """
        payload = {"state": "ON"}
        if brightness is not None:
            payload["brightness"] = brightness
        if color is not None:
            payload["color"] = color # Home Assistant espera diferentes formatos de color
        self._publish_command("light", entity_id, payload)

    def turn_off_light(self, entity_id):
        """
        Apaga una luz en Home Assistant.
        :param entity_id: ID de la entidad de la luz.
        """
        payload = {"state": "OFF"}
        self._publish_command("light", entity_id, payload)

    def set_thermostat_temperature(self, entity_id, temperature):
        """
        Establece la temperatura de un termostato.
        :param entity_id: ID de la entidad del termostato.
        :param temperature: Temperatura deseada.
        """
        payload = {"temperature": temperature}
        self._publish_command("climate", entity_id, payload)

    def activate_alarm(self, entity_id, code=None, mode="armed_away"):
        """
        Activa una alarma en Home Assistant.
        :param entity_id: ID de la entidad de la alarma.
        :param code: Código de alarma si es necesario.
        :param mode: Modo de activación (ej. "armed_away", "armed_home").
        """
        payload = {"state": mode}
        if code:
            payload["code"] = code
        self._publish_command("alarm_control_panel", entity_id, payload)
        logging.info(f"Alarma {entity_id} activada en modo {mode}.")

    def disarm_alarm(self, entity_id, code=None):
        """
        Desactiva una alarma en Home Assistant.
        :param entity_id: ID de la entidad de la alarma.
        :param code: Código de alarma si es necesario.
        """
        payload = {"state": "DISARMED"}
        if code:
            payload["code"] = code
        self._publish_command("alarm_control_panel", entity_id, payload)
        logging.info(f"Alarma {entity_id} desactivada.")

    # Puedes añadir más funciones aquí para otros dominios de Home Assistant
    # Por ejemplo: switch, cover, media_player, etc.

    def subscribe_to_state_changes(self, entity_id, domain="sensor"):
        """
        Se suscribe a los cambios de estado de una entidad específica en Home Assistant.
        Home Assistant publica estados en tópicos como homeassistant/sensor/mysensor/state
        :param entity_id: ID de la entidad.
        :param domain: Dominio de la entidad (ej. "sensor", "binary_sensor").
        """
        topic = f"{self.base_topic}/{domain}/{entity_id}/state"
        self.mqtt_client.subscribe(topic)
        logging.info(f"Suscrito a cambios de estado para {domain}.{entity_id}")

    def subscribe_to_all_ha_states(self):
        """
        Se suscribe a todos los tópicos de estado de Home Assistant.
        ¡Útil para monitorear, pero puede generar mucho tráfico!
        """
        topic = f"{self.base_topic}/#/#/state" # Tópico wildcard para todos los estados
        self.mqtt_client.subscribe(topic)
        logging.info(f"Suscrito a todos los cambios de estado de Home Assistant.")


# Ejemplo de uso (para pruebas, no se ejecuta directamente en la app)
if __name__ == "__main__":
    # Configuración de prueba (debería coincidir con tu Home Assistant)
    BROKER_ADDRESS = "localhost" # Cambia esto a la IP/hostname de tu broker MQTT
    BROKER_PORT = 1883
    MQTT_USERNAME = "your_mqtt_user"
    MQTT_PASSWORD = "your_mqtt_password"

    # Inicializar el cliente MQTT
    mqtt_client_test = MQTTClient(BROKER_ADDRESS, BROKER_PORT, MQTT_USERNAME, MQTT_PASSWORD)
    mqtt_client_test.connect()

    # Inicializar la API de Home Assistant con el cliente MQTT
    ha_api = HomeAssistantAPI(mqtt_client_test)

    # Definir un callback para manejar los mensajes que Home Assistant envía
    def ha_message_handler(topic, payload):
        print(f"HA API recibió mensaje: Tópico='{topic}', Payload={payload}")
        if "home/sensor/motion_garden/state" in topic and payload == "ON":
            print("¡Movimiento detectado en el jardín!")
            # Aquí podrías llamar a una función de neuron_network para procesar el evento
            # Por ejemplo: neuron_network.process_security_event("motion_garden")

    mqtt_client_test.set_message_callback(ha_message_handler)

    # Suscribirse a un sensor de movimiento de ejemplo
    ha_api.subscribe_to_state_changes("motion_garden", domain="binary_sensor")

    # Enviar algunos comandos de prueba
    time.sleep(3) # Dar tiempo para conectar y suscribirse
    ha_api.turn_on_light("living_room_light", brightness=150)
    time.sleep(1)
    ha_api.turn_off_light("living_room_light")
    time.sleep(1)
    ha_api.set_thermostat_temperature("main_thermostat", 22.5)
    time.sleep(1)
    ha_api.activate_alarm("house_alarm", mode="armed_away")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Desconectando API de Home Assistant...")
        mqtt_client_test.disconnect()
        print("Desconectado.")
