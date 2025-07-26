import paho.mqtt.client as mqtt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MQTTClient:
    # ¡CORREGIDO! Ahora acepta todos los argumentos pasados desde app.py
    def __init__(self, broker_address, broker_port, username, password, client_id, message_callback):
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.client_id = client_id
        self.message_callback = message_callback # Callback para procesar mensajes recibidos

        # Crear una instancia del cliente MQTT
        # Usamos VERSION1 por compatibilidad, si hay problemas se puede intentar VERSION2
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=self.client_id)
        
        # Asignar funciones de callback
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        # Configurar credenciales si se proporcionan
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        logging.info(f"Inicializando clase MQTTClient con ID: {self.client_id}...")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info(f"Conectado exitosamente al broker MQTT en {self.broker_address}:{self.broker_port}")
            # Suscribirse a los tópicos después de la conexión exitosa
            # Esto se maneja en HomeAssistantAPI y NeuronNetwork, pero aquí podemos poner una suscripción básica si es necesario.
            # Por ahora, HomeAssistantAPI se encarga de las suscripciones.
        else:
            logging.error(f"Fallo al conectar al broker MQTT. Código de retorno: {rc}")

    def _on_message(self, client, userdata, msg):
        # logging.info(f"Mensaje recibido: Tópico='{msg.topic}', Payload='{msg.payload.decode()}'")
        if self.message_callback:
            self.message_callback(msg.topic, msg.payload.decode())

    def connect(self):
        try:
            self.client.connect(self.broker_address, self.broker_port, 60)
        except Exception as e:
            logging.error(f"Error al intentar conectar al broker MQTT: {e}")

    def loop_start(self):
        self.client.loop_start() # Inicia un hilo en segundo plano para manejar la red MQTT
        logging.info("Bucle de MQTT iniciado en segundo plano.")

    def loop_stop(self):
        self.client.loop_stop() # Detiene el hilo en segundo plano
        logging.info("Bucle de MQTT detenido.")

    def publish(self, topic, payload):
        try:
            self.client.publish(topic, payload)
            # logging.info(f"Mensaje publicado: Tópico='{topic}', Payload='{payload}'")
        except Exception as e:
            logging.error(f"Error al publicar mensaje MQTT: {e}")

    def subscribe(self, topic):
        try:
            self.client.subscribe(topic)
            logging.info(f"Suscrito al tópico MQTT: {topic}")
        except Exception as e:
            logging.error(f"Error al suscribirse al tópico MQTT: {e}")

    def subscribe_to_all_ha_topics(self, base_topic):
        # Suscribirse a todos los tópicos de configuración de Home Assistant
        self.subscribe(f"{base_topic}/#")
        logging.info(f"Suscrito a todos los tópicos de Home Assistant: {base_topic}/#")

        # Suscripciones adicionales para Tasmota si no usa HA Discovery
        self.subscribe("tasmota/discovery/+/config")
        logging.info("Suscrito al tópico de descubrimiento nativo de Tasmota: tasmota/discovery/+/config")
        self.subscribe("tele/+/STATE")
        logging.info("Suscrito a tópicos de telemetría de Tasmota: tele/+/STATE")
        self.subscribe("stat/+/POWER")
        logging.info("Suscrito a tópicos de estado de Power de Tasmota: stat/+/POWER")
