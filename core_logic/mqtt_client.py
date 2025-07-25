import paho.mqtt.client as mqtt
import time
import logging
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MQTTClient:
    def __init__(self, broker_address, port, username, password):
        self.broker_address = broker_address
        self.port = port
        self.username = username
        self.password = password
        self.client_id = f"smart_home_ai_client-{uuid.uuid4().hex[:8]}" # ID único para el cliente
        self.client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv311)
        
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        
        self.client.username_pw_set(self.username, self.password)
        logging.info(f"Inicializando clase MQTTClient con ID: {self.client_id}...")

        self.message_callbacks = []

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info(f"Conectado exitosamente al broker MQTT en {self.broker_address}:{self.port}")
        else:
            logging.error(f"Fallo al conectar al broker MQTT, código: {rc}")

    def _on_message(self, client, userdata, msg):
        decoded_payload = msg.payload.decode(errors='ignore')
        logging.info(f"Mensaje MQTT recibido: Tópico='{msg.topic}', Payload='{decoded_payload}'")
        
        for callback_func in self.message_callbacks:
            callback_func(msg.topic, decoded_payload)

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logging.warning(f"Desconectado del broker MQTT con código: {rc}.")
            logging.warning("Intentando reconectar en 5 segundos...")
            time.sleep(5)
            try:
                self.client.reconnect()
                logging.info("Reconexión MQTT iniciada.")
            except Exception as e:
                logging.error(f"Error al intentar reconectar MQTT: {e}")
        else:
            logging.info("Desconexión MQTT limpia.")

    def _on_publish(self, client, userdata, mid):
        """
        Callback que se llama cuando un mensaje PUBLISH ha sido enviado al broker.
        """
        logging.info(f"Mensaje MQTT publicado exitosamente (MID: {mid}).")

    def connect(self):
        try:
            self.client.loop_start()
            self.client.connect(self.broker_address, self.port, 60)
            logging.info("Bucle de MQTT iniciado en segundo plano.")
        except Exception as e:
            logging.error(f"Error al conectar o iniciar bucle MQTT: {e}")

    def publish(self, topic, payload, qos=0, retain=False):
        try:
            info = self.client.publish(topic, payload, qos, retain)
            info.wait_for_publish()
            logging.info(f"Mensaje MQTT publicado: Tópico='{topic}', Payload='{payload[:50]}...'")
            return True
        except Exception as e:
            logging.error(f"Error al publicar mensaje MQTT: {e}")
            return False

    def subscribe(self, topic, qos=0):
        self.client.subscribe(topic, qos)
        logging.info(f"Suscrito al tópico MQTT: {topic}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        logging.info("Cliente MQTT desconectado.")

    def register_message_callback(self, callback_func):
        self.message_callbacks.append(callback_func)

    def subscribe_to_all_ha_topics(self, base_topic: str): # ¡base_topic como argumento!
        """
        Se suscribe a todos los tópicos bajo el tópico base de Home Assistant.
        Esto nos permitirá ver todos los mensajes de estado y descubrimiento de HA.
        """
        full_topic = f"{base_topic}/#" # Usar el argumento base_topic
        self.subscribe(full_topic)
        logging.info(f"Suscrito a todos los tópicos de Home Assistant: {full_topic}")

