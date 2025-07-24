import paho.mqtt.client as mqtt
import time
import json
import logging
import uuid # Importar uuid para generar IDs únicos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MQTTClient:
    # Generar un client_id único para cada instancia
    def __init__(self, broker_address, port=1883, username=None, password=None, client_id_prefix="smart_home_ai_client"):
        self.client_id = f"{client_id_prefix}-{uuid.uuid4().hex[:8]}" # Añadir un sufijo único
        logging.info(f"Inicializando clase MQTTClient con ID: {self.client_id}...") 
        self.broker_address = str(broker_address) 
        self.port = port
        self.username = username
        self.password = password
        self.client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv311, clean_session=True) 
        self.on_message_callback = None 

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info(f"Conectado exitosamente al broker MQTT en {self.broker_address}:{self.port}")
        else:
            logging.error(f"Fallo al conectar al broker MQTT, código de retorno: {rc}")
            
    def _on_message(self, client, userdata, msg):
        logging.info(f"Mensaje MQTT recibido: Tópico='{msg.topic}', Payload='{msg.payload.decode()}'")
        if self.on_message_callback:
            try:
                payload_decoded = msg.payload.decode('utf-8')
                try:
                    payload_json = json.loads(payload_decoded)
                    self.on_message_callback(msg.topic, payload_json)
                except json.JSONDecodeError:
                    self.on_message_callback(msg.topic, payload_decoded)
            except Exception as e:
                logging.error(f"Error al procesar mensaje MQTT en callback: {e}")

    def _on_disconnect(self, client, userdata, rc):
        logging.warning(f"Desconectado del broker MQTT con código: {rc}.")
        if rc != 0: 
            logging.warning("Intentando reconectar en 5 segundos...")
            time.sleep(5) 
            try:
                self.client.reconnect() 
                logging.info("Reconexión MQTT iniciada.")
            except Exception as e:
                logging.error(f"Error al intentar reconectar MQTT: {e}")
        else:
            logging.info("Desconexión MQTT limpia.")

    def set_message_callback(self, callback_func):
        self.on_message_callback = callback_func

    def connect(self):
        try:
            self.client.connect(self.broker_address, self.port, 60)
            self.client.loop_start() 
            logging.info("Bucle de MQTT iniciado en segundo plano.")
        except Exception as e:
            logging.error(f"Error al iniciar la conexión MQTT: {e}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        logging.info("Cliente MQTT desconectado.")

    def publish(self, topic, payload, qos=0, retain=False):
        try:
            if isinstance(payload, dict) or isinstance(payload, list):
                payload = json.dumps(payload) 
            
            self.client.publish(topic, payload, qos, retain)
            logging.info(f"Mensaje MQTT publicado: Tópico='{topic}', Payload='{payload[:100]}...'")
        except Exception as e:
            logging.error(f"Error al publicar mensaje MQTT: {e}")

    def subscribe(self, topic, qos=0):
        try:
            self.client.subscribe(topic, qos)
            logging.info(f"Suscrito al tópico MQTT: '{topic}'")
        except Exception as e:
            logging.error(f"Error al suscribirse al tópico MQTT '{topic}': {e}")

if __name__ == "__main__":
    BROKER_ADDRESS_TEST = "192.168.1.11" 
    BROKER_PORT_TEST = 1883 
    MQTT_USERNAME_TEST = "leo"
    MQTT_PASSWORD_TEST = "Kolke.2576"

    mqtt_client_test = MQTTClient(BROKER_ADDRESS_TEST, BROKER_PORT_TEST, MQTT_USERNAME_TEST, MQTT_PASSWORD_TEST)

    def handle_incoming_message(topic, payload):
        print(f"Callback de la aplicación: Recibido en {topic}: {payload}")

    mqtt_client_test.set_message_callback(handle_incoming_message)
    mqtt_client_test.connect()

    time.sleep(2) 
    mqtt_client_test.publish("home/test/status", "Hello from Smart Home AI!")
    mqtt_client_test.publish("home/light/living_room/set", {"state": "ON", "brightness": 255})
    mqtt_client_test.subscribe("home/+/status")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Desconectando MQTT...")
        mqtt_client_test.disconnect()
        print("Desconectado.")
