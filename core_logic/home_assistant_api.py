import paho.mqtt.client as mqtt
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MQTTClient:
    def __init__(self, broker_address, port=1883, username=None, password=None, client_id="smart_home_ai_client"):
        self.broker_address = broker_address
        self.port = port
        self.username = username
        self.password = password
        self.client_id = client_id
        self.client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv311)
        self.on_message_callback = None # Callback para manejar mensajes entrantes

        # Asignar callbacks de Paho MQTT
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

    def _on_connect(self, client, userdata, flags, rc):
        """Callback cuando el cliente se conecta al broker MQTT."""
        if rc == 0:
            logging.info(f"Conectado exitosamente al broker MQTT en {self.broker_address}:{self.port}")
            # Puedes suscribirte a tópicos aquí automáticamente al conectar
            # Por ejemplo, para recibir estados de Home Assistant
            # self.subscribe("homeassistant/#") 
        else:
            logging.error(f"Fallo al conectar al broker MQTT, código de retorno: {rc}")

    def _on_message(self, client, userdata, msg):
        """Callback cuando se recibe un mensaje MQTT."""
        logging.info(f"Mensaje MQTT recibido: Tópico='{msg.topic}', Payload='{msg.payload.decode()}'")
        if self.on_message_callback:
            try:
                # Intentar decodificar el payload como JSON si es posible
                payload_decoded = msg.payload.decode('utf-8')
                try:
                    payload_json = json.loads(payload_decoded)
                    self.on_message_callback(msg.topic, payload_json)
                except json.JSONDecodeError:
                    # Si no es JSON, pasar el string plano
                    self.on_message_callback(msg.topic, payload_decoded)
            except Exception as e:
                logging.error(f"Error al procesar mensaje MQTT en callback: {e}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback cuando el cliente se desconecta del broker MQTT."""
        logging.warning(f"Desconectado del broker MQTT con código: {rc}. Intentando reconectar...")
        # Puedes añadir lógica de reconexión aquí si lo deseas
        # self.connect() # Esto intentaría reconectar inmediatamente

    def set_message_callback(self, callback_func):
        """Establece una función de callback para procesar mensajes MQTT entrantes."""
        self.on_message_callback = callback_func

    def connect(self):
        """Inicia la conexión al broker MQTT en un bucle no bloqueante."""
        try:
            # loop_start() inicia un hilo en segundo plano que maneja la conexión, reconexión y procesamiento de mensajes.
            self.client.connect(self.broker_address, self.port, 60)
            self.client.loop_start() 
            logging.info("Bucle de MQTT iniciado en segundo plano.")
        except Exception as e:
            logging.error(f"Error al iniciar la conexión MQTT: {e}")

    def disconnect(self):
        """Desconecta el cliente MQTT."""
        self.client.loop_stop()
        self.client.disconnect()
        logging.info("Cliente MQTT desconectado.")

    def publish(self, topic, payload, qos=0, retain=False):
        """
        Publica un mensaje en un tópico MQTT.
        :param topic: El tópico al que publicar.
        :param payload: El mensaje a publicar (puede ser string o bytes).
        :param qos: Calidad de Servicio (0, 1, 2).
        :param retain: Si el mensaje debe ser retenido por el broker.
        """
        try:
            if isinstance(payload, dict) or isinstance(payload, list):
                payload = json.dumps(payload) # Convertir dict/list a JSON string
            
            self.client.publish(topic, payload, qos, retain)
            logging.info(f"Mensaje MQTT publicado: Tópico='{topic}', Payload='{payload[:100]}...'")
        except Exception as e:
            logging.error(f"Error al publicar mensaje MQTT: {e}")

    def subscribe(self, topic, qos=0):
        """
        Se suscribe a un tópico MQTT.
        :param topic: El tópico al que suscribirse.
        :param qos: Calidad de Servicio (0, 1, 2).
        """
        try:
            self.client.subscribe(topic, qos)
            logging.info(f"Suscrito al tópico MQTT: '{topic}'")
        except Exception as e:
            logging.error(f"Error al suscribirse al tópico MQTT '{topic}': {e}")

# Ejemplo de uso (para pruebas, no se ejecuta directamente en la app)
if __name__ == "__main__":
    # Configura tu broker MQTT de Home Assistant aquí
    # Si Home Assistant está en Docker, usa el nombre del servicio (ej. 'homeassistant')
    # Si está en la misma red, usa su IP (ej. '192.168.1.100')
    BROKER_ADDRESS = "localhost" # Cambia esto a la IP/hostname de tu broker MQTT
    BROKER_PORT = 1883
    MQTT_USERNAME = "your_mqtt_user" # Reemplaza con tu usuario MQTT de Home Assistant
    MQTT_PASSWORD = "your_mqtt_password" # Reemplaza con tu contraseña MQTT de Home Assistant

    mqtt_client = MQTTClient(BROKER_ADDRESS, BROKER_PORT, MQTT_USERNAME, MQTT_PASSWORD)

    def handle_incoming_message(topic, payload):
        print(f"Callback de la aplicación: Recibido en {topic}: {payload}")

    mqtt_client.set_message_callback(handle_incoming_message)
    mqtt_client.connect()

    # Publicar un mensaje de prueba después de un breve retraso
    time.sleep(2) 
    mqtt_client.publish("home/test/status", "Hello from Smart Home AI!")
    mqtt_client.publish("home/light/living_room/set", {"state": "ON", "brightness": 255})

    # Suscribirse a un tópico para ver mensajes
    mqtt_client.subscribe("home/+/status")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Desconectando MQTT...")
        mqtt_client.disconnect()
        print("Desconectado.")

