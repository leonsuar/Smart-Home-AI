import json
import logging
import requests
import asyncio
from sentence_transformers import SentenceTransformer
# NO IMPORTAR HomeAssistantAPI aquí para evitar importaciones circulares.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RedNeuronal:
    def __init__(self, ml_server_ip: str, gemini_api_key: str, home_assistant_api): 
        self.ml_server_ip = ml_server_ip
        self.gemini_api_key = gemini_api_key
        self.home_assistant_api = home_assistant_api 

        self.memory = []
        self.load_memory()
        self.last_interaction = None 

    def load_memory(self):
        try:
            with open('./knowledge/network_state.json', 'r') as f:
                self.memory = json.load(f)
            logging.info(f"Memoria cargada desde './knowledge/network_state.json'. {len(self.memory)} entradas.")
        except FileNotFoundError:
            logging.warning("Archivo 'network_state.json' no encontrado. La memoria de la IA está vacía.")
            self.memory = []
        except json.JSONDecodeError:
            logging.error("Error al decodificar 'network_state.json'. La memoria de la IA está vacía.")
            self.memory = []

    def save_memory(self):
        try:
            with open('./knowledge/network_state.json', 'w') as f:
                json.dump(self.memory, f, indent=4)
            logging.info("Memoria guardada en 'network_state.json'.")
        except Exception as e:
            logging.error(f"Error al guardar la memoria: {e}")

    async def get_embedding(self, text: str):
        url = f"http://{self.ml_server_ip}:5001/get_embedding"
        try:
            response = requests.post(url, json={"text": text}, timeout=10)
            response.raise_for_status() 
            embedding = response.json().get("embedding")
            if embedding:
                return embedding
            else:
                logging.error("ML Server no devolvió un embedding válido.")
                return None
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Error de conexión con ML Server: {e}")
            raise 
        except requests.exceptions.Timeout:
            logging.error("Tiempo de espera agotado al conectar con ML Server.")
            raise
        except requests.exceptions.RequestException as e:
            logging.error(f"Error al solicitar embedding al ML Server: {e}")
            raise
        except Exception as e:
            logging.error(f"Error inesperado en get_embedding: {e}")
            raise

    async def process_command(self, command: str):
        for entry in self.memory:
            if entry["command"].lower() == command.lower():
                self.last_interaction = {"command": command, "response": entry["response"]}
                return {"action_type": "text_response", "response_text": entry["response"]}

        logging.info("No se encontró respuesta en memoria local. Consultando LLM...")
        
        discovered_devices = self.home_assistant_api.ha_entity_info 
        device_list_str = ""
        if discovered_devices:
            device_list_str = "Dispositivos disponibles:\n"
            for entity_id, info in discovered_devices.items():
                device_list_str += f"- {info['name']} (ID: {entity_id}, Dominio: {info['domain']})\n"
        else:
            device_list_str = "No se han descubierto dispositivos MQTT."

        ha_command_example_on = json.dumps({"action_type": "ha_command", "command": {"domain": "light", "service": "turn_on", "entity_id": "light.sala_de_estar", "payload": "{}"}})
        ha_command_example_off = json.dumps({"action_type": "ha_command", "command": {"domain": "fan", "service": "turn_off", "entity_id": "fan.dormitorio", "payload": "{}"}})
        text_response_example = json.dumps({"action_type": "text_response", "response_text": "La hora actual es..."})
        
        prompt = f"""
        Eres un asistente de hogar inteligente. Tu objetivo es responder a las preguntas del usuario y controlar dispositivos en su hogar.

        Aquí está la lista actual de dispositivos descubiertos en el hogar:
        {device_list_str}

        Si el usuario te pide que controles un dispositivo, debes responder con un objeto JSON que contenga:
        {{
          "action_type": "ha_command",
          "command": {{
            "domain": "dominio_de_home_assistant (ej. 'light', 'switch', 'fan')",
            "service": "servicio_de_home_assistant (ej. 'turn_on', 'turn_off', 'toggle')",
            "entity_id": "ID_de_la_entidad_de_home_assistant (ej. 'light.sala_de_estar')",
            "payload": "carga_util_JSON_para_el_servicio_como_una_cadena_de_texto_JSON (ej. '{{\\"brightness_pct\\": 50}}')"
          }}
        }}
        
        Si no se requiere un comando de Home Assistant, debes responder con un objeto JSON que contenga:
        {{
          "action_type": "text_response",
          "response_text": "Tu respuesta de texto aquí"
        }}

        Ejemplos de respuestas:
        - Para encender la luz de la sala: {ha_command_example_on}
        - Para apagar el ventilador del dormitorio: {ha_command_example_off}
        - Para preguntar la hora: {text_response_example}

        Considera los nombres amigables de los dispositivos para mapearlos a sus entity_id.
        Si el usuario pide algo que no puedes hacer o no entiendes, responde con un mensaje de texto indicando que no puedes realizar esa acción.

        Comando del usuario: {command}
        """

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "action_type": {"type": "STRING", "enum": ["ha_command", "text_response"]},
                "command": {
                    "type": "OBJECT",
                    "properties": {
                        "domain": {"type": "STRING", "description": "Dominio de Home Assistant (ej. 'light', 'switch', 'fan')"},
                        "service": {"type": "STRING", "description": "Servicio de Home Assistant (ej. 'turn_on', 'turn_off', 'toggle')"},
                        "entity_id": {"type": "STRING", "description": "ID de la entidad de Home Assistant (ej. 'light.sala_de_estar')"},
                        "payload": {"type": "STRING", "description": "Carga útil JSON para el servicio como una cadena de texto JSON (ej. '{\"brightness_pct\": 50}')"}
                    },
                    "required": ["domain", "service", "entity_id"]
                },
                "response_text": {"type": "STRING", "description": "Respuesta de texto si no se ejecuta un comando HA"}
            },
            "required": ["action_type"]
        }

        chat_history = []
        chat_history.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": chat_history,
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }
        }

        logging.info(f"Enviando prompt estructurado a Gemini: '{prompt[:100]}...' con esquema: {response_schema}")

        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
            response = await asyncio.to_thread(
                requests.post, api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=30
            )
            response.raise_for_status()
            result = response.json()
            logging.info(f"HTTP Request: POST {api_url} \"HTTP/1.1 {response.status_code} {response.reason}\"")

            if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                json_response_str = result["candidates"][0]["content"]["parts"][0]["text"]
                logging.info(f"Respuesta estructurada de Gemini recibida: {json_response_str}")
                
                try:
                    parsed_response = json.loads(json_response_str)
                    
                    if parsed_response.get("action_type") == "ha_command":
                        cmd = parsed_response.get("command")
                        if cmd and all(k in cmd for k in ["domain", "service", "entity_id"]):
                            domain = cmd["domain"]
                            service = cmd["service"]
                            entity_id = cmd["entity_id"]
                            payload_str = cmd.get("payload", "{}")
                            try:
                                payload = json.loads(payload_str if payload_str.strip() else "{}")
                            except json.JSONDecodeError:
                                logging.warning(f"Payload de Gemini no es un JSON válido: '{payload_str}'. Usando payload vacío.")
                                payload = {}
                            
                            # --- Lógica de enrutamiento de comandos ---
                            success, message = False, "Comando no ejecutado."
                            
                            # Verificar si la entidad es un dispositivo Tasmota nativo descubierto por nuestra app
                            # y si el servicio es turn_on o turn_off
                            if entity_id in self.home_assistant_api.ha_entity_info and \
                                self.home_assistant_api.ha_entity_info[entity_id].get('command_topic') and \
                                (service == "turn_on" or service == "turn_off"):
                                
                                tasmota_state = "ON" if service == "turn_on" else "OFF"
                                success, message = self.home_assistant_api.send_tasmota_command(entity_id, tasmota_state)
                            
                            # Si no es Tasmota o el comando Tasmota falló/no aplica, intentar como comando HA de servicio
                            if not success:
                                success, message = self.home_assistant_api.send_ha_command(domain, service, entity_id, payload)
                            # --- Fin de lógica de enrutamiento ---

                            if success:
                                self.last_interaction = {"command": command, "response": message}
                                return {"action_type": "text_response", "response_text": message}
                            else:
                                self.last_interaction = {"command": command, "response": f"Error al ejecutar comando: {message}"}
                                return {"action_type": "text_response", "response_text": f"Error al ejecutar comando: {message}"}
                        else:
                            logging.error(f"Comando HA incompleto o inválido de Gemini: {parsed_response}")
                            response_text = "La IA generó un comando incompleto o inválido."
                            self.last_interaction = {"command": command, "response": response_text}
                            return {"action_type": "text_response", "response_text": response_text}
                    
                    elif parsed_response.get("action_type") == "text_response":
                        response_text = parsed_response.get("response_text", "No pude generar una respuesta de texto.")
                        self.last_interaction = {"command": command, "response": response_text}
                        return {"action_type": "text_response", "response_text": response_text}
                    
                    else:
                        logging.error(f"Tipo de acción desconocido de Gemini: {parsed_response}")
                        response_text = "La IA generó un tipo de acción desconocido."
                        self.last_interaction = {"command": command, "response": response_text}
                        return {"action_type": "text_response", "response_text": response_text}

                except json.JSONDecodeError as e:
                    logging.error(f"Error al parsear la respuesta JSON de Gemini: {e} - Respuesta: {json_response_str}")
                    response_text = "La IA generó una respuesta que no pude entender. Por favor, intenta de nuevo."
                    self.last_interaction = {"command": command, "response": response_text}
                    return {"action_type": "text_response", "response_text": response_text}
            else:
                logging.error(f"Respuesta de Gemini vacía o inesperada: {result}")
                response_text = "No pude obtener una respuesta de la IA. Intenta de nuevo."
                self.last_interaction = {"command": command, "response": response_text}
                return {"action_type": "text_response", "response_text": response_text}

        except requests.exceptions.RequestException as e:
            logging.error(f"Error al conectar con la API de Gemini: {e}")
            response_text = "No se pudo establecer conexión con la IA. Por favor, verifica tu conexión a internet o la clave de API."
            self.last_interaction = {"command": command, "response": response_text}
            return {"action_type": "text_response", "response_text": response_text}
        except Exception as e:
            logging.error(f"Error inesperado al procesar comando con Gemini: {e}")
            response_text = "Ocurrió un error inesperado al procesar tu comando."
            self.last_interaction = {"command": command, "response": response_text}
            return {"action_type": "text_response", "response": response_text}

    async def save_last_interaction(self):
        if self.last_interaction:
            self.memory.append(self.last_interaction)
            self.save_memory()
            self.last_interaction = None

    def discard_last_interaction(self):
        self.last_interaction = None
