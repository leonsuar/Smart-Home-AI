import json
import os
import time
import httpx
import logging
import asyncio # Importar asyncio para sleep

# Importaciones de módulos locales
from core_logic.utils import get_available_ram_mb, get_cpu_core_count, get_disk_usage_percentage, normalize_text
from core_logic.home_assistant_api import HomeAssistantAPI
from core_logic.llm_service import LLMService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RedNeuronal:
    def __init__(self, home_assistant_api: HomeAssistantAPI):
        self.memoria = self._cargar_memoria()
        self.mensajes = [] 
        self.home_assistant_api = home_assistant_api
        self.llm_service = LLMService()
        # Obtener la IP del ML Server de la variable de entorno
        self.ml_server_url = f"http://{os.getenv('ML_SERVER_INTERNAL_IP', 'ml_server')}:5001/get_embedding" # ¡CAMBIO CLAVE AQUÍ!
        self.initial_knowledge_loaded = False 

    def _cargar_memoria(self):
        if not os.path.exists('./knowledge'):
            os.makedirs('./knowledge')
            logging.info("Directorio './knowledge' creado.")

        memoria_path = './knowledge/network_state.json'
        if os.path.exists(memoria_path) and os.path.getsize(memoria_path) > 0:
            with open(memoria_path, 'r') as f:
                try:
                    memoria_cargada = json.load(f)
                    logging.info(f"Memoria cargada desde '{memoria_path}'. {len(memoria_cargada)} entradas.")
                    return memoria_cargada
                except json.JSONDecodeError as e:
                    logging.error(f"Error al decodificar JSON de '{memoria_path}': {e}. Iniciando con memoria vacía.")
                    os.rename(memoria_path, memoria_path + ".bak")
                    logging.info(f"Archivo corrupto '{memoria_path}' renombrado a '{memoria_path}.bak'.")
        logging.info(f"No se encontró '{memoria_path}' o está vacío/corrupto. Iniciando con memoria vacía.")
        return []

    def _guardar_memoria(self):
        if not os.path.exists('./knowledge'):
            os.makedirs('./knowledge')
        with open('./knowledge/network_state.json', 'w') as f:
            json.dump(self.memoria, f, indent=4)
        logging.info("Estado de la memoria guardado en './network_state.json'.")

    def log_mensaje(self, mensaje, tipo="info"):
        self.mensajes.append({
            "tiempo": time.strftime("%H:%M:%S"),
            "mensaje": mensaje,
            "tipo": tipo
        })
        if len(self.mensajes) > 100:
            self.mensajes = self.mensajes[-100:]

    async def initialize_network_automatically(self):
        if not self.initial_knowledge_loaded:
            self.log_mensaje("Inicializando red neuronal...", tipo="info")
            
            if not await self._check_ml_server_connection():
                self.log_mensaje("No se pudo establecer conexión con ML Server. La IA no funcionará correctamente.", tipo="error")
                return 

            if not self.memoria:
                self.log_mensaje("Cargando conocimiento inicial predefinido...", tipo="info")
                await self._cargar_conocimiento_inicial()
                self._guardar_memoria()
            self.initial_knowledge_loaded = True
            self.log_mensaje("Red neuronal inicializada y lista.", tipo="info")
        
    async def _cargar_conocimiento_inicial(self):
        initial_entries = [
            {"pregunta": "hola", "respuesta": "¡Hola! ¿En qué puedo ayudarte hoy?"},
            {"pregunta": "¿cómo estás?", "respuesta": "Soy una IA, no tengo sentimientos, pero estoy lista para servirte."},
            {"pregunta": "quién eres", "respuesta": "Soy tu asistente de hogar inteligente, diseñada para ayudarte a controlar tus dispositivos y responder a tus preguntas."},
            {"pregunta": "qué puedes hacer", "respuesta": "Puedo controlar dispositivos en tu hogar inteligente (si está integrado con Home Assistant), responder preguntas, y aprender nuevas respuestas."},
            {"pregunta": "qué hora es", "respuesta": f"La hora actual es {time.strftime('%H:%M:%S')}."},
            {"pregunta": "cuánta ram tengo", "respuesta": f"Tienes {get_available_ram_mb()} MB de RAM disponible."},
            {"pregunta": "cuántos núcleos de cpu tengo", "respuesta": f"Tienes {get_cpu_core_count()} núcleos de CPU."},
            {"pregunta": "cuánto espacio en disco tengo", "respuesta": f"El uso de disco es del {get_disk_usage_percentage()}%."},
            {"pregunta": "enciende la luz de la sala", "respuesta": "Comando para encender la luz de la sala enviado a Home Assistant."},
            {"pregunta": "apaga la luz de la cocina", "respuesta": "Comando para apagar la luz de la cocina enviado a Home Assistant."},
            {"pregunta": "cual es el estado de la luz de la sala", "respuesta": "Consultando el estado de la luz de la sala en Home Assistant."},
            {"pregunta": "pon música", "respuesta": "Reproduciendo tu lista de reproducción favorita."},
            {"pregunta": "dime un chiste", "respuesta": "¿Qué hace una abeja en el gimnasio? ¡Zum-ba!"},
            {"pregunta": "gracias", "respuesta": "De nada. Estoy aquí para ayudarte."},
            {"pregunta": "adiós", "respuesta": "¡Hasta pronto! Que tengas un buen día."},
        ]

        for entry in initial_entries:
            embedding = await self.get_embedding_from_pytorch(entry["pregunta"])
            if embedding:
                self.memoria.append({
                    "pregunta": normalize_text(entry["pregunta"]),
                    "respuesta": entry["respuesta"],
                    "embedding": embedding
                })
                self.log_mensaje(f"Conocimiento inicial cargado: '{entry['pregunta']}'", tipo="info")
            else:
                self.log_mensaje(f"Error: No se pudo generar embedding para el conocimiento inicial: '{entry['pregunta']}'", tipo="error")

    async def get_embedding_from_pytorch(self, text: str, retries: int = 5, delay: int = 2):
        for i in range(retries):
            try:
                logging.info(f"Solicitando embedding para '{text[:50]}...' al ML Server en {self.ml_server_url} (Intento {i+1}/{retries})")
                async with httpx.AsyncClient() as client:
                    response = await client.post(self.ml_server_url, json={"text": text}, timeout=10.0)
                    response.raise_for_status()
                    embedding_data = response.json()
                    embedding = embedding_data.get("embedding")
                    if embedding:
                        logging.info("Embedding recibido exitosamente del ML Server.")
                        return embedding
                    else:
                        logging.error(f"Respuesta de embedding inválida del ML Server: {embedding_data}")
                        return None 
            except httpx.RequestError as e:
                self.log_mensaje(f"Error de conexión con ML Server: {e} (Intento {i+1}/{retries})", tipo="error")
                logging.error(f"Error de conexión con ML Server: {e}")
                if i < retries - 1:
                    await asyncio.sleep(delay)
                else:
                    self.log_mensaje("Máximo de reintentos alcanzado para ML Server.", tipo="error")
            except httpx.HTTPStatusError as e:
                self.log_mensaje(f"Error HTTP del ML Server: {e.response.status_code} - {e.response.text} (Intento {i+1}/{retries})", tipo="error")
                logging.error(f"Error HTTP del ML Server: {e.response.status_code} - {e.response.text}")
                return None 
            except Exception as e:
                self.log_mensaje(f"Error inesperado al obtener embedding: {e} (Intento {i+1}/{retries})", tipo="error")
                logging.error(f"Error inesperado al obtener embedding: {e}")
                return None
        return None

    async def _check_ml_server_connection(self):
        test_text = "test"
        embedding = await self.get_embedding_from_pytorch(test_text, retries=5, delay=3) # Aumentar reintentos y retardo
        if embedding:
            self.log_mensaje("Conexión con ML Server establecida.", tipo="info")
            return True
        else:
            self.log_mensaje("Fallo al conectar con ML Server. La IA puede no funcionar correctamente.", tipo="error")
            return False

    async def get_local_or_llm_response(self, prompt: str) -> tuple[str, bool]:
        normalized_prompt = normalize_text(prompt)
        
        for entry in self.memoria:
            if normalized_prompt == entry["pregunta"]:
                self.log_mensaje(f"Respuesta encontrada en memoria local para: '{prompt}'", tipo="info")
                return entry["respuesta"], False

        self.log_mensaje(f"No se encontró respuesta en memoria local para: '{prompt}'. Consultando LLM...", tipo="info")

        llm_response = await self.llm_service.generate_text(prompt)
        return llm_response, True

    async def train_with_feedback(self, prompt: str, response: str, embedding: list, save_memory: bool = True):
        normalized_prompt = normalize_text(prompt)
        
        for entry in self.memoria:
            if entry["pregunta"] == normalized_prompt:
                self.log_mensaje(f"La entrada '{prompt}' ya existe en la memoria. Actualizando respuesta.", tipo="warning")
                entry["respuesta"] = response
                entry["embedding"] = embedding
                if save_memory:
                    self._guardar_memoria()
                return

        self.memoria.append({
            "pregunta": normalized_prompt,
            "respuesta": response,
            "embedding": embedding
        })
        self.log_mensaje(f"Nueva entrada añadida a la memoria: '{prompt}'", tipo="info")

        if save_memory:
            self._guardar_memoria()

    def obtener_estado_red(self):
        estado = []
        for i, entry in enumerate(self.memoria):
            estado.append({
                "id": i,
                "pregunta": entry["pregunta"],
                "respuesta_corta": entry["respuesta"][:50] + "..." if len(entry["respuesta"]) > 50 else entry["respuesta"],
                "embedding_len": len(entry["embedding"]) if entry.get("embedding") else 0
            })
            
        estado.append({"tipo": "Sistema", "RAM Disponible (MB)": get_available_ram_mb()})
        estado.append({"tipo": "Sistema", "Núcleos CPU": get_cpu_core_count()})
        estado.append({"tipo": "Sistema", "Uso Disco (%)": get_disk_usage_percentage()})

        return estado
