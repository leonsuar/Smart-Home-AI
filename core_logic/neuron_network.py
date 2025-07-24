import uuid
import random
import time
import re
import os
import httpx
import numpy as np

# Importar los módulos con rutas absolutas
from core_logic.knowledge_manager import KnowledgeManager
from core_logic.llm_service import generate_llm_response # ¡Verificado!
from core_logic.utils import get_available_ram_mb, get_cpu_core_count, get_disk_usage_percentage, normalize_text
from core_logic.home_assistant_api import HomeAssistantAPI

# --- Clase Neurona ---
class Neurona:
    def __init__(self, neurona_id=None, peso=0.5):
        self.id = neurona_id if neurona_id else str(uuid.uuid4())
        self.peso = peso
        self.conexiones_salida = []

    def procesar_entrada(self, entrada):
        return entrada * self.peso

    def copiar(self):
        mutacion_inicial = random.uniform(-0.1, 0.1)
        nueva_neurona = Neurona(peso=self.peso + mutacion_inicial)
        return nueva_neurona

    def mutar(self, factor_mutacion=0.1):
        old_peso = self.peso
        self.peso += random.uniform(-factor_mutacion, factor_mutacion)
        self.peso = max(0, min(1, self.peso))
        return self.peso

    def conectar_a(self, otra_neurona_id):
        if otra_neurona_id not in self.conexiones_salida:
            self.conexiones_salida.append(otra_neurona_id)

    def learn(self, error_signal, learning_rate=0.01):
        old_peso = self.peso
        self.peso += error_signal * learning_rate
        self.peso = max(0, min(1, self.peso))
        

    def __str__(self):
        return f"Neurona(ID: {self.id}, Peso: {self.peso:.2f}, Conexiones: {len(self.conexiones_salida)})"

    def to_dict(self):
        return {
            "id": self.id,
            "peso": self.peso,
            "conexiones_salida": self.conexiones_salida
        }

    @classmethod
    def from_dict(cls, data):
        neuron = cls(neurona_id=data["id"], peso=data["peso"])
        neuron.conexiones_salida = data["conexiones_salida"]
        return neuron

# --- Clase RedNeuronal ---
class RedNeuronal:
    SELF_DESCRIPTION_KEYWORDS = {
        "neurona": "Una neurona es la unidad básica de mi red. Recibe, procesa y transmite señales.",
        "red neuronal": "Soy una red neuronal, un sistema de neuronas interconectadas que trabajan juntas. Mi propósito es simular el procesamiento de información.",
        "peso": "El peso de una neurona es su factor de influencia. Se ajusta para que mi red 'aprende'.",
        "mutar": "Mutar es cambiar el peso de una neurona aleatoriamente, lo que ayuda a la red a explorar nuevas configuraciones.",
        "aprender": "Mi aprendizaje implica ajustar los pesos de mis neuronas para mejorar mis respuestas, a menudo con la ayuda de una IA externa.",
        "conectar": "Las neuronas se conectan para formar caminos por donde fluye la información en la red.",
        "calcular": "Sí, puedo realizar operaciones aritméticas básicas como suma, resta, multiplicación y división. Usa el comando 'calcular'.",
        "hola": "¡Hola! ¿Cómo puedo asistirte hoy?",
        "funcionas": "Mi funcionamiento se basa en el procesamiento de señales a través de mis neuronas interconectadas.",
        "que eres": "Soy una red neuronal simplificada. Puedo crear, mutar y conectar neuronas, y también realizar cálculos básicos.",
        "como funcionas": "Proceso información a través de neuronas interconectadas. Cada neurona tiene un peso que influye en su salida. Intento responderte localmente y, si no sé, consulto a una IA externa.",
        "cual es tu proposito": "Mi propósito es simular el comportamiento básico de una red neuronal y demostrar cómo puede procesar información y 'aprender' de interacciones.",
        "cuantas neuronas tienes": "La cantidad de neuronas que tengo se ajusta automáticamente al iniciar, basándose en los recursos de hardware de tu sistema (RAM y CPU).",
        "donde estas": "Soy un programa de software, existo dentro de los sistemas informáticos que me ejecutan. No tengo una ubicación física en el mundo real.",
        "donde vives": "Soy un programa de software, existo dentro de los sistemas informáticos que me ejecutan. No tengo una ubicación física en el mundo real.",
        "sabes donde vivo": "Soy un programa de software, existo dentro de los sistemas informáticos que me ejecutan. No tengo una ubicación física en el mundo real.",
        "como te llamas": "Mi nombre es {ai_name}. ¡Un placer!"
    }

    OUT_OF_SCOPE_KEYWORDS = [
        "cámara ip", "camara ip", "red de computadoras", "revisar red", "tomar imagen", "proporcionar datos", "autentiques", "red local", "computadoras", "escanear", "acceder", "conectar a internet"
    ]

    def __init__(self, home_assistant_api: HomeAssistantAPI = None):
        self.neuronas = {}
        self.mensajes = []
        self.knowledge_manager = KnowledgeManager()
        self.home_assistant_api = home_assistant_api
        self.pytorch_embedding_url = "http://ml_server:5001/get_embedding"

    def añadir_neurona(self, neurona):
        if neurona.id in self.neuronas:
            self.log_mensaje(f"Error: La neurona {neurona.id} ya existe.", tipo="error")
            return False
        self.neuronas[neurona.id] = neurona
        self.log_mensaje(f"Neurona {neurona.id} añadida a la red.")
        return True

    def obtener_neurona(self, neurona_id):
        return self.neuronas.get(neurona_id)

    def replicar_neurona(self, neurona_id):
        neurona_original = self.obtener_neurona(neurona_id)
        if neurona_original:
            nueva_neurona = neurona_original.copiar()
            self.añadir_neurona(nueva_neurona)
            self.knowledge_manager.save_state()
            return nueva_neurona
        self.log_mensaje(f"Error: No se pudo replicar. Neurona {neurona_id} no encontrada.", tipo="error")
        return None

    def mutar_neurona(self, neurona_id, factor=0.1):
        neurona = self.obtener_neurona(neurona_id)
        if neurona:
            old_peso = neurona.peso
            neurona.mutar(factor)
            self.log_mensaje(f"Neurona {neurona_id} mutada. Peso: {old_peso:.2f} -> {neurona.peso:.2f}.")
            self.knowledge_manager.save_state()
            return True
        self.log_mensaje(f"Error: No se pudo mutar. Neurona {neurona_id} no encontrada.", tipo="error")
        return False

    def establecer_conexion(self, id_origen, id_destino):
        neurona_origen = self.obtener_neurona(id_origen)
        neurona_destino = self.obtener_neurona(id_destino)
        if neurona_origen and neurona_destino:
            neurona_origen.conectar_a(id_destino)
            self.log_mensaje(f"Conexión establecida: {id_origen} -> {id_destino}.")
            self.knowledge_manager.save_state()
            return True
        self.log_mensaje(f"Error: No se pudo establecer la conexión entre {id_origen} y {id_destino}. Asegúrese de que ambas neuronas existan.", tipo="error")
        return False

    def enviar_activacion(self, id_origen, entrada):
        neurona_origen = self.obtener_neurona(id_origen)
        if not neurona_origen:
            self.log_mensaje(f"Error: Neurona origen {id_origen} no encontrada para activar.", tipo="error")
            return

        salida_origen = neurona_origen.procesar_entrada(entrada)
        self.log_mensaje(f"Neurona {id_origen} (peso: {neurona_origen.peso:.2f}) recibe entrada {entrada}, produce salida: {salida_origen:.2f}.")

        if not neurona_origen.conexiones_salida:
            self.log_mensaje(f"  Neurona {id_origen} no tiene conexiones de salida.", tipo="info")
            return

        for id_destino in neurona_origen.conexiones_salida:
            neurona_destino = self.obtener_neurona(id_destino)
            if neurona_destino:
                self.log_mensaje(f"  Neurona {id_origen} envía {salida_origen:.2f} a Neurona {id_destino}.")
            else:
                self.log_mensaje(f"  Advertencia: Conexión a neurona {id_destino} no encontrada (posiblemente eliminada?).", tipo="warning")

    def log_mensaje(self, mensaje, tipo="info"):
        self.mensajes.append({"tiempo": time.strftime("%H:%M:%S"), "mensaje": mensaje, "tipo": tipo})
        if len(self.mensajes) > 100:
            self.mensajes.pop(0)

    def obtener_estado_red(self):
        estado = []
        for n_id, neurona in self.neuronas.items():
            estado.append({
                "id": neurona.id,
                "peso": f"{neurona.peso:.2f}",
                "conexiones": neurona.conexiones_salida
            })
        return estado

    def get_network_output_for_text(self, text_input):
        if not self.neuronas:
            return 0.0, None

        numerical_input = abs(sum(ord(char) for char in text_input)) % 1000

        random_neuron_id = random.choice(list(self.neuronas.keys()))
        neuron = self.neuronas[random_neuron_id]
        
        output = neuron.procesar_entrada(numerical_input)
        return output, random_neuron_id

    async def get_embedding_from_pytorch(self, text: str) -> list:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.pytorch_embedding_url, json={"text": text})
                response.raise_for_status()
                result = response.json()
                if "embedding" in result:
                    return result["embedding"]
                else:
                    self.log_mensaje(f"Error Embedding Server: Respuesta inesperada: {result}", tipo="error")
                    return None
        except httpx.RequestError as e:
            self.log_mensaje(f"Error de conexión al servidor de embeddings: {e}", tipo="error")
            return None
        except httpx.HTTPStatusError as e:
            self.log_mensaje(f"Error HTTP del servidor de embeddings {e.response.status_code}: {e.response.text}", tipo="error")
            return None
        except Exception as e:
            self.log_mensaje(f"Error inesperado al obtener embedding: {e}", tipo="error")
            return None

    # Este método ya no se usa para generación de texto, solo para embeddings
    # Lo mantengo comentado para referencia, pero no se llama en get_local_or_llm_response
    # async def generate_text_with_pytorch(self, prompt: str) -> str:
    #     try:
    #         async with httpx.AsyncClient() as client:
    #             response = await client.post(self.pytorch_text_gen_url, json={"prompt": prompt, "max_length": 100})
    #             response.raise_for_status()
    #             result = response.json()
    #             if "generated_text" in result:
    #                 generated_text = result["generated_text"]
    #                 if generated_text.startswith(prompt):
    #                     return generated_text[len(prompt):].strip()
    #                 return generated_text
    #             else:
    #                 self.log_mensaje(f"Error PyTorch Text Gen Server: Respuesta inesperada: {result}", tipo="error")
    #                 return "Error al generar texto con el modelo local."
    #     except httpx.RequestError as e:
    #         self.log_mensaje(f"Error de conexión al servidor PyTorch para generación de texto: {e}", tipo="error")
    #         return "No pude conectar con el modelo local de PyTorch."
    #     except httpx.HTTPStatusError as e:
    #         self.log_mensaje(f"Error HTTP del servidor PyTorch para generación de texto {e.response.status_code}: {e.response.text}", tipo="error")
    #         return f"El modelo local de PyTorch devolvió un error: {e.response.status_code}"
    #     except Exception as e:
    #         self.log_mensaje(f"Error inesperado al generar texto con PyTorch: {e}", tipo="error")
    #         return "Error inesperado del modelo local de PyTorch."

    async def get_local_or_llm_response(self, text_input):
        original_text_input_lower = text_input.lower()
        normalized_text_input = normalize_text(text_input) 

        self.log_mensaje(f"DEBUG: Entrada original: '{text_input}', Normalizada: '{normalized_text_input}'", tipo="debug")

        # --- Manejo de Comandos de Domótica ---
        if self.home_assistant_api:
            if "enciende la luz" in normalized_text_input:
                entity_id = "living_room_light"
                self.log_mensaje(f"Comando de domótica detectado: Encender luz {entity_id}", tipo="info")
                self.home_assistant_api.turn_on_light(entity_id)
                return f"¡Claro! Encendiendo la luz del {entity_id.replace('_light', '').replace('_', ' ')}.", False
            elif "apaga la luz" in normalized_text_input:
                entity_id = "living_room_light"
                self.log_mensaje(f"Comando de domótica detectado: Apagar luz {entity_id}", tipo="info")
                self.home_assistant_api.turn_off_light(entity_id)
                return f"¡Hecho! Apagando la luz del {entity_id.replace('_light', '').replace('_', ' ')}.", False
            elif "activa la alarma" in normalized_text_input:
                entity_id = "house_alarm"
                self.log_mensaje(f"Comando de domótica detectado: Activando alarma {entity_id}", tipo="info")
                self.home_assistant_api.activate_alarm(entity_id, mode="armed_away")
                return f"Alarma {entity_id} activada en modo ausente.", False
            elif "desactiva la alarma" in normalized_text_input:
                entity_id = "house_alarm"
                self.log_mensaje(f"Comando de domótica detectado: Desactivando alarma {entity_id}", tipo="info")
                self.home_assistant_api.disarm_alarm(entity_id)
                return f"Alarma {entity_id} desactivada.", False
        
        # 1. Intentar responder si se pregunta por el nombre del usuario
        if "como me llamo" in normalized_text_input or \
           "cual es mi nombre" in normalized_text_input or \
           "sabes mi nombre" in normalized_text_input:
            self.log_mensaje(f"DEBUG: Matched user name query.", tipo="debug")
            user_name = self.knowledge_manager.get_user_name()
            if user_name:
                self.log_mensaje(f"Red Local: Recordando el nombre para: '{original_text_input_lower}'", tipo="info")
                return f"¡Te llamas {user_name}! Un placer conocerte.", False
            else:
                self.log_mensaje(f"Red Local: No recuerdo tu nombre. No está en mi memoria. Para que lo recuerde, puedes decirme 'Me llamo [tu nombre]'.", tipo="info")
                return "No recuerdo tu nombre. Para que lo recuerde, puedes decirme 'Me llamo [tu nombre]'.", False
        
        # 2. Verificar si la pregunta es sobre la propia red neuronal (usando palabras clave externalizadas - coincidencia exacta)
        for kw, response_template in self.knowledge_manager.self_description_keywords.items():
            if kw == normalized_text_input:
                self.log_mensaje(f"DEBUG: Matched EXACT self-description keyword: '{kw}'", tipo="debug")
                self.log_mensaje(f"Red Local: Respondiendo desde conocimiento predefinido sobre mí para: '{original_text_input_lower}' (palabra clave exacta: '{kw}')", tipo="info")
                if "{ai_name}" in response_template:
                    return response_template.format(ai_name=self.knowledge_manager.get_ai_name()), False
                return response_template, False

        # 3. Verificar si la pregunta es sobre la propia red neuronal (usando palabras clave externalizadas - búsqueda SEMÁNTICA)
        self.log_mensaje(f"DEBUG: Generando embedding para búsqueda semántica de palabras clave de auto-descripción: '{text_input}'...", tipo="debug")
        query_embedding_for_identity = await self.get_embedding_from_pytorch(text_input)

        if query_embedding_for_identity is not None:
            similar_identity_responses = self.knowledge_manager.find_similar_response_by_embedding(
                query_embedding_for_identity, 
                self.knowledge_manager.self_description_embeddings,
                self.knowledge_manager.self_description_keywords,
                top_k=1, 
                threshold=0.8
            )
            if similar_identity_responses:
                best_identity_response_text, similarity_score = similar_identity_responses[0]
                self.log_mensaje(f"DEBUG: Matched SEMANTIC self-description with similarity {similarity_score:.2f}.", tipo="debug")
                self.log_mensaje(f"Red Local: Respondiendo desde conocimiento predefinido sobre mí (semántico) para: '{original_text_input_lower}' con similitud {similarity_score:.2f}.", tipo="info")
                
                if "{ai_name}" in best_identity_response_text:
                    return best_identity_response_text.format(ai_name=self.knowledge_manager.get_ai_name()), False
                return best_identity_response_text, False
        else:
            self.log_mensaje("Advertencia: No se pudo obtener el embedding para la búsqueda semántica de identidad.", tipo="warning")

        # 4. Verificar si la pregunta implica una acción fuera de alcance (usando palabras clave externalizadas)
        for oskw in self.knowledge_manager.out_of_scope_keywords:
            if oskw in normalized_text_input:
                self.log_mensaje(f"DEBUG: Matched out-of-scope keyword: '{oskw}'", tipo="debug")
                self.log_mensaje(f"Red Local: Detecté palabras clave relacionadas con una acción que está fuera de mi alcance (como '{oskw}'). No puedo realizar operaciones en redes físicas o cámaras IP. Consultaré a una IA externa para información general.", tipo="info")
                llm_response = await generate_llm_response(text_input)
                self.log_mensaje(f"IA (Gemini): {llm_response}", tipo="info")
                return llm_response, True

        # 5. Intentar búsqueda semántica en conocimiento general (default_knowledge y learned_responses)
        self.log_mensaje(f"Red Local: Generando embedding para búsqueda semántica de conocimiento general de: '{text_input}'...", tipo="info")
        query_embedding_general = await self.get_embedding_from_pytorch(text_input)
        
        if query_embedding_general is not None:
            combined_embeddings = {**self.knowledge_manager.general_knowledge_embeddings, **self.knowledge_manager.learned_responses_embeddings}
            combined_texts = {**self.knowledge_manager.general_knowledge, **self.knowledge_manager.learned_responses}

            similar_general_responses = self.knowledge_manager.find_similar_response_by_embedding(
                query_embedding_general, 
                combined_embeddings, 
                combined_texts, 
                top_k=1, 
                threshold=0.75
            )
            
            if similar_general_responses:
                best_response, similarity_score = similar_general_responses[0]
                self.log_mensaje(f"DEBUG: Matched general semantic knowledge with similarity {similarity_score:.2f}.", tipo="debug")
                self.log_mensaje(f"Respondiendo desde la memoria local (semántica general) para: '{text_input}' con similitud {similarity_score:.2f}.", tipo="info")
                return best_response, False
            else:
                self.log_mensaje(f"DEBUG: No general semantic match found above threshold.", tipo="debug")
                self.log_mensaje(f"No se encontraron respuestas semánticamente similares en la memoria local para: '{text_input}' (similitud < 0.75).", tipo="info")
        else:
            self.log_mensaje("Advertencia: No se pudo obtener el embedding para la búsqueda semántica general.", tipo="warning")

        # 6. Fallback a Gemini
        if len(self.neuronas) > 0:
            self.log_mensaje(f"DEBUG: Falling back to Gemini LLM.", tipo="debug")
            self.log_mensaje(f"Red Local no tiene respuesta directa. Consultando a IA (Gemini) para: '{text_input}'...", tipo="info")
            llm_response = await generate_llm_response(text_input)
            self.log_mensaje(f"IA (Gemini): {llm_response}", tipo="info")
            
            return llm_response, True
        else:
            self.log_mensaje("No tengo neuronas activas para procesar tu solicitud. Prueba a reiniciar la red.", tipo="warning")
            return "No tengo neuronas activas para tu solicitud. Prueba a reiniciar la red.", False


    async def train_with_feedback(self, user_prompt, response_text, embedding=None, save_memory=True):
        if not self.neuronas:
            self.log_mensaje("No hay neuronas para entrenar.", tipo="warning")
            return

        self.knowledge_manager.add_learned_response(user_prompt, response_text, embedding=embedding, save=save_memory)
        self.log_mensaje(f"Pregunta '{user_prompt}' y respuesta almacenadas en la memoria local (con embedding).", tipo="info")

        current_output, processed_neuron_id = self.get_network_output_for_text(user_prompt)
        if processed_neuron_id is None:
            self.log_mensaje("No hay neuronas disponibles para procesar la entrada y entrenar.", tipo="warning")
            return

        target_output = abs(sum(ord(char) for char in response_text)) % 1000
        error_signal = target_output - current_output

        neuron_to_train = self.neuronas.get(processed_neuron_id)
        if neuron_to_train:
            neuron_to_train.learn(error_signal)
            self.log_mensaje(f"Neurona {processed_neuron_id} ajustada con error {error_signal:.2f}.", tipo="info")
        else:
            self.log_mensaje(f"Neurona {processed_neuron_id} no encontrada para entrenamiento.", tipo="error")

    async def train_from_dataset(self, dataset: list[dict]):
        self.log_mensaje(f"Iniciando entrenamiento en lote con {len(dataset)} ejemplos...", tipo="info")
        for i, item in enumerate(dataset):
            prompt = item.get("prompt")
            response = item.get("response")
            if prompt and response:
                prompt_embedding = await self.get_embedding_from_pytorch(prompt)
                if prompt_embedding is not None:
                    await self.train_with_feedback(prompt, response, embedding=prompt_embedding, save_memory=False)
                    if (i + 1) % 50 == 0:
                        self.log_mensaje(f"  Procesados {i + 1}/{len(dataset)} ejemplos.", tipo="info")
                else:
                    self.log_mensaje(f"  Advertencia: No se pudo generar embedding para el prompt '{prompt}'. Saltando ejemplo.", tipo="warning")
            else:
                self.log_mensaje(f"  Advertencia: Ejemplo inválido en el dataset: {item}", tipo="warning")
        self.log_mensaje("Entrenamiento en lote finalizado.", tipo="info")
        self.knowledge_manager.save_state()

    async def initialize_network_automatically(self):
        self.log_mensaje("Intentando cargar estado de red existente...", tipo="info")
        load_successful = self.knowledge_manager.load_state()

        if not self.knowledge_manager.general_knowledge:
            self.knowledge_manager.load_default_knowledge()

        self.log_mensaje("Verificando y generando embeddings para el conocimiento general...", tipo="info")
        for prompt, response in list(self.knowledge_manager.general_knowledge.items()):
            if prompt not in self.knowledge_manager.general_knowledge_embeddings or self.knowledge_manager.general_knowledge_embeddings[prompt] is None:
                embedding = await self.get_embedding_from_pytorch(prompt)
                if embedding is not None:
                    self.knowledge_manager.add_general_knowledge(prompt, response, embedding=embedding)
                    self.log_mensaje(f"  Embedding generado y almacenado para conocimiento general: '{prompt}'", tipo="info")
                else:
                    self.log_mensaje(f"  Advertencia: No se pudo generar embedding para conocimiento general: '{prompt}'.", tipo="warning")
        
        self.log_mensaje("Verificando y generando embeddings para las respuestas aprendidas...", tipo="info")
        for prompt, response in list(self.knowledge_manager.learned_responses.items()):
            if prompt not in self.knowledge_manager.learned_responses_embeddings or self.knowledge_manager.learned_responses_embeddings[prompt] is None:
                embedding = await self.get_embedding_from_pytorch(prompt)
                if embedding is not None:
                    self.knowledge_manager.add_learned_response(prompt, response, embedding=embedding, save=False)
                    self.log_mensaje(f"  Embedding generado y almacenado para respuesta aprendida: '{prompt}'", tipo="info")
                else:
                    self.log_mensaje(f"  Advertencia: No se pudo generar embedding para respuesta aprendida: '{prompt}'.", tipo="warning")

        self.log_mensaje("Verificando y generando embeddings para palabras clave de auto-descripción...", tipo="info")
        for keyword in list(self.knowledge_manager.self_description_keywords.keys()): 
            if keyword not in self.knowledge_manager.self_description_embeddings or self.knowledge_manager.self_description_embeddings[keyword] is None:
                embedding = await self.get_embedding_from_pytorch(keyword)
                if embedding is not None:
                    self.knowledge_manager.add_self_description_embedding(keyword, embedding)
                    self.log_mensaje(f"  Embedding generado y almacenado para palabra clave de auto-descripción: '{keyword}'", tipo="info")
                else:
                    self.log_mensaje(f"  Advertencia: No se pudo generar embedding para palabra clave de auto-descripción: '{keyword}'.", tipo="warning")
        
        self.knowledge_manager.save_state() 
        self.log_mensaje("Embeddings de conocimiento por defecto, aprendido y auto-descripción verificados/generados y guardados.", tipo="info")


        if not load_successful or not self.neuronas:
            if load_successful and not self.neuronas:
                self.log_mensaje("Red cargada exitosamente, pero está vacía. Iniciando creación automática de neuronas...", tipo="info")
            else:
                self.log_mensaje("No se pudo cargar la red o no existe. Iniciando creación automática de neuronas y cargando conocimiento por defecto...", tipo="info")
            
            available_ram_mb = get_available_ram_mb()
            cpu_cores = get_cpu_core_count()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            max_disk_usage_percent = 90

            ESTIMATED_BYTES_PER_NEURON = 1024
            MIN_NEURONS = 50
            MAX_HARDWARE_BASED_NEURONS = 100000

            neurons_from_ram = int((available_ram_mb * 1024 * 1024 * 0.8) / ESTIMATED_BYTES_PER_NEURON)
            neurons_from_cpu = cpu_cores * 5000

            calculated_max_neurons = min(neurons_from_ram, neurons_from_cpu, MAX_HARDWARE_BASED_NEURONS)
            max_neurons_for_creation = max(MIN_NEURONS, calculated_max_neurons)

            print(f"DEBUG (Server Init): RAM disponible: {available_ram_mb:.2f} MB, núcleos de CPU: {cpu_cores}")
            print(f"DEBUG (Server Init): Neuronas máximas calculadas (RAM): {neurons_from_ram}")
            print(f"DEBUG (Server Init): Neuronas máximas calculadas (CPU): {neurons_from_cpu}")
            print(f"DEBUG (Server Init): Límite final de neuronas para creación: {max_neurons_for_creation}")

            self.log_mensaje(f"Hardware detectado: RAM disponible {available_ram_mb:.2f} MB, {cpu_cores} núcleos de CPU.", tipo="info")
            self.log_mensaje(f"Estableciendo límite de creación de neuronas en {max_neurons_for_creation} basado en hardware.", tipo="info")

            first_neuron_id = "N_inicio"
            if self.añadir_neurona(Neurona(neurona_id=first_neuron_id, peso=0.5)):
                self.log_mensaje(f"Creada neurona inicial: {first_neuron_id}", tipo="info")

            for i in range(1, max_neurons_for_creation + 1):
                disk_percent = get_disk_usage_percentage(current_dir)
                if disk_percent >= max_disk_usage_percent:
                    self.log_mensaje(f"Límite de uso de disco ({max_disk_usage_percent}%) alcanzado. Deteniendo creación de neuronas.", tipo="warning")
                    break

                new_neuron_id = f"N_{i}"
                if self.añadir_neurona(Neurona(neurona_id=new_neuron_id, peso=random.uniform(0.1, 0.9))):
                    existing_neuron_ids = list(self.neuronas.keys())
                    if len(existing_neuron_ids) > 1:
                        num_connections = random.randint(1, min(2, len(existing_neuron_ids) - 1))
                        connections_made = 0
                        while connections_made < num_connections:
                            target_id = random.choice(existing_neuron_ids)
                            if target_id != new_neuron_id:
                                self.establecer_conexion(new_neuron_id, target_id)
                                connections_made += 1
                    elif len(existing_neuron_ids) == 1 and existing_neuron_ids[0] != new_neuron_id:
                        self.establecer_conexion(new_neuron_id, existing_neuron_ids[0])
                else:
                    self.log_mensaje(f"No se pudo añadir la neurona {new_neuron_id}.", tipo="error")
            
            self.knowledge_manager.save_state() 

        self.log_mensaje(f"Creación automática de neuronas finalizada. Neuronas en red: {len(self.neuronas)}", tipo="info")
        self.log_mensaje("La red neuronal está lista para interactuar. ¡Prueba a chatear!", tipo="info")
        self.log_mensaje("Nota: La IA local aprende de las respuestas de Gemini. Gemini genera las respuestas de texto, mientras la red local ajusta sus pesos para asociar patrones numéricos.", tipo="info")

