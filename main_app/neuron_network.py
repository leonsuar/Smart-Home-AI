import uuid
import random
import time
import re
import os # ¡Importar el módulo os aquí!
import httpx # Importar httpx para hacer llamadas al servidor PyTorch
import numpy as np # Necesario para manejar embeddings

# Importar los nuevos módulos
from knowledge_manager import KnowledgeManager
from llm_service import generate_llm_response
from utils import get_available_ram_mb, get_cpu_core_count, get_disk_usage_percentage, normalize_text # Importar normalize_text

# --- Clase Neurona ---
class Neurona:
    def __init__(self, neurona_id=None, peso=0.5):
        """
        Inicializa una nueva neurona.
        :param neurona_id: ID opcional para la neurona. Si no se proporciona, se genera uno único.
        :param peso: El peso inicial de la neurona, que afecta su procesamiento.
        """
        self.id = neurona_id if neurona_id else str(uuid.uuid4())
        self.peso = peso
        self.conexiones_salida = []

    def procesar_entrada(self, entrada):
        """
        Simula el procesamiento de una señal de entrada.
        En este modelo simple, la salida es la entrada multiplicada por el peso.
        :param entrada: El valor numérico de la señal de entrada.
        :return: El valor numérico de la señal de salida.
        """
        return entrada * self.peso

    def copiar(self):
        """
        Crea una nueva neurona que es una copia de la actual.
        La copia puede tener una ligera mutación inicial en su peso.
        :return: Una nueva instancia de Neurona.
        """
        mutacion_inicial = random.uniform(-0.1, 0.1)
        nueva_neurona = Neurona(peso=self.peso + mutacion_inicial)
        return nueva_neurona

    def mutar(self, factor_mutacion=0.1):
        """
        Modifica aleatoriamente el peso de la neurona.
        :param factor_mutacion: El rango máximo de cambio para la mutación.
        :return: El nuevo peso de la neurona.
        """
        old_peso = self.peso
        self.peso += random.uniform(-factor_mutacion, factor_mutacion)
        self.peso = max(0, min(1, self.peso))
        return self.peso

    def conectar_a(self, otra_neurona_id):
        """
        Establece una conexión de salida de esta neurona a otra neurona.
        :param otra_neurona_id: El ID de la neurona a la que se conectará.
        """
        if otra_neurona_id not in self.conexiones_salida:
            self.conexiones_salida.append(otra_neurona_id)

    def learn(self, error_signal, learning_rate=0.01):
        """
        Ajusta el peso de la neurona basado en una señal de error.
        """
        old_peso = self.peso
        self.peso += error_signal * learning_rate
        self.peso = max(0, min(1, self.peso))
        

    def __str__(self):
        """
        Representación de cadena de la neurona para facilitar la depuración.
        """
        return f"Neurona(ID: {self.id}, Peso: {self.peso:.2f}, Conexiones: {len(self.conexiones_salida)})"

    def to_dict(self):
        """
        Convierte la neurona a un diccionario para serialización JSON.
        """
        return {
            "id": self.id,
            "peso": self.peso,
            "conexiones_salida": self.conexiones_salida
        }

    @classmethod
    def from_dict(cls, data):
        """
        Crea una instancia de Neurona a partir de un diccionario.
        """
        neuron = cls(neurona_id=data["id"], peso=data["peso"])
        neuron.conexiones_salida = data["conexiones_salida"]
        return neuron

# --- Clase RedNeuronal ---
class RedNeuronal:
    def __init__(self):
        """
        Inicializa la red neuronal.
        """
        self.neuronas = {}
        self.mensajes = [] # Para la consola de la interfaz web
        self.knowledge_manager = KnowledgeManager() # Instancia del gestor de conocimiento
        # URL del servidor de modelo PyTorch. 'pytorch_model_server' es el nombre del servicio en Docker Compose.
        self.pytorch_text_gen_url = "http://pytorch_model_server:5001/generate_text" 
        # Nuevo: URL del servidor de modelo PyTorch para embeddings
        self.pytorch_embedding_url = "http://pytorch_model_server:5001/get_embedding"

    def añadir_neurona(self, neurona):
        """
        Añade una neurona a la red.
        """
        if neurona.id in self.neuronas:
            self.log_mensaje(f"Error: La neurona {neurona.id} ya existe.", tipo="error")
            return False
        self.neuronas[neurona.id] = neurona
        self.log_mensaje(f"Neurona {neurona.id} añadida a la red.")
        # self.knowledge_manager.save_state() # ¡REMOVIDO! Guardado se maneja en initialize_network_automatically o app.py
        return True

    def obtener_neurona(self, neurona_id):
        """
        Obtiene una neurona de la red por su ID.
        """
        return self.neuronas.get(neurona_id)

    def replicar_neurona(self, neurona_id):
        """
        Replica una neurona existente en la red.
        """
        neurona_original = self.obtener_neurona(neurona_id)
        if neurona_original:
            nueva_neurona = neurona_original.copiar()
            self.añadir_neurona(nueva_neurona) # añadir_neurona ya no guarda
            self.log_mensaje(f"Neurona {neurona_id} replicada a {nueva_neurona.id}.")
            # self.knowledge_manager.save_state() # ¡REMOVIDO! Guardado se maneja en app.py
            return nueva_neurona
        self.log_mensaje(f"Error: No se pudo replicar. Neurona {neurona_id} no encontrada.", tipo="error")
        return None

    def mutar_neurona(self, neurona_id, factor=0.1):
        """
        Muta el peso de una neurona específica en la red.
        """
        neurona = self.obtener_neurona(neurona_id)
        if neurona:
            old_peso = neurona.peso
            neurona.mutar(factor)
            self.log_mensaje(f"Neurona {neurona_id} mutada. Peso: {old_peso:.2f} -> {neurona.peso:.2f}.")
            # self.knowledge_manager.save_state() # ¡REMOVIDO! Guardado se maneja en app.py
            return True
        self.log_mensaje(f"Error: No se pudo mutar. Neurona {neurona_id} no encontrada.", tipo="error")
        return False

    def establecer_conexion(self, id_origen, id_destino):
        """
        Establece una conexión de una neurona origen a una neurona destino.
        """
        neurona_origen = self.obtener_neurona(id_origen)
        neurona_destino = self.obtener_neurona(id_destino)
        if neurona_origen and neurona_destino:
            neurona_origen.conectar_a(id_destino)
            self.log_mensaje(f"Conexión establecida: {id_origen} -> {id_destino}.")
            # self.knowledge_manager.save_state() # ¡REMOVIDO! Guardado se maneja en app.py
            return True
        self.log_mensaje(f"Error: No se pudo establecer la conexión entre {id_origen} y {id_destino}. Asegúrese de que ambas neuronas existan.", tipo="error")
        return False

    def enviar_activacion(self, id_origen, entrada):
        """
        Simula el envío de una señal de activación desde una neurona.
        """
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
        """
        Añade un mensaje a la lista de mensajes de la consola.
        """
        self.mensajes.append({"tiempo": time.strftime("%H:%M:%S"), "mensaje": mensaje, "tipo": tipo})
        if len(self.mensajes) > 100:
            self.mensajes.pop(0)

    def obtener_estado_red(self):
        """
        Devuelve el estado actual de todas las neuronas en la red.
        """
        estado = []
        for n_id, neurona in self.neuronas.items():
            estado.append({
                "id": neurona.id,
                "peso": f"{neurona.peso:.2f}",
                "conexiones": neurona.conexiones_salida
            })
        return estado

    def get_network_output_for_text(self, text_input):
        """
        Convierte el texto a una entrada numérica y la procesa a través de una neurona aleatoria.
        """
        if not self.neuronas:
            return 0.0, None

        numerical_input = abs(sum(ord(char) for char in text_input)) % 1000

        random_neuron_id = random.choice(list(self.neuronas.keys()))
        neuron = self.neuronas[random_neuron_id]
        
        output = neuron.procesar_entrada(numerical_input)
        return output, random_neuron_id

    async def get_embedding_from_pytorch(self, text: str) -> list:
        """
        Obtiene el embedding de un texto utilizando el servidor de modelo PyTorch local.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.pytorch_embedding_url, json={"text": text})
                response.raise_for_status()
                result = response.json()
                if "embedding" in result:
                    return result["embedding"]
                else:
                    self.log_mensaje(f"Error PyTorch Embedding Server: Respuesta inesperada: {result}", tipo="error")
                    return None
        except httpx.RequestError as e:
            self.log_mensaje(f"Error de conexión al servidor PyTorch para embeddings: {e}", tipo="error")
            return None
        except httpx.HTTPStatusError as e:
            self.log_mensaje(f"Error HTTP del servidor PyTorch para embeddings {e.response.status_code}: {e.response.text}", tipo="error")
            return None
        except Exception as e:
            self.log_mensaje(f"Error inesperado al obtener embedding con PyTorch: {e}", tipo="error")
            return None

    async def generate_text_with_pytorch(self, prompt: str) -> str:
        """
        Genera texto utilizando el servidor de modelo PyTorch local.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.pytorch_text_gen_url, json={"prompt": prompt, "max_length": 100})
                response.raise_for_status()
                result = response.json()
                if "generated_text" in result:
                    generated_text = result["generated_text"]
                    if generated_text.startswith(prompt):
                        return generated_text[len(prompt):].strip()
                    return generated_text
                else:
                    self.log_mensaje(f"Error PyTorch Text Gen Server: Respuesta inesperada: {result}", tipo="error")
                    return "Error al generar texto con el modelo local."
        except httpx.RequestError as e:
            self.log_mensaje(f"Error de conexión al servidor PyTorch para generación de texto: {e}", tipo="error")
            return "No pude conectar con el modelo local de PyTorch."
        except httpx.HTTPStatusError as e:
            self.log_mensaje(f"Error HTTP del servidor PyTorch para generación de texto {e.response.status_code}: {e.response.text}", tipo="error")
            return f"El modelo local de PyTorch devolvió un error: {e.response.status_code}"
        except Exception as e:
            self.log_mensaje(f"Error inesperado al generar texto con PyTorch: {e}", tipo="error")
            return "Error inesperado del modelo local de PyTorch."

    async def get_local_or_llm_response(self, text_input):
        """
        Intenta generar una respuesta localmente (exacta, semántica), luego con PyTorch, y finalmente con Gemini.
        Retorna (respuesta_texto, debe_ofrecer_guardar_bandera).
        """
        original_text_input_lower = normalize_text(text_input) 
        normalized_text_input = normalize_text(text_input) 

        self.log_mensaje(f"DEBUG: Entrada original: '{text_input}', Normalizada: '{normalized_text_input}'", tipo="debug")

        # 1. Intentar responder si se pregunta por el nombre del usuario
        if "como me llamo" in normalized_text_input or \
           "cual es mi nombre" in normalized_text_input or \
           "sabes mi nombre" in normalized_text_input:
            self.log_mensaje(f"DEBUG: Matched user name query.", tipo="debug")
            user_name = self.knowledge_manager.get_user_name()
            if user_name:
                self.log_mensaje(f"Red Local: Recordando el nombre para: '{original_text_input_lower}'", tipo="info")
                return f"¡Te llamas {user_name}! Un placer conocerte.", False # No ofrecer guardar
            else:
                self.log_mensaje(f"Red Local: No recuerdo tu nombre. No está en mi memoria. Para que lo recuerde, puedes decirme 'Me llamo [tu nombre]'.", tipo="info")
                return "No recuerdo tu nombre. Para que lo recuerde, puedes decirme 'Me llamo [tu nombre]'.", False # No ofrecer guardar
        
        # 2. Verificar si la pregunta es sobre la propia red neuronal (usando palabras clave externalizadas - coincidencia exacta)
        for kw, response_template in self.knowledge_manager.self_description_keywords.items():
            if kw == normalized_text_input: # Coincidencia EXACTA de palabra clave normalizada
                self.log_mensaje(f"DEBUG: Matched EXACT self-description keyword: '{kw}'", tipo="debug")
                self.log_mensaje(f"Red Local: Respondiendo desde conocimiento predefinido sobre mí para: '{original_text_input_lower}' (palabra clave exacta: '{kw}')", tipo="info")
                if "{ai_name}" in response_template:
                    return response_template.format(ai_name=self.knowledge_manager.get_ai_name()), False # No ofrecer guardar
                return response_template, False # No ofrecer guardar

        # 3. Verificar si la pregunta es sobre la propia red neuronal (usando palabras clave externalizadas - búsqueda SEMÁNTICA)
        self.log_mensaje(f"DEBUG: Generando embedding para búsqueda semántica de palabras clave de auto-descripción: '{text_input}'...", tipo="debug")
        query_embedding_for_identity = await self.get_embedding_from_pytorch(text_input)

        if query_embedding_for_identity is not None:
            similar_identity_responses = self.knowledge_manager.find_similar_response_by_embedding(
                query_embedding_for_identity, 
                self.knowledge_manager.self_description_embeddings, # Buscar en los embeddings de identidad
                self.knowledge_manager.self_description_keywords, # Obtener la respuesta de texto original
                top_k=1, 
                threshold=0.8 # Umbral alto para asegurar relevancia para identidad
            )
            if similar_identity_responses:
                best_identity_response_text, similarity_score = similar_identity_responses[0]
                self.log_mensaje(f"DEBUG: Matched SEMANTIC self-description with similarity {similarity_score:.2f}.", tipo="debug")
                self.log_mensaje(f"Red Local: Respondiendo desde conocimiento predefinido sobre mí (semántico) para: '{original_text_input_lower}' con similitud {similarity_score:.2f}.", tipo="info")
                
                if "{ai_name}" in best_identity_response_text:
                    return best_identity_response_text.format(ai_name=self.knowledge_manager.get_ai_name()), False # No ofrecer guardar
                return best_identity_response_text, False # No ofrecer guardar
        else:
            self.log_mensaje("Advertencia: No se pudo obtener el embedding para la búsqueda semántica de identidad.", tipo="warning")

        # 4. Verificar si la pregunta implica una acción fuera de alcance (usando palabras clave externalizadas)
        # Esto sigue siendo una coincidencia de subcadena, ya que es para detectar *presencia* de un término.
        for oskw in self.knowledge_manager.out_of_scope_keywords:
            if oskw in normalized_text_input:
                self.log_mensaje(f"DEBUG: Matched out-of-scope keyword: '{oskw}'", tipo="debug")
                self.log_mensaje(f"Red Local: Detecté palabras clave relacionadas con una acción que está fuera de mi alcance (como '{oskw}'). No puedo realizar operaciones en redes físicas o cámaras IP. Consultaré a una IA externa para información general.", tipo="info")
                # Las respuestas fuera de alcance son generadas por LLM, por lo que se deben ofrecer para guardar
                llm_response = await generate_llm_response(text_input)
                self.log_mensaje(f"IA (Gemini): {llm_response}", tipo="info")
                return llm_response, True # Ofrecer guardar

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
                threshold=0.75 # Umbral para conocimiento general
            )
            
            if similar_general_responses:
                best_response, similarity_score = similar_general_responses[0]
                self.log_mensaje(f"DEBUG: Matched general semantic knowledge with similarity {similarity_score:.2f}.", tipo="debug")
                self.log_mensaje(f"Respondiendo desde la memoria local (semántica general) para: '{text_input}' con similitud {similarity_score:.2f}.", tipo="info")
                return best_response, False # No ofrecer guardar
            else:
                self.log_mensaje(f"DEBUG: No general semantic match found above threshold.", tipo="debug")
                self.log_mensaje(f"No se encontraron respuestas semánticamente similares en la memoria local para: '{text_input}' (similitud < 0.75).", tipo="info")
        else:
            self.log_mensaje("Advertencia: No se pudo obtener el embedding para la búsqueda semántica general.", tipo="warning")

        # 6. Fallback: Si no hay respuesta local (semántica o exacta), consultar a PyTorch o Gemini
        if len(self.neuronas) > 0:
            # Intentar generar respuesta con PyTorch localmente (si la pregunta es compleja o no se encontró en memoria)
            if len(text_input.split()) > 3 or "que es" in normalized_text_input or "como" in normalized_text_input or "explica" in normalized_text_input:
                self.log_mensaje(f"DEBUG: Attempting PyTorch text generation.", tipo="debug")
                self.log_mensaje(f"Red Local no tiene respuesta directa. Intentando generación con modelo PyTorch para: '{text_input}'...", tipo="info")
                pytorch_response = await self.generate_text_with_pytorch(text_input)
                if pytorch_response and "Error" not in pytorch_response:
                    self.log_mensaje(f"IA (PyTorch Local): {pytorch_response}", tipo="info")
                    return pytorch_response, True # Ofrecer guardar
                else:
                    self.log_mensaje(f"DEBUG: PyTorch generation failed or returned error.", tipo="debug")
                    self.log_mensaje(f"Modelo PyTorch local no pudo generar una respuesta o hubo un error: {pytorch_response}. Recurriendo a Gemini...", tipo="warning")
            
            # Fallback a Gemini si PyTorch no se usó o falló
            self.log_mensaje(f"DEBUG: Falling back to Gemini LLM.", tipo="debug")
            self.log_mensaje(f"Red Local no tiene respuesta directa. Consultando a IA (Gemini) para: '{text_input}'...", tipo="info")
            llm_response = await generate_llm_response(text_input)
            self.log_mensaje(f"IA (Gemini): {llm_response}", tipo="info")
            
            return llm_response, True # Ofrecer guardar
        else:
            self.log_mensaje("No tengo neuronas activas para procesar tu solicitud. Prueba a reiniciar la red.", tipo="warning")
            return "No tengo neuronas activas para tu solicitud. Prueba a reiniciar la red.", False # No ofrecer guardar


    async def train_with_feedback(self, user_prompt, response_text, embedding=None, save_memory=True):
        """
        Entrena la red localmente usando la respuesta como feedback y almacena el embedding.
        :param embedding: El embedding de la pregunta del usuario.
        :param save_memory: Si es True, guarda el estado de la memoria después de añadir la respuesta.
        """
        if not self.neuronas:
            self.log_mensaje("No hay neuronas para entrenar.", tipo="warning")
            return

        # 1. Almacenar la pregunta y respuesta en el conocimiento local, junto con su embedding
        self.knowledge_manager.add_learned_response(user_prompt, response_text, embedding=embedding, save=save_memory)
        self.log_mensaje(f"Pregunta '{user_prompt}' y respuesta almacenadas en la memoria local (con embedding).", tipo="info")

        # 2. Obtener la salida actual de la red para el prompt del usuario
        current_output, processed_neuron_id = self.get_network_output_for_text(user_prompt)
        if processed_neuron_id is None:
            self.log_mensaje("No hay neuronas disponibles para procesar la entrada y entrenar.", tipo="warning")
            return

        # 3. Definir un "objetivo" numérico basado en la respuesta.
        target_output = abs(sum(ord(char) for char in response_text)) % 1000

        # 4. Calcular un "error" simple
        error_signal = target_output - current_output

        # 5. Ajustar la neurona que procesó la entrada
        neuron_to_train = self.neuronas.get(processed_neuron_id)
        if neuron_to_train:
            neuron_to_train.learn(error_signal)
            self.log_mensaje(f"Neurona {processed_neuron_id} ajustada con error {error_signal:.2f}.", tipo="info")
        else:
            self.log_mensaje(f"Neurona {processed_neuron_id} no encontrada para entrenamiento.", tipo="error")

    async def train_from_dataset(self, dataset: list[dict]):
        """
        Entrena la red con un conjunto de datos predefinido (preguntas y respuestas).
        Esto simula un entrenamiento más rápido y en lotes.
        :param dataset: Una lista de diccionarios, donde cada diccionario tiene 'prompt' y 'response'.
        """
        self.log_mensaje(f"Iniciando entrenamiento en lote con {len(dataset)} ejemplos...", tipo="info")
        for i, item in enumerate(dataset):
            prompt = item.get("prompt")
            response = item.get("response")
            if prompt and response:
                # Obtener el embedding para el prompt del dataset
                prompt_embedding = await self.get_embedding_from_pytorch(prompt)
                if prompt_embedding is not None:
                    # Pasar el embedding a train_with_feedback
                    await self.train_with_feedback(prompt, response, embedding=prompt_embedding, save_memory=False)
                    if (i + 1) % 50 == 0: # Loguear el progreso cada 50 ejemplos
                        self.log_mensaje(f"  Procesados {i + 1}/{len(dataset)} ejemplos.", tipo="info")
                else:
                    self.log_mensaje(f"  Advertencia: No se pudo generar embedding para el prompt '{prompt}'. Saltando ejemplo.", tipo="warning")
            else:
                self.log_mensaje(f"  Advertencia: Ejemplo inválido en el dataset: {item}", tipo="warning")
        self.log_mensaje("Entrenamiento en lote finalizado.", tipo="info")
        self.knowledge_manager.save_state() # Guardar el estado UNA VEZ al final del entrenamiento en lote

    async def initialize_network_automatically(self): # Ahora es async def
        """
        Función para inicializar automáticamente la red neuronal al inicio.
        Crea neuronas y las conecta, respetando un límite de uso de disco y recursos de hardware.
        """
        self.log_mensaje("Intentando cargar estado de red existente...", tipo="info")
        load_successful = self.knowledge_manager.load_state()

        # Cargar el conocimiento por defecto desde el archivo si no se cargó o si está vacío
        # y generar embeddings para él si no existen
        if not self.knowledge_manager.general_knowledge: # Solo cargar si está vacío
            self.knowledge_manager.load_default_knowledge()

        # Asegurar que todas las entradas de conocimiento general tengan embeddings
        self.log_mensaje("Verificando y generando embeddings para el conocimiento general...", tipo="info")
        for prompt, response in list(self.knowledge_manager.general_knowledge.items()): # Usar list() para evitar RuntimeError al modificar durante iteración
            if prompt not in self.knowledge_manager.general_knowledge_embeddings or self.knowledge_manager.general_knowledge_embeddings[prompt] is None:
                embedding = await self.get_embedding_from_pytorch(prompt)
                if embedding is not None:
                    self.knowledge_manager.add_general_knowledge(prompt, response, embedding=embedding)
                    self.log_mensaje(f"  Embedding generado y almacenado para conocimiento general: '{prompt}'", tipo="info")
                else:
                    self.log_mensaje(f"  Advertencia: No se pudo generar embedding para conocimiento general: '{prompt}'.", tipo="warning")
        
        # Asegurar que todas las respuestas aprendidas tengan embeddings
        self.log_mensaje("Verificando y generando embeddings para las respuestas aprendidas...", tipo="info")
        for prompt, response in list(self.knowledge_manager.learned_responses.items()):
            if prompt not in self.knowledge_manager.learned_responses_embeddings or self.knowledge_manager.learned_responses_embeddings[prompt] is None:
                embedding = await self.get_embedding_from_pytorch(prompt)
                if embedding is not None:
                    self.knowledge_manager.add_learned_response(prompt, response, embedding=embedding, save=False) # No guardar en cada iteración
                    self.log_mensaje(f"  Embedding generado y almacenado para respuesta aprendida: '{prompt}'", tipo="info")
                else:
                    self.log_mensaje(f"  Advertencia: No se pudo generar embedding para respuesta aprendida: '{prompt}'.", tipo="warning")

        # NUEVO: Asegurar que las palabras clave de auto-descripción tengan embeddings
        self.log_mensaje("Verificando y generando embeddings para palabras clave de auto-descripción...", tipo="info")
        for keyword in list(self.knowledge_manager.self_description_keywords.keys()):
            if keyword not in self.knowledge_manager.self_description_embeddings or self.knowledge_manager.self_description_embeddings[keyword] is None:
                embedding = await self.get_embedding_from_pytorch(keyword)
                if embedding is not None:
                    self.knowledge_manager.add_self_description_embedding(keyword, embedding)
                    self.log_mensaje(f"  Embedding generado y almacenado para palabra clave de auto-descripción: '{keyword}'", tipo="info")
                else:
                    self.log_mensaje(f"  Advertencia: No se pudo generar embedding para palabra clave de auto-descripción: '{keyword}'.", tipo="warning")
        
        # Guardar el estado después de generar todos los embeddings si hubo cambios
        self.knowledge_manager.save_state() 
        self.log_mensaje("Embeddings de conocimiento por defecto, aprendido y auto-descripción verificados/generados y guardados.", tipo="info")


        # Si la carga fue exitosa pero la red está vacía, o si la carga falló, proceder con la creación automática.
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
            
            # Guardar el estado de la red una vez, después de que todas las neuronas se hayan añadido
            self.knowledge_manager.save_state() 

        self.log_mensaje(f"Creación automática de neuronas finalizada. Neuronas en red: {len(self.neuronas)}", tipo="info")
        self.log_mensaje("La red neuronal está lista para interactuar. ¡Prueba a chatear!", tipo="info")
        self.log_mensaje("Nota: La IA local aprende de las respuestas de Gemini. Gemini genera las respuestas de texto, mientras la red local ajusta sus pesos para asociar patrones numéricos.", tipo="info")

