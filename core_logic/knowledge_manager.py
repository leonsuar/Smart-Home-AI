import json
import os
import numpy as np # Necesario para operaciones con embeddings

class KnowledgeManager:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.knowledge_file = os.path.join(base_dir, "knowledge.json")
        self.network_state_file = os.path.join(base_dir, "network_state.json")
        self.default_knowledge_file = os.path.join(base_dir, "default_knowledge.json")
        self.keywords_file = os.path.join(base_dir, "keywords.json")
        
        self.general_knowledge = {} # {pregunta: respuesta}
        self.general_knowledge_embeddings = {} # {pregunta: embedding_vector}

        self.learned_responses = {} # {pregunta: respuesta}
        self.learned_responses_embeddings = {} # {pregunta: embedding_vector}

        self.ai_name = "Neo" # Nombre por defecto de la IA
        self.user_name = None # Nombre del usuario

        self.self_description_keywords = {} # Palabras clave de auto-descripción (texto)
        self.self_description_embeddings = {} # NUEVO: Embeddings para palabras clave de auto-descripción
        self.out_of_scope_keywords = [] # Palabras clave fuera de alcance
        
        # Cargar conocimiento por defecto, palabras clave y estado inicial al inicio
        self.load_default_knowledge()
        self.load_keywords_from_file() # Cargar las palabras clave (texto)
        self.load_state() # Cargar el estado de la red y embeddings (incluyendo los nuevos de self_description)

    def add_general_knowledge(self, prompt, response, embedding=None):
        """
        Añade conocimiento general a la memoria.
        :param embedding: El vector de embedding de la pregunta.
        """
        self.general_knowledge[prompt] = response
        if embedding is not None:
            self.general_knowledge_embeddings[prompt] = embedding
        self.save_state()

    def add_learned_response(self, prompt, response, embedding=None, save=True):
        """
        Añade una respuesta aprendida (de Gemini o PyTorch local) a la memoria.
        :param embedding: El vector de embedding de la pregunta.
        :param save: Si es True, guarda el estado inmediatamente.
        """
        self.learned_responses[prompt] = response
        if embedding is not None:
            self.learned_responses_embeddings[prompt] = embedding
        if save:
            self.save_state()

    def add_self_description_embedding(self, keyword, embedding):
        """
        Añade un embedding para una palabra clave de auto-descripción.
        """
        self.self_description_embeddings[keyword] = embedding
        self.save_state() # Guardar el estado después de añadir un embedding de palabra clave

    def get_response_from_memory(self, prompt):
        """
        Busca una respuesta exacta en el conocimiento general y luego en las respuestas aprendidas.
        """
        if prompt in self.general_knowledge:
            return self.general_knowledge[prompt]
        if prompt in self.learned_responses:
            return self.learned_responses[prompt]
        return None

    def set_ai_name(self, name):
        self.ai_name = name
        self.save_state()

    def get_ai_name(self):
        return self.ai_name

    def set_user_name(self, name):
        self.user_name = name
        self.save_state()

    def get_user_name(self):
        return self.user_name

    def load_default_knowledge(self):
        """
        Carga el conocimiento por defecto desde un archivo JSON.
        Este conocimiento no tendrá embeddings inicialmente, se generarán bajo demanda.
        """
        if os.path.exists(self.default_knowledge_file):
            try:
                with open(self.default_knowledge_file, 'r', encoding='utf-8') as f:
                    default_data = json.load(f)
                    for item in default_data.get("general_knowledge", []):
                        prompt = item.get("prompt")
                        response = item.get("response")
                        if prompt and response and prompt not in self.general_knowledge:
                            self.general_knowledge[prompt] = response
                print(f"INFO: Conocimiento por defecto cargado desde '{self.default_knowledge_file}'.")
            except Exception as e:
                print(f"ERROR: No se pudo cargar el conocimiento por defecto desde '{self.default_knowledge_file}': {e}")
        else:
            print(f"INFO: No se encontró el archivo de conocimiento por defecto '{self.default_knowledge_file}'.")

    def load_keywords_from_file(self):
        """
        Carga las palabras clave de auto-descripción y fuera de alcance desde un archivo JSON.
        """
        if os.path.exists(self.keywords_file):
            try:
                with open(self.keywords_file, 'r', encoding='utf-8') as f:
                    keywords_data = json.load(f)
                    self.self_description_keywords = keywords_data.get("self_description_keywords", {})
                    self.out_of_scope_keywords = keywords_data.get("out_of_scope_keywords", [])
                print(f"INFO: Palabras clave cargadas desde '{self.keywords_file}'.")
            except Exception as e:
                print(f"ERROR: No se pudieron cargar las palabras clave desde '{self.keywords_file}': {e}")
        else:
            print(f"INFO: No se encontró el archivo de palabras clave '{self.keywords_file}'. Se usarán valores por defecto o vacíos.")
            # Si el archivo no existe, inicializar con los valores por defecto que estaban en neuron_network.py
            self.self_description_keywords = {
                "neurona": "Una neurona es la unidad básica de mi red. Recibe, procesa y transmite señales.",
                "red neuronal": "Soy una red neuronal, un sistema de neuronas interconectadas que trabajan juntas. Mi propósito es simular el procesamiento de información.",
                "peso": "El peso de una neurona es su factor de influencia. Se ajusta para que mi red 'aprende'.",
                "mutar": "Mutar es cambiar el peso de una neurona aleatoriamente, lo que ayuda a la red a explorar nuevas configuraciones.",
                "aprender": "Mi aprendizaje implica ajustar los pesos de mis neuronas para mejorar mis respuestas, a menudo con la ayuda de una IA externa.",
                "conectar": "Las neuronas se conectan para formar caminos por donde fluye la información en la red.",
                "calcular": "Sí, puedo realizar operaciones aritméticas básicas como suma, resta, multiplicación y división. Usa el comando 'calcular'.",
                "hola": "¡Hola! ¿Cómo puedo asistirte hoy?",
                "como estas": "Estoy funcionando perfectamente, ¡gracias por preguntar! ¿En qué puedo ayudarte hoy?",
                "funcionas": "Mi funcionamiento se basa en el procesamiento de señales a través de mis neuronas interconectadas.",
                "que eres": "Soy una red neuronal simplificada. Puedo crear, mutar y conectar neuronas, y también realizar cálculos básicos.",
                "como funcionas": "Proceso información a través de neuronas interconectadas. Cada neurona tiene un peso que influye en su salida. Intento responderte localmente y, si no sé, consulto a una IA externa.",
                "cual es tu proposito": "Mi propósito es simular el comportamiento básico de una red neuronal y demostrar cómo puede procesar información y 'aprende' de interacciones.",
                "cuantas neuronas tienes": "La cantidad de neuronas que tengo se ajusta automáticamente al iniciar, basándose en los recursos de hardware de tu sistema (RAM y CPU).",
                "donde estas": "Soy un programa de software, existo dentro de los sistemas informáticos que me ejecutan. No tengo una ubicación física en el mundo real.",
                "donde vives": "Soy un programa de software, existo dentro de los sistemas informáticos que me ejecutan. No tengo una ubicación física en el mundo real.",
                "sabes donde vivo": "Soy un programa de software, existo dentro de los sistemas informáticos que me ejecutan. No tengo una ubicación física en el mundo real.",
                "puedes verme": "Como programa de software, no tengo la capacidad de 'ver' en el sentido humano. Existo en el ámbito digital.",
                "eres de aca": "Soy un programa de software. No tengo un lugar de origen físico, existo donde me ejecutes.",
                "cuantos años tienes": "No tengo edad en el sentido humano, ya que soy un programa de software. Mi 'existencia' comenzó cuando fui creado.",
                "que edad tienes": "No tengo edad en el sentido humano, ya que soy un programa de software. Mi 'existencia' comenzó cuando fui creado.",
                "como te llamas": "Mi nombre es {ai_name}. ¡Un placer!"
            }
            self.out_of_scope_keywords = [
                "cámara ip", "camara ip", "red de computadoras", "revisar red", "tomar imagen", "proporcionar datos", "autentiques", "red local", "computadoras", "escanear", "acceder", "conectar a internet"
            ]


    def save_state(self):
        """
        Guarda el estado actual de la red y el conocimiento en un archivo JSON.
        Los embeddings se guardan como listas de Python.
        """
        state_data = {
            "general_knowledge": self.general_knowledge,
            "general_knowledge_embeddings": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.general_knowledge_embeddings.items()},
            "learned_responses": self.learned_responses,
            "learned_responses_embeddings": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.learned_responses_embeddings.items()},
            "self_description_embeddings": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.self_description_embeddings.items()}, # NUEVO
            "ai_name": self.ai_name,
            "user_name": self.user_name
        }
        try:
            with open(self.network_state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=4)
            print(f"INFO: Estado de la memoria guardado en '{self.network_state_file}'.")
        except Exception as e:
            print(f"ERROR: No se pudo guardar el estado de la memoria en '{self.network_state_file}': {e}")

    def load_state(self):
        """
        Carga el estado de la red y el conocimiento desde un archivo JSON.
        Los embeddings se cargan y se convierten de nuevo a arrays de NumPy.
        """
        if os.path.exists(self.network_state_file):
            try:
                with open(self.network_state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    self.general_knowledge = state_data.get("general_knowledge", {})
                    self.learned_responses = state_data.get("learned_responses", {})
                    self.ai_name = state_data.get("ai_name", "Neo") 
                    self.user_name = state_data.get("user_name")

                    # Cargar embeddings y convertirlos a NumPy arrays
                    self.general_knowledge_embeddings = {
                        k: np.array(v) for k, v in state_data.get("general_knowledge_embeddings", {}).items()
                    }
                    self.learned_responses_embeddings = {
                        k: np.array(v) for k, v in state_data.get("learned_responses_embeddings", {}).items()
                    }
                    self.self_description_embeddings = { # NUEVO
                        k: np.array(v) for k, v in state_data.get("self_description_embeddings", {}).items()
                    }
                print(f"INFO: Estado de la memoria cargado desde '{self.network_state_file}'.")
                print(f"INFO:   Conocimiento general: {len(self.general_knowledge)} entradas ({len(self.general_knowledge_embeddings)} con embeddings).")
                print(f"INFO:   Respuestas aprendidas: {len(self.learned_responses)} entradas ({len(self.learned_responses_embeddings)} con embeddings).")
                print(f"INFO:   Palabras clave de auto-descripción: {len(self.self_description_keywords)} entradas ({len(self.self_description_embeddings)} con embeddings).") # NUEVO
                print(f"INFO:   Nombre de la IA: {self.ai_name}.")
                return True
            except json.JSONDecodeError as e:
                print(f"ERROR: Error al decodificar JSON en '{self.network_state_file}': {e}. El archivo podría estar corrupto.")
                os.remove(self.network_state_file)
                print(f"INFO: Archivo '{self.network_state_file}' corrupto eliminado. Se creará uno nuevo.")
                return False
            except Exception as e:
                print(f"ERROR: No se pudo cargar el estado de la memoria desde '{self.network_state_file}': {e}")
                return False
        else:
            print(f"INFO: No se encontró el archivo de estado de red '{self.network_state_file}'. Se creará uno nuevo al guardar.")
            return False

    def find_similar_response_by_embedding(self, query_embedding, target_embeddings_dict, target_text_dict, top_k=1, threshold=0.7):
        """
        Busca las respuestas más similares en una colección de embeddings dada.
        :param query_embedding: El embedding de la pregunta del usuario.
        :param target_embeddings_dict: El diccionario de embeddings donde buscar (ej: self_description_embeddings, general_knowledge_embeddings).
        :param target_text_dict: El diccionario de texto correspondiente a los embeddings (ej: self_description_keywords, general_knowledge).
        :param top_k: Número de resultados más similares a devolver.
        :param threshold: Umbral de similitud del coseno (0.0 a 1.0).
        :return: Una lista de tuplas (respuesta, similitud), ordenadas por similitud descendente.
        """
        if query_embedding is None or not target_embeddings_dict:
            return []

        all_embeddings = []
        all_responses = []
        
        # Asegurarse de que los embeddings sean arrays de numpy y recopilar respuestas
        for prompt, emb in target_embeddings_dict.items():
            all_embeddings.append(emb)
            all_responses.append(target_text_dict.get(prompt, "Respuesta no encontrada")) # Obtener la respuesta de texto

        if not all_embeddings:
            return []

        query_embedding = np.array(query_embedding)
        all_embeddings = np.array(all_embeddings)

        # Calcular similitud del coseno
        dot_product = np.dot(all_embeddings, query_embedding)
        norm_embeddings = np.linalg.norm(all_embeddings, axis=1)
        norm_query = np.linalg.norm(query_embedding)

        if norm_query == 0:
            return []
        
        similarities = dot_product / (norm_embeddings * norm_query)
        
        results = []
        for i, sim in enumerate(similarities):
            if sim >= threshold:
                results.append((all_responses[i], sim))
        
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear_all_memory(self):
        """
        Limpia todo el conocimiento almacenado (general y aprendido) y sus embeddings.
        También restablece el nombre de la IA y del usuario.
        """
        self.general_knowledge = {}
        self.general_knowledge_embeddings = {}
        self.learned_responses = {}
        self.learned_responses_embeddings = {}
        self.self_description_embeddings = {} # NUEVO: Limpiar embeddings de auto-descripción
        self.ai_name = "Neo" # Restablecer a "Neo" al limpiar
        self.user_name = None
        self.load_keywords_from_file() # Recargar las palabras clave desde el archivo (texto)
        self.save_state() # Guardar el estado después de limpiar

