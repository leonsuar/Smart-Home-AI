from flask import Flask, render_template, request, jsonify
import re
import os
import json # Importar json para cargar el archivo de entrenamiento
import asyncio # Importar asyncio para manejar operaciones asíncronas
import time # Importar time para time.strftime

# Importar los módulos refactorizados
from neuron_network import RedNeuronal
from utils import normalize_text # Importar normalize_text desde utils


# --- Configuración de Flask ---
app = Flask(__name__)
red_neuronal_global = None # Se inicializará de forma asíncrona

@app.route('/')
def index():
    """
    Ruta principal que renderiza la plantilla HTML de la interfaz.
    """
    return render_template('index.html')

@app.route('/enviar_comando', methods=['POST'])
async def enviar_comando():
    """
    Endpoint para recibir comandos de la interfaz web y ejecutarlos en la red neuronal.
    """
    comando = request.json.get('comando', '').strip()
    print(f"DEBUG: Comando recibido en el servidor: '{comando}'")

    if red_neuronal_global is None:
        # Usar time.strftime para obtener la hora actual
        return jsonify(log=[{"tiempo": time.strftime("%H:%M:%S"), "mensaje": "La red neuronal aún se está inicializando. Por favor, espera un momento.", "tipo": "warning"}], estado_red=[]), 503

    red_neuronal_global.log_mensaje(f"> {comando}", tipo="comando")

    respuesta = "Comando desconocido. Usa 'ayuda' para ver los comandos."

    # Normalizar el comando para las comparaciones de regex
    normalized_comando = normalize_text(comando)
    print(f"DEBUG: Comando normalizado: '{normalized_comando}'")

    partes = comando.split()
    accion = partes[0].lower() if partes else ""
    print(f"DEBUG: Acción extraída: '{accion}')")

    # Intentar reconocer el nombre del usuario (usando el comando normalizado)
    match_name = re.match(r"(me llamo|mi nombre es|yo soy|soy)\s*(.+)", normalized_comando)
    if match_name:
        name = match_name.group(2).strip()
        red_neuronal_global.knowledge_manager.set_user_name(name)
        respuesta = f"¡Hola, {name}! Un placer conocerte. He recordado tu nombre."
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
        return jsonify(log=red_neuronal_global.mensajes, estado_red=red_neuronal_global.obtener_estado_red())

    # Intentar reconocer el nombre de la IA (usando el comando normalizado)
    match_ai_name = re.match(r"(prefiero que te llames|llamarte|te llamaras)\s*[:]?\s*(.+)", normalized_comando)
    if match_ai_name:
        ai_new_name = match_ai_name.group(2).strip()
        red_neuronal_global.knowledge_manager.set_ai_name(ai_new_name)
        respuesta = f"¡Entendido! Me llamaré {ai_new_name} de ahora en adelante. ¡Gracias por el nombre!"
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
        return jsonify(log=red_neuronal_global.mensajes, estado_red=red_neuronal_global.obtener_estado_red())


    if accion == "crear_neurona" and len(partes) > 1:
        neurona_id = partes[1]
        try:
            peso = float(partes[2]) if len(partes) > 2 else 0.5
            from neuron_network import Neurona # Importar Neurona aquí si es necesario
            nueva_neurona = Neurona(neurona_id=neurona_id, peso=peso)
            if red_neuronal_global.añadir_neurona(nueva_neurona):
                respuesta = f"Neurona '{neurona_id}' creada con peso {peso:.2f}."
            else:
                respuesta = red_neuronal_global.mensajes[-1]['mensaje']
        except ValueError:
            respuesta = "Error: El peso debe ser un número válido."
        except Exception as e:
            respuesta = f"Error inesperado al crear neurona: {e}"
            print(f"ERROR: {e}")
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
    elif accion == "replicar" and len(partes) > 1:
        neurona_id = partes[1]
        nueva = red_neuronal_global.replicar_neurona(neurona_id)
        respuesta = red_neuronal_global.mensajes[-1]['mensaje'] if nueva else red_neuronal_global.mensajes[-1]['mensaje']
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
    elif accion == "conectar" and len(partes) > 2:
        id_origen, id_destino = partes[1], partes[2]
        if red_neuronal_global.establecer_conexion(id_origen, id_destino):
            respuesta = f"Conexión establecida: '{id_origen}' -> '{id_destino}'."
        else:
            respuesta = red_neuronal_global.mensajes[-1]['mensaje']
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
    elif accion == "activar" and len(partes) > 2:
        id_neurona = partes[1]
        try:
            entrada = float(partes[2])
            red_neuronal_global.enviar_activacion(id_neurona, entrada)
            respuesta = f"Activación enviada a '{id_neurona}' con entrada {entrada}."
        except ValueError:
            respuesta = "Error: La entrada para activar debe ser un número válido."
        except Exception as e:
            respuesta = f"Error al activar: {e}"
            print(f"ERROR: {e}")
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
    elif accion == "mutar" and len(partes) > 1:
        neurona_id = partes[1]
        try:
            factor = float(partes[2]) if len(partes) > 2 else 0.1
            if red_neuronal_global.mutar_neurona(neurona_id, factor):
                respuesta = f"Neurona '{neurona_id}' mutada."
            else:
                respuesta = red_neuronal_global.mensajes[-1]['mensaje']
        except ValueError:
            respuesta = "Error: El factor de mutación debe ser un número válido."
        except Exception as e:
            respuesta = f"Error inesperado al mutar neurona: {e}"
            print(f"ERROR: {e}")
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
    elif accion == "listar_neuronas":
        neuronas_info = red_neuronal_global.obtener_estado_red()
        if neuronas_info:
            respuesta = "Neuronas en la red:\n"
            for n_data in neuronas_info:
                respuesta += f"- ID: {n_data['id']}, Peso: {n_data['peso']}, Conexiones: {', '.join(n_data['conexiones']) if n_data['conexiones'] else 'Ninguna'}\n"
        else:
            respuesta = "No hay neuronas en la red."
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
    elif accion == "limpiar":
        red_neuronal_global.neuronas = {}
        red_neuronal_global.mensajes = []
        red_neuronal_global.knowledge_manager.clear_all_memory() # Limpiar toda la memoria
        respuesta = "Red neuronal y consola limpiadas."
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
    elif accion == "ayuda":
        respuesta = """
Comandos disponibles:
- **crear_neurona [ID] [peso]**: Crea una nueva neurona. Ej: `crear_neurona N1 0.7`
- **replicar [ID_original]**: Crea una copia de una neurona existente. Ej: `replicar N1`
- **conectar [ID_origen] [ID_destino]**: Conecta dos neuronas. Ej: `conectar N1 N2`
- **activar [ID_neurona] [entrada]**: Envía una señal a una neurona. Ej: `activar N1 100`
- **mutar [ID_neurona] [factor]**: Mutar el peso de una neurona. Ej: `mutar N1 0.2`
- **listar_neuronas**: Muestra todas las neuronas y sus conexiones.
- **guardar_red**: Guarda el estado actual de la red en un archivo.
- **limpiar**: Limpia la red y la consola.
- **reiniciar_red**: Limpia la red y la vuelve a crear automáticamente (con conocimiento básico).
- **calcular [operación] [número1] [número2]**: Realiza una operación aritmética. Ej: `calcular suma 5 3`, `calcular dividir 10 2`
- **entrenar_lote**: Entrena la red con un conjunto de datos predefinido de preguntas y respuestas.
- **ayuda**: Muestra esta ayuda.
---
**Nota sobre IA:** La red neuronal intentará responder localmente (usando su memoria y conocimiento básico). Si no 'sabe' la respuesta, preguntará a Gemini y aprenderá de ella.
        """
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
    elif accion == "guardar_red":
        red_neuronal_global.knowledge_manager.save_state() # Guardar el estado de la memoria
        respuesta = "Estado de la red guardado manualmente."
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
    elif accion == "reiniciar_red":
        red_neuronal_global.neuronas = {}
        red_neuronal_global.mensajes = []
        red_neuronal_global.knowledge_manager.clear_all_memory() # Limpiar toda la memoria
        await red_neuronal_global.initialize_network_automatically() # Llamar al método asíncrono de inicialización
        respuesta = "Red neuronal reiniciada y creada automáticamente."
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
    elif accion == "calcular" and len(partes) == 4:
        operacion = partes[1].lower()
        try:
            num1 = float(partes[2])
            num2 = float(partes[3])
            resultado = None
            if operacion == "suma":
                resultado = num1 + num2
            elif operacion == "resta":
                resultado = num1 - num2
            elif operacion == "multiplicar":
                resultado = num1 * num2
            elif operacion == "dividir":
                if num2 != 0:
                    resultado = num1 / num2
                else:
                    respuesta = "Error: División por cero no permitida."
            
            if resultado is not None:
                respuesta = f"La red neuronal calculó '{operacion} {num1} {num2}': {resultado:.2f}"
            
        except ValueError:
            respuesta = "Error: Los números para calcular deben ser válidos."
        except Exception as e:
            respuesta = f"Error al realizar la operación: {e}"
            print(f"ERROR: {e}")
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
    elif accion == "entrenar_lote": # Nuevo comando para entrenamiento en lote
        training_file = "training_data.json"
        if not os.path.exists(training_file):
            respuesta = f"Error: El archivo de entrenamiento '{training_file}' no se encontró en el directorio de la aplicación."
            red_neuronal_global.log_mensaje(respuesta, tipo="error")
        else:
            try:
                with open(training_file, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                await red_neuronal_global.train_from_dataset(training_data) # Llamar al nuevo método de entrenamiento (ahora es async)
                respuesta = f"Entrenamiento en lote completado con {len(training_data)} ejemplos. La red ha aprendido nuevas respuestas."
                red_neuronal_global.log_mensaje(respuesta, tipo="info")
            except json.JSONDecodeError:
                respuesta = f"Error: El archivo '{training_file}' no es un JSON válido. Asegúrate de que esté bien formateado."
                red_neuronal_global.log_mensaje(respuesta, tipo="error")
            except Exception as e:
                respuesta = f"Error al cargar o entrenar con el lote de datos: {e}"
                red_neuronal_global.log_mensaje(respuesta, tipo="error")
    else:
        # La lógica de respuesta local o LLM ahora está en RedNeuronal
        respuesta_contenido = await red_neuronal_global.get_local_or_llm_response(comando)
        respuesta = respuesta_contenido # La función ya devuelve el prefijo "Red Local:" o "IA (Gemini):"
        red_neuronal_global.log_mensaje(respuesta, tipo="info")
        
    return jsonify(log=red_neuronal_global.mensajes, estado_red=red_neuronal_global.obtener_estado_red())

@app.route('/obtener_log')
def obtener_log():
    """
    Endpoint para que la interfaz web obtenga los mensajes de la consola y el estado de la red periódicamente.
    """
    print("DEBUG: Solicitud recibida en /obtener_log")
    if red_neuronal_global is None:
        # Usar time.strftime para obtener la hora actual
        return jsonify(log=[{"tiempo": time.strftime("%H:%M:%S"), "mensaje": "La red neuronal aún se está inicializando. Por favor, espera un momento.", "tipo": "warning"}], estado_red=[]), 503
    return jsonify(log=red_neuronal_global.mensajes, estado_red=red_neuronal_global.obtener_estado_red())

# Función wrapper asíncrona para inicializar la red neuronal al inicio
async def initialize_network_automatically_wrapper():
    global red_neuronal_global
    red_neuronal_global = RedNeuronal() # Crear la instancia de RedNeuronal
    await red_neuronal_global.initialize_network_automatically() # Llamar al método asíncrono

if __name__ == '__main__':
    # Ejecutar la inicialización automática de la red antes de iniciar el servidor Flask
    # Usamos asyncio.run para ejecutar la función asíncrona
    asyncio.run(initialize_network_automatically_wrapper())

    # Ejecutar la aplicación Flask.
    # host='0.0.0.0' hace que el servidor sea accesible desde cualquier IP (útil en entornos de contenedores).
    # port=5000 es el puerto estándar de Flask.
    # debug=True permite recarga automática y depuración más fácil (desactivar en producción).
    app.run(debug=True, host='0.0.0.0', port=5000)
