import shutil # Para verificar el uso del disco
import os # Para obtener la ruta del directorio actual
import psutil # Para la monitorización de recursos del sistema (RAM, CPU)
import unicodedata # Para normalizar caracteres (ej. ñ a n)
import re # Para expresiones regulares para eliminar puntuación

def get_disk_usage_percentage(path='.'):
    """
    Obtiene el porcentaje de uso del disco para la ruta especificada.
    :param path: La ruta del directorio a verificar. Por defecto, el directorio actual.
    :return: El porcentaje de uso del disco (0-100).
    """
    try:
        total, used, free = shutil.disk_usage(path)
        percentage = (used / total) * 100
        return percentage
    except Exception as e:
        print(f"ERROR: No se pudo obtener el uso del disco para '{path}': {e}")
        return 0.0 # Devolver 0.0 en caso de error

def get_available_ram_mb():
    """
    Obtiene la RAM disponible en megabytes.
    Requiere la librería psutil.
    """
    try:
        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception as e:
        print(f"ERROR: No se pudo obtener la RAM disponible: {e}. Asegúrese de que psutil esté instalado.")
        return 0.0 # Devolver 0.0 en caso de error

def get_cpu_core_count():
    """
    Obtiene el número de núcleos lógicos de CPU.
    Requiere la librería psutil.
    """
    try:
        return psutil.cpu_count(logical=True)
    except Exception as e:
        print(f"ERROR: No se pudo obtener el número de núcleos de CPU: {e}. Asegúrese de que psutil esté instalado.")
        return 1 # Valor predeterminado de 1 núcleo si falla

def normalize_text(text: str) -> str:
    """
    Normaliza el texto para mejorar la coincidencia de palabras clave.
    Convierte a minúsculas, elimina puntuación y normaliza caracteres Unicode (ej. ñ a n).
    """
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar acentos y caracteres especiales (como ñ)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Eliminar puntuación
    text = re.sub(r'[^\w\s]', '', text)
    return text
