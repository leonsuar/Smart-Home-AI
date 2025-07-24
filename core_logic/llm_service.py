import httpx # Para realizar solicitudes HTTP asíncronas
import json # Para manejar la serialización/deserialización de JSON
import os # Para obtener variables de entorno
import logging # Para logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def generate_llm_response(prompt: str) -> str:
    """
    Genera una respuesta utilizando el modelo de lenguaje grande (LLM) de Gemini.
    La clave de API se obtiene de una variable de entorno.
    :param prompt: El texto de la pregunta del usuario.
    :return: La respuesta generada por el LLM.
    """
    # Obtener la clave de API de una variable de entorno por seguridad
    api_key = os.getenv("GEMINI_API_KEY", "") 
    if not api_key:
        logging.error("ERROR LLM: La clave de API de Gemini no está configurada en las variables de entorno.")
        return "Lo siento, no puedo conectar con el servicio de IA porque la clave de API no está configurada."

    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    payload = {"contents": chat_history}
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status() # Lanza una excepción para códigos de estado HTTP 4xx/5xx
            result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return text
        else:
            logging.error(f"ERROR LLM: Estructura de respuesta inesperada: {result}")
            return "Lo siento, no pude generar una respuesta en este momento (estructura de respuesta inesperada)."
    except httpx.RequestError as e:
        logging.error(f"ERROR LLM: Error de red o solicitud: {e}")
        return f"Lo siento, no pude conectar con el servicio de IA: {e}"
    except httpx.HTTPStatusError as e:
        logging.error(f"ERROR LLM: Error de estado HTTP {e.response.status_code}: {e.response.text}")
        return f"Lo siento, el servicio de IA devolvió un error: {e.response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(f"ERROR LLM: Error al decodificar JSON: {e}")
        return "Lo siento, hubo un problema al procesar la respuesta del servicio de IA."
    except Exception as e:
        logging.error(f"ERROR LLM: Error inesperado al generar respuesta LLM: {e}")
        return f"Lo siento, ocurrió un error inesperado al procesar tu solicitud."

