import httpx
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMService:
    def __init__(self):
        # Obtener la clave de API de Gemini desde las variables de entorno
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logging.error("La clave de API de Gemini no está configurada en las variables de entorno (GEMINI_API_KEY).")
            raise ValueError("GEMINI_API_KEY no configurada.")
        
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        self.headers = {'Content-Type': 'application/json'}

    async def generate_text(self, prompt: str) -> str:
        """
        Genera texto utilizando el modelo Gemini 2.0 Flash.
        """
        logging.info(f"Enviando prompt a Gemini: '{prompt[:100]}...'")
        chat_history = []
        chat_history.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": chat_history,
            "generationConfig": {
                "responseMimeType": "text/plain" # Aseguramos que la respuesta sea texto plano
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.api_url, headers=self.headers, json=payload, timeout=30.0)
                response.raise_for_status() # Lanza una excepción para códigos de estado HTTP 4xx/5xx
                
                result = response.json()
                
                if result.get("candidates") and len(result["candidates"]) > 0 and \
                   result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
                   len(result["candidates"][0]["content"]["parts"]) > 0:
                    text = result["candidates"][0]["content"]["parts"][0].get("text", "")
                    logging.info(f"Respuesta de Gemini recibida: '{text[:100]}...'")
                    return text
                else:
                    logging.warning(f"Respuesta inesperada de Gemini: {result}")
                    return "No pude generar una respuesta. La estructura de la respuesta de Gemini es inesperada."
        except httpx.RequestError as e:
            logging.error(f"Error de red o de solicitud al llamar a la API de Gemini: {e}")
            return f"Error de conexión con la IA: {e}"
        except httpx.HTTPStatusError as e:
            logging.error(f"Error de estado HTTP de la API de Gemini: {e.response.status_code} - {e.response.text}")
            return f"Error de la IA (HTTP {e.response.status_code}): {e.response.text}"
        except json.JSONDecodeError as e:
            logging.error(f"Error al decodificar la respuesta JSON de Gemini: {e} - Respuesta: {response.text}")
            return f"Error al procesar la respuesta de la IA: {e}"
        except Exception as e:
            logging.error(f"Ocurrió un error inesperado al llamar a la API de Gemini: {e}")
            return f"Error inesperado al comunicarse con la IA: {e}"

    async def generate_structured_response(self, prompt: str, schema: dict) -> dict:
        """
        Genera una respuesta estructurada (JSON) utilizando el modelo Gemini 2.0 Flash.
        """
        logging.info(f"Enviando prompt estructurado a Gemini: '{prompt[:100]}...' con esquema: {schema}")
        chat_history = []
        chat_history.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": chat_history,
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.api_url, headers=self.headers, json=payload, timeout=30.0)
                response.raise_for_status()
                
                result = response.json()
                
                if result.get("candidates") and len(result["candidates"]) > 0 and \
                   result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
                   len(result["candidates"][0]["content"]["parts"]) > 0:
                    json_text = result["candidates"][0]["content"]["parts"][0].get("text", "")
                    parsed_json = json.loads(json_text)
                    logging.info(f"Respuesta estructurada de Gemini recibida: {parsed_json}")
                    return parsed_json
                else:
                    logging.warning(f"Respuesta estructurada inesperada de Gemini: {result}")
                    return {"error": "No pude generar una respuesta estructurada. La estructura de la respuesta de Gemini es inesperada."}
        except httpx.RequestError as e:
            logging.error(f"Error de red o de solicitud al llamar a la API de Gemini para respuesta estructurada: {e}")
            return {"error": f"Error de conexión con la IA para respuesta estructurada: {e}"}
        except httpx.HTTPStatusError as e:
            logging.error(f"Error de estado HTTP de la API de Gemini para respuesta estructurada: {e.response.status_code} - {e.response.text}")
            return {"error": f"Error de la IA (HTTP {e.response.status_code}) para respuesta estructurada: {e.response.text}"}
        except json.JSONDecodeError as e:
            logging.error(f"Error al decodificar la respuesta JSON estructurada de Gemini: {e} - Respuesta: {response.text}")
            return {"error": f"Error al procesar la respuesta estructurada de la IA: {e}"}
        except Exception as e:
            logging.error(f"Ocurrió un error inesperado al llamar a la API de Gemini para respuesta estructurada: {e}")
            return {"error": f"Error inesperado al comunicarse con la IA para respuesta estructurada: {e}"}

