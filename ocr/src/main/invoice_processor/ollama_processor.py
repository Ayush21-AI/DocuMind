import json
import re
import httpx
import logging
from typing import TypeVar, Type
from async_lru import alru_cache
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

JSON_REGEX = re.compile(r'({.*})', re.DOTALL)

T = TypeVar('T', bound=BaseModel)


def parse_ollama_response(result: dict, model_class: Type[T]) -> T:
    """
    Parse Ollama API response and return validated Pydantic model instance.

    Args:
        result: Raw response dict from Ollama API
        model_class: Pydantic model class to parse response into

    Returns:
        Instance of model_class with extracted data, or empty instance on failure
    """
    # /api/chat returns content inside message.content
    response_text = ""
    if "message" in result and "content" in result["message"]:
        response_text = result["message"]["content"].strip()
    else:
        response_text = result.get('response', '').strip()

    response_text = response_text.replace('json\n', '')
    logging.debug(f"Raw response text from Ollama: {response_text}")

    # Use regex to extract JSON content if there is extra text
    json_match = JSON_REGEX.search(response_text)
    json_str = json_match.group(1) if json_match else response_text

    try:
        data = json.loads(json_str)
        return model_class(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        logging.error(f"Error parsing Ollama response: {str(e)}. Raw response: {response_text}")
        return model_class()  # Returns all fields as default ""


class OllamaProcessor:
    """
    Generic Ollama processor for extracting structured data from text using LLM.

    This processor can be used with any Pydantic model by specifying the
    response_model during initialization.
    """

    def __init__(
            self,
            model_name: str,
            ollama_host: str,
            system_prompt: str,
            http_client: httpx.AsyncClient,
            response_model: Type[T],
            num_predict: int = 200
    ):
        self.model_name = model_name
        self.api_endpoint = f"{ollama_host}/api/chat"
        self.system_prompt = system_prompt
        self.client = http_client
        self.response_model = response_model
        self.num_predict = num_predict

    @alru_cache(maxsize=128)
    async def extract_data(self, text: str) -> T:
        """Extract structured data from text using cached LLM inference."""
        logging.info(f"Extracting {self.response_model.__name__} from text: {text}")
        try:
            return await self._retryable_extract(text)
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Unable to connect to Ollama service at {self.api_endpoint}. Ensure the service is running and accessible."
            )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail=f"Ollama service at {self.api_endpoint} timed out after retries."
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to communicate with Ollama service: {type(e).__name__}: {str(e)}"
            )
        except HTTPException:
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.RequestError),
        reraise=True,
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING)
    )
    async def _retryable_extract(self, text: str) -> T:
        """
        Extracts structured data by sending text along with the system prompt
        to the Ollama Chat API with retry logic for transient failures.
        System prompt is sent as a separate 'system' role message to enable
        Ollama's KV cache reuse across requests.
        """
        try:
            response = await self.client.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Document Text:\n{text}\n\nReturn raw JSON only, no markdown or extra text or nothing but just JSON strictly:"}
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0,
                        "num_predict": self.num_predict
                    }
                },
                timeout=30.0
            )
            response.raise_for_status()
            logging.info(f"Full Ollama response: {response.json()}")
            return parse_ollama_response(response.json(), self.response_model)

        except httpx.HTTPStatusError as e:
            logging.error(f"Ollama API returned an error response: {str(e)}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ollama returned HTTP {e.response.status_code}: {e.response.text}"
            )
        except httpx.RequestError:
            raise  # Let tenacity handle retries
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
