import json
import logging
import re
from typing import Type, TypeVar

import httpx
from async_lru import alru_cache
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError
from tenacity import (before_sleep_log, retry, retry_if_exception_type,
                      stop_after_attempt, wait_exponential)

JSON_REGEX = re.compile(r"({.*})", re.DOTALL)

T = TypeVar("T", bound=BaseModel)


def parse_ollama_response(result: dict, model_class: Type[T]) -> T:
    """
    Parse an Ollama API response into a validated Pydantic model.

    With schema-constrained decoding (`format=<json schema>`) the content is
    already valid JSON matching the schema, so `model_validate_json` succeeds
    directly. The regex fallback is retained for robustness if the server is
    ever run without structured outputs.
    """
    if "message" in result and "content" in result["message"]:
        response_text = result["message"]["content"].strip()
    else:
        response_text = result.get("response", "").strip()

    logging.debug(f"Raw response text from Ollama: {response_text}")

    # Fast path: constrained decoding guarantees clean JSON.
    try:
        return model_class.model_validate_json(response_text)
    except ValidationError:
        pass

    # Fallback: dig a JSON object out of noisy text (markdown fences, prose).
    cleaned = response_text.replace("json\n", "")
    match = JSON_REGEX.search(cleaned)
    json_str = match.group(1) if match else cleaned
    try:
        return model_class(**json.loads(json_str))
    except (json.JSONDecodeError, ValidationError) as e:
        logging.error(f"Error parsing Ollama response: {e}. Raw: {response_text}")
        return model_class()  # all fields default to ""


class OllamaProcessor:
    """
    Generic, schema-constrained Ollama extractor.

    Works with any Pydantic model: the model's JSON Schema is passed to
    Ollama's `format` parameter so generation is constrained at the token
    level to match the schema (Ollama >= 0.5.0). This eliminates the malformed
    /renamed-key failures you get with the older `format: "json"` mode.
    """

    def __init__(
        self,
        model_name: str,
        ollama_host: str,
        system_prompt: str,
        http_client: httpx.AsyncClient,
        response_model: Type[T],
        num_predict: int = 256,
        num_ctx: int = 4096,
        seed: int = 42,
        keep_alive: str = "-1",
    ):
        self.model_name = model_name
        self.api_endpoint = f"{ollama_host}/api/chat"
        self.system_prompt = system_prompt
        self.client = http_client
        self.response_model = response_model
        self.schema = response_model.model_json_schema()
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.seed = seed
        self.keep_alive = keep_alive

    @alru_cache(maxsize=128)
    async def extract_data(self, text: str) -> T:
        """Extract structured data from text (LRU-cached on identical input)."""
        logging.info(f"Extracting {self.response_model.__name__} from {len(text)} chars of text")
        try:
            return await self._retryable_extract(text)
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Unable to connect to Ollama at {self.api_endpoint}.",
            )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail=f"Ollama at {self.api_endpoint} timed out after retries.",
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to communicate with Ollama: {type(e).__name__}: {e}",
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.RequestError),
        reraise=True,
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    )
    async def _retryable_extract(self, text: str) -> T:
        """
        POST to Ollama's chat API with retry on transient network errors.

        The system prompt is a separate `system` message (enabling server-side
        KV-cache reuse of the prompt prefix) and the response is constrained to
        `self.schema`. Deterministic options: temperature 0, fixed seed,
        repeat_penalty disabled (legitimate repeats in JSON), bounded context.
        """
        try:
            response = await self.client.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Document Text:\n{text}"},
                    ],
                    "stream": False,
                    "format": self.schema,
                    "keep_alive": self.keep_alive,
                    "options": {
                        "temperature": 0,
                        "top_p": 1.0,
                        "seed": self.seed,
                        "repeat_penalty": 1.0,
                        "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict,
                    },
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return parse_ollama_response(response.json(), self.response_model)

        except httpx.HTTPStatusError as e:
            logging.error(f"Ollama API error: {e}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ollama returned HTTP {e.response.status_code}: {e.response.text}",
            )
        except httpx.RequestError:
            raise  # tenacity retries
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")
