from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


class OpenAIConfigurationError(RuntimeError):
    pass


class OpenAIServiceError(RuntimeError):
    pass


class OpenAITimeoutError(OpenAIServiceError):
    pass


class OpenAIResponsesGateway:
    def __init__(self, api_key: str, timeout_s: float):
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._client: Any | None = None

    def _build_client(self) -> Any:
        if not self._api_key:
            raise OpenAIConfigurationError("OPENAI_API_KEY no esta configurada en el backend.")

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise OpenAIConfigurationError(
                "La dependencia 'openai' no esta instalada. Ejecuta la sincronizacion de dependencias."
            ) from exc

        return AsyncOpenAI(api_key=self._api_key, timeout=self._timeout_s)

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def create_response(self, **kwargs: Any) -> Any:
        try:
            return await self.client.responses.create(**kwargs)
        except Exception as exc:
            raise self._map_exception(exc) from exc

    def _map_exception(self, exc: Exception) -> Exception:
        error_name = exc.__class__.__name__

        if error_name == "APITimeoutError":
            return OpenAITimeoutError("OpenAI no respondio a tiempo.")

        if error_name in {
            "APIConnectionError",
            "APIStatusError",
            "AuthenticationError",
            "PermissionDeniedError",
            "RateLimitError",
            "BadRequestError",
            "InternalServerError",
        }:
            logger.warning("Fallo al invocar OpenAI Responses API: %s", error_name)
            return OpenAIServiceError("No fue posible obtener una respuesta del servicio de chat.")

        logger.exception("Error inesperado al invocar OpenAI Responses API.")
        return OpenAIServiceError("Ocurrio un error inesperado al consultar el servicio de chat.")
