from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.ai.loader import load_labels
from app.core.config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MAX_CHARS,
    OPENAI_CHAT_MAX_MESSAGES,
    OPENAI_CHAT_MODEL,
    OPENAI_CHAT_TIMEOUT_S,
    UNKNOWN_LABEL,
)
from app.integrations.openai_responses import (
    OpenAIConfigurationError,
    OpenAIResponsesGateway,
    OpenAIServiceError,
    OpenAITimeoutError,
)
from app.schemas.chat import BacteriaChatRequest, BacteriaChatResponse, ChatMessage


logger = logging.getLogger(__name__)

CHAT_SCOPE = "bacteria"
TOOL_NAME = "get_bacteria_context"
MAX_MESSAGE_CHARS = 1000

FINAL_RESPONSE_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "bacteria_chat_response",
        "description": "Respuesta estructurada del chat sobre una bacteria clasificada.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Respuesta final en espanol para el usuario.",
                },
                "bacteria_label": {
                    "type": "string",
                    "description": "Nombre canonico de la bacteria sobre la que se esta hablando.",
                },
                "refused": {
                    "type": "boolean",
                    "description": "Indica si la respuesta rechazo la pregunta por estar fuera de alcance.",
                },
                "scope": {
                    "type": "string",
                    "enum": [CHAT_SCOPE],
                    "description": "Alcance fijo del chat.",
                },
            },
            "required": ["answer", "bacteria_label", "refused", "scope"],
            "additionalProperties": False,
        },
    }
}


class BacteriaChatValidationError(ValueError):
    pass


def _build_gateway() -> OpenAIResponsesGateway:
    return OpenAIResponsesGateway(
        api_key=OPENAI_API_KEY,
        timeout_s=OPENAI_CHAT_TIMEOUT_S,
    )


def _build_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": TOOL_NAME,
            "description": (
                "Obtiene el contexto canonico de la bacteria ya clasificada y las reglas "
                "de alcance del chat para responder solo en espanol sobre esa bacteria."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "bacteria_label": {
                        "type": "string",
                        "description": "Nombre de la bacteria clasificada por la API.",
                    }
                },
                "required": ["bacteria_label"],
                "additionalProperties": False,
            },
        }
    ]


def _build_developer_instructions() -> str:
    return (
        "Eres un asistente para un sistema de clasificacion de bacterias. "
        "Debes responder siempre en espanol. "
        "Solo puedes hablar sobre la bacteria clasificada. "
        "Puedes responder sobre caracteristicas, contexto, riesgos, prevencion, impacto y temas relacionados con esa bacteria. "
        "Si el usuario pregunta sobre otra bacteria o sobre temas no relacionados con la bacteria clasificada, "
        "debes rechazar con cortesia, mantenerte dentro del alcance permitido y marcar refused=true. "
        "No inventes datos. Si no estas seguro, dilo claramente. "
        "La respuesta final debe cumplir exactamente el esquema JSON solicitado."
    )


def _get_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _load_known_labels() -> list[str]:
    try:
        return load_labels()
    except (FileNotFoundError, ValueError) as exc:
        raise OpenAIConfigurationError(
            "No hay etiquetas de bacterias disponibles para habilitar el chat."
        ) from exc


def _normalize_bacteria_label(raw_label: str, known_labels: list[str]) -> str:
    if not raw_label.strip():
        raise BacteriaChatValidationError("Debes indicar la bacteria clasificada.")

    if raw_label.strip() == UNKNOWN_LABEL:
        raise BacteriaChatValidationError(
            "No se puede abrir el chat si la clasificacion fue 'No reconocido'."
        )

    labels_by_key = {label.casefold(): label for label in known_labels}
    canonical_label = labels_by_key.get(raw_label.strip().casefold())
    if canonical_label is None:
        raise BacteriaChatValidationError("La bacteria indicada no pertenece a las etiquetas conocidas del modelo.")

    return canonical_label


def _normalize_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    if not messages:
        raise BacteriaChatValidationError("Debes enviar al menos un mensaje para el chat.")

    trimmed_messages = messages[-OPENAI_CHAT_MAX_MESSAGES:]
    normalized_messages: list[dict[str, str]] = []

    for message in trimmed_messages:
        content = message.content.strip()
        if len(content) > MAX_MESSAGE_CHARS:
            raise BacteriaChatValidationError(
                f"Cada mensaje debe tener como maximo {MAX_MESSAGE_CHARS} caracteres."
            )

        normalized_messages.append({"role": message.role, "content": content})

    while normalized_messages and sum(len(message["content"]) for message in normalized_messages) > OPENAI_CHAT_MAX_CHARS:
        normalized_messages.pop(0)

    if not normalized_messages:
        raise BacteriaChatValidationError("El historial del chat supera el limite permitido de caracteres.")

    if normalized_messages[-1]["role"] != "user":
        raise BacteriaChatValidationError("El ultimo mensaje del historial debe pertenecer al usuario.")

    return normalized_messages


def _build_bacteria_context(canonical_label: str) -> dict[str, Any]:
    return {
        "bacteria_label": canonical_label,
        "language": "es",
        "scope": CHAT_SCOPE,
        "allowed_domain": "solo temas relacionados a la bacteria clasificada",
        "policy": {
            "answer_only_about_this_bacteria": True,
            "reject_other_bacteria": True,
            "reject_unrelated_topics": True,
        },
        "refusal_style": (
            "Si la pregunta esta fuera de alcance, explica con cortesia que solo puedes ayudar "
            "sobre la bacteria clasificada."
        ),
    }


def _extract_tool_call(response: Any) -> tuple[str, dict[str, Any]]:
    for item in _get_value(response, "output") or []:
        if _get_value(item, "type") != "function_call":
            continue

        arguments = _get_value(item, "arguments") or "{}"
        parsed_arguments = json.loads(arguments)
        return _get_value(item, "call_id"), parsed_arguments

    raise OpenAIServiceError("OpenAI no solicito la herramienta requerida para contextualizar la bacteria.")


def _parse_final_response(output_text: str, canonical_label: str) -> BacteriaChatResponse:
    if not output_text.strip():
        raise OpenAIServiceError("OpenAI devolvio una respuesta vacia para el chat.")

    try:
        payload = BacteriaChatResponse.model_validate_json(output_text)
    except Exception as exc:
        raise OpenAIServiceError("No fue posible interpretar la respuesta estructurada del chat.") from exc

    if not payload.answer.strip():
        raise OpenAIServiceError("La respuesta del chat no incluyo contenido util.")

    return BacteriaChatResponse(
        answer=payload.answer.strip(),
        bacteria_label=canonical_label,
        refused=payload.refused,
        scope=CHAT_SCOPE,
    )


async def chat_about_bacteria(
    chat_request: BacteriaChatRequest,
    gateway: OpenAIResponsesGateway | None = None,
) -> BacteriaChatResponse:
    start_time = time.perf_counter()
    known_labels = _load_known_labels()
    canonical_label = _normalize_bacteria_label(chat_request.bacteria_label, known_labels)
    normalized_messages = _normalize_messages(chat_request.messages)
    gateway = gateway or _build_gateway()

    first_response = await gateway.create_response(
        model=OPENAI_CHAT_MODEL,
        instructions=_build_developer_instructions(),
        input=normalized_messages,
        tools=_build_tools(),
        tool_choice={"type": "function", "name": TOOL_NAME},
        parallel_tool_calls=False,
    )

    call_id, tool_arguments = _extract_tool_call(first_response)
    requested_label = str(tool_arguments.get("bacteria_label", "")).strip()
    if requested_label and requested_label.casefold() != canonical_label.casefold():
        logger.info(
            "OpenAI solicito contexto para una bacteria distinta. solicitada=%s canonica=%s",
            requested_label,
            canonical_label,
        )

    tool_output = json.dumps(_build_bacteria_context(canonical_label), ensure_ascii=False)

    second_response = await gateway.create_response(
        model=OPENAI_CHAT_MODEL,
        instructions=_build_developer_instructions(),
        previous_response_id=_get_value(first_response, "id"),
        input=[
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_output,
            }
        ],
        text=FINAL_RESPONSE_FORMAT,
    )

    result = _parse_final_response(_get_value(second_response, "output_text") or "", canonical_label)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    logger.info(
        "Chat bacteria completado. bacteria_label=%s refused=%s duration_ms=%s",
        canonical_label,
        result.refused,
        elapsed_ms,
    )

    return result


__all__ = [
    "BacteriaChatValidationError",
    "CHAT_SCOPE",
    "chat_about_bacteria",
    "OpenAIConfigurationError",
    "OpenAIServiceError",
    "OpenAITimeoutError",
]
