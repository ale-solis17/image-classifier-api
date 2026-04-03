import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat_routes import router as chat_router
from app.integrations.openai_responses import OpenAITimeoutError
from app.schemas.chat import BacteriaChatRequest, ChatMessage
from app.services.bacteria_chat import (
    BacteriaChatValidationError,
    CHAT_SCOPE,
    chat_about_bacteria,
)


class FakeGateway:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create_response(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class BacteriaChatServiceTests(unittest.TestCase):
    @patch("app.services.bacteria_chat.load_labels", return_value=["Bacteroides fragilis", "Staphylococcus aureus"])
    def test_chat_about_bacteria_runs_tool_flow_and_returns_structured_response(self, _load_labels_mock):
        first_response = SimpleNamespace(
            id="resp_1",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="get_bacteria_context",
                    call_id="call_1",
                    arguments=json.dumps({"bacteria_label": "Bacteroides fragilis"}),
                )
            ],
        )
        second_response = SimpleNamespace(
            output_text=json.dumps(
                {
                    "answer": "Bacteroides fragilis puede presentar caracteristicas relevantes segun el contexto consultado.",
                    "bacteria_label": "Bacteroides fragilis",
                    "refused": False,
                    "scope": CHAT_SCOPE,
                }
            )
        )
        gateway = FakeGateway([first_response, second_response])
        request = BacteriaChatRequest(
            bacteria_label="bacteroides fragilis",
            messages=[ChatMessage(role="user", content="Como afecta esta bacteria a la agricultura?")],
        )

        result = asyncio.run(chat_about_bacteria(request, gateway=gateway))

        self.assertEqual(result.bacteria_label, "Bacteroides fragilis")
        self.assertFalse(result.refused)
        self.assertEqual(result.scope, CHAT_SCOPE)
        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(gateway.calls[0]["tool_choice"], {"type": "function", "name": "get_bacteria_context"})
        self.assertEqual(gateway.calls[1]["previous_response_id"], "resp_1")

    @patch("app.services.bacteria_chat.load_labels", return_value=["Bacteroides fragilis"])
    def test_chat_about_bacteria_rejects_unrecognized_label(self, _load_labels_mock):
        request = BacteriaChatRequest(
            bacteria_label="No reconocido",
            messages=[ChatMessage(role="user", content="Que significa esto para el cultivo?")],
        )

        with self.assertRaisesRegex(BacteriaChatValidationError, "No se puede abrir el chat"):
            asyncio.run(chat_about_bacteria(request, gateway=FakeGateway([])))

    @patch("app.services.bacteria_chat.load_labels", return_value=["Bacteroides fragilis"])
    def test_chat_about_bacteria_rejects_unknown_bacteria(self, _load_labels_mock):
        request = BacteriaChatRequest(
            bacteria_label="Escherichia coli",
            messages=[ChatMessage(role="user", content="Que impacto tiene en el campo?")],
        )

        with self.assertRaisesRegex(BacteriaChatValidationError, "no pertenece"):
            asyncio.run(chat_about_bacteria(request, gateway=FakeGateway([])))

    @patch("app.services.bacteria_chat.load_labels", return_value=["Bacteroides fragilis"])
    def test_chat_about_bacteria_requires_last_message_from_user(self, _load_labels_mock):
        request = BacteriaChatRequest(
            bacteria_label="Bacteroides fragilis",
            messages=[ChatMessage(role="assistant", content="Hola, en que te ayudo?")],
        )

        with self.assertRaisesRegex(BacteriaChatValidationError, "ultimo mensaje"):
            asyncio.run(chat_about_bacteria(request, gateway=FakeGateway([])))

    @patch("app.services.bacteria_chat.OPENAI_CHAT_MAX_MESSAGES", 2)
    @patch("app.services.bacteria_chat.OPENAI_CHAT_MAX_CHARS", 40)
    @patch("app.services.bacteria_chat.load_labels", return_value=["Bacteroides fragilis"])
    def test_chat_about_bacteria_trims_history_before_calling_openai(self, _load_labels_mock):
        first_response = SimpleNamespace(
            id="resp_trim",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="get_bacteria_context",
                    call_id="call_trim",
                    arguments=json.dumps({"bacteria_label": "Bacteroides fragilis"}),
                )
            ],
        )
        second_response = SimpleNamespace(
            output_text=json.dumps(
                {
                    "answer": "Respuesta corta.",
                    "bacteria_label": "Bacteroides fragilis",
                    "refused": False,
                    "scope": CHAT_SCOPE,
                }
            )
        )
        gateway = FakeGateway([first_response, second_response])
        request = BacteriaChatRequest(
            bacteria_label="Bacteroides fragilis",
            messages=[
                ChatMessage(role="user", content="Mensaje muy viejo que debe salir del contexto"),
                ChatMessage(role="assistant", content="Respuesta vieja"),
                ChatMessage(role="user", content="Consulta actual"),
            ],
        )

        asyncio.run(chat_about_bacteria(request, gateway=gateway))

        self.assertEqual(
            gateway.calls[0]["input"],
            [
                {"role": "assistant", "content": "Respuesta vieja"},
                {"role": "user", "content": "Consulta actual"},
            ],
        )


class BacteriaChatRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(chat_router)
        self.client = TestClient(app)

    @patch("app.api.chat_routes.chat_about_bacteria", new_callable=AsyncMock)
    def test_route_returns_200_on_success(self, chat_mock):
        chat_mock.return_value = {
            "answer": "Respuesta sobre la bacteria",
            "bacteria_label": "Bacteroides fragilis",
            "refused": False,
            "scope": CHAT_SCOPE,
        }

        response = self.client.post(
            "/chat/bacteria",
            json={
                "bacteria_label": "Bacteroides fragilis",
                "messages": [{"role": "user", "content": "Que caracteristicas tiene esta bacteria?"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["scope"], CHAT_SCOPE)

    @patch("app.api.chat_routes.chat_about_bacteria", new_callable=AsyncMock)
    def test_route_maps_validation_errors_to_400(self, chat_mock):
        chat_mock.side_effect = BacteriaChatValidationError("Etiqueta invalida")

        response = self.client.post(
            "/chat/bacteria",
            json={
                "bacteria_label": "No reconocido",
                "messages": [{"role": "user", "content": "Pregunta"}],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Etiqueta invalida")

    @patch("app.api.chat_routes.chat_about_bacteria", new_callable=AsyncMock)
    def test_route_maps_timeout_errors_to_504(self, chat_mock):
        chat_mock.side_effect = OpenAITimeoutError("Tiempo agotado")

        response = self.client.post(
            "/chat/bacteria",
            json={
                "bacteria_label": "Bacteroides fragilis",
                "messages": [{"role": "user", "content": "Pregunta"}],
            },
        )

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["detail"], "Tiempo agotado")


if __name__ == "__main__":
    unittest.main()
