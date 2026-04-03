from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("El contenido del mensaje no puede estar vacio.")
        return value


class BacteriaChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    bacteria_label: str = Field(min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)


class BacteriaChatResponse(BaseModel):
    answer: str
    bacteria_label: str
    refused: bool
    scope: Literal["bacteria"]
