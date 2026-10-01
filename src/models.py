"""Pydantic models for the function calling pipeline."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class ParameterDef(BaseModel):
    """Type description of a single function parameter."""

    type: str


class ReturnDef(BaseModel):
    """Type description of a function return value."""

    type: str


class FunctionDefinition(BaseModel):
    """One entry of the functions definition file."""

    name: str
    description: str
    parameters: Dict[str, ParameterDef] = Field(default_factory=dict)
    returns: ReturnDef


class PromptInput(BaseModel):
    """One natural language prompt from the test file."""

    prompt: str


class FunctionCallResult(BaseModel):
    """One entry of the output file."""

    prompt: str
    name: str
    parameters: Dict[str, Any]
