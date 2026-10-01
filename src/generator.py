"""Turns one natural language prompt into one function call."""

import json
from typing import Any, Callable, Dict, List

from src.constrained_decoder import (
    GetLogits,
    TokenSets,
    generate_choice,
    generate_number,
    generate_string,
    parse_number,
)
from src.models import FunctionDefinition, ParameterDef

Encode = Callable[[str], List[int]]

SYSTEM_PROMPT = (
    "You convert a request into one function call written as JSON: "
    '{"name": ..., "parameters": {...}}. Never answer the request itself.\n'
    "Rules:\n"
    "- Copy text values exactly from the request, without surrounding "
    "quotes.\n"
    "- Numbers are plain numbers from the request.\n"
    "- Regular expressions use JSON escapes (\\\\d+ for numbers) and "
    "character classes such as [xyz] for any of several characters.\n"
    "- A replacement is the literal text to insert (- for a dash), never "
    "its name.\n"
    "Example (made-up function):\n"
    "Request: Replace the letters x and y in 'xy1' with a dash\n"
    '{"name": "fn_example", "parameters": {"text": "xy1", '
    '"pattern": "[xy]", "replacement": "-"}}'
)


def build_prompt(
    user_prompt: str,
    functions: List[FunctionDefinition]
) -> str:
    """Build the chat-formatted prompt that precedes the generated JSON.

    Args:
        user_prompt: The natural language request.
        functions: The functions the model may choose from.

    Returns:
        The prompt text given to the model (Qwen chat format, no thinking).
    """
    lines: List[str] = []
    for fn in functions:
        params = ", ".join(
            f"{name}: {p.type}" for name, p in fn.parameters.items()
        )
        lines.append(f"- {fn.name}({params}): {fn.description}")
    user = "Functions:\n" + "\n".join(lines) + f"\nRequest: {user_prompt}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def build_context(
    prompt_text: str,
    function_name: str,
    done: Dict[str, Any],
    next_param: str,
    next_is_string: bool,
) -> str:
    """Build the text ending right where the next value starts.

    Args:
        prompt_text: Text produced by ``build_prompt``.
        function_name: Name of the chosen function.
        done: Parameters generated so far.
        next_param: Name of the parameter about to be generated.
        next_is_string: Whether the next value opens with a quote.

    Returns:
        The context to feed to the model.
    """
    parts = [
        f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}"
        for k, v in done.items()
    ]
    parts.append(f"{json.dumps(next_param)}: ")
    text = (
        prompt_text
        + '{"name": ' + json.dumps(function_name)
        + ', "parameters": {' + ", ".join(parts)
    )
    return text + '"' if next_is_string else text


def generate_value(
    get_logits: GetLogits,
    input_ids: List[int],
    param: ParameterDef,
    sets: TokenSets,
) -> Any:
    """Generate one parameter value according to its declared type.

    Args:
        get_logits: Function returning next-token logits for token ids.
        input_ids: Token ids of the context.
        param: Definition of the parameter.
        sets: Pre-computed token sets.

    Returns:
        The value, typed as declared (float, int, str or bool).

    Raises:
        ValueError: If the parameter type is not supported.
    """
    if param.type == "number":
        text = generate_number(get_logits, input_ids, sets)
        return parse_number(text, False)
    if param.type == "integer":
        text = generate_number(get_logits, input_ids, sets, integer=True)
        return int(parse_number(text, True))
    if param.type == "string":
        return generate_string(get_logits, input_ids, sets)
    if param.type == "boolean":
        word = generate_choice(
            get_logits, input_ids, ["true", "false"], sets, sets.stop_ids
        )
        return word == "true"
    raise ValueError(f"unsupported parameter type {param.type!r}")


def process_prompt(
    get_logits: GetLogits,
    encode: Encode,
    user_prompt: str,
    functions: List[FunctionDefinition],
    sets: TokenSets,
) -> Dict[str, Any]:
    """Translate one prompt into a function name and its arguments.

    Args:
        get_logits: Function returning next-token logits for token ids.
        encode: Function converting text to token ids.
        user_prompt: The natural language request.
        functions: The available function definitions.
        sets: Pre-computed token sets.

    Returns:
        A dict with the keys ``name`` and ``parameters``.
    """
    names = [fn.name for fn in functions]
    # The name is generated right after the opening quote. Forcing part of
    # it (e.g. "fn_") would split it at an unnatural token boundary and
    # mislead the model.
    ids = encode(build_prompt(user_prompt, functions) + '{"name": "')
    name = generate_choice(get_logits, ids, names, sets, sets.quote_ids)
    chosen = next(fn for fn in functions if fn.name == name)

    # Shorter prompt for the arguments: only the chosen function is listed.
    prompt_text = build_prompt(user_prompt, [chosen])
    params: Dict[str, Any] = {}
    for param_name, param_def in chosen.parameters.items():
        context = build_context(
            prompt_text, chosen.name, params, param_name,
            param_def.type == "string",
        )
        params[param_name] = generate_value(
            get_logits, encode(context), param_def, sets
        )
    return {"name": chosen.name, "parameters": params}
