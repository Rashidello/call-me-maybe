"""Constrained decoding primitives.

Every ``generate_*`` function runs the model token by token. At each step the
logits of every token that would break the expected structure are replaced by
negative infinity, so the model can only ever pick a valid continuation.
"""

import json
import re
from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

GetLogits = Callable[[List[int]], List[float]]

ESCAPABLE = frozenset('"\\/bfnrt')
NUMBER_CHARS = frozenset("0123456789.-")
_PARTIAL_FLOAT = re.compile(r"^-?(\d+(\.\d*)?)?$")
_PARTIAL_INT = re.compile(r"^-?\d*$")
_COMPLETE_FLOAT = re.compile(r"^-?\d+(\.\d+)?$")
_COMPLETE_INT = re.compile(r"^-?\d+$")


class TokenSets(BaseModel):
    """Token id sets pre-computed once from the vocabulary."""

    id_to_token: Dict[int, str]
    text_to_ids: Dict[str, List[int]]
    number_ids: List[int]
    stop_ids: List[int]
    stop_set: Set[int]
    quote_ids: List[int]
    free_ids: List[int]
    escaped_ids: List[int]


def mask_logits(
    logits: List[float],
    valid_token_ids: Sequence[int]
) -> npt.NDArray[np.float64]:
    """Set every logit to -inf except those of the valid token ids.

    Args:
        logits: Raw logits returned by the model.
        valid_token_ids: Token ids that may be selected.

    Returns:
        A copy of the logits where invalid tokens are -inf.
    """
    arr = np.asarray(logits, dtype=np.float64)
    masked = np.full(arr.shape, -np.inf)
    ids = np.asarray(valid_token_ids, dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < arr.shape[0])]
    masked[ids] = arr[ids]
    return masked


def select_token(masked_logits: npt.NDArray[np.float64]) -> int:
    """Return the id of the highest scoring token.

    Args:
        masked_logits: Logits after masking.

    Returns:
        The selected token id.

    Raises:
        ValueError: If no token is allowed.
    """
    token_id = int(np.argmax(masked_logits))
    if masked_logits[token_id] == -np.inf:
        raise ValueError("no valid token available for this step")
    return token_id


def scan_token(text: str, pending: bool) -> Tuple[bool, int, bool]:
    """Check a token against the JSON string grammar.

    Args:
        text: Decoded text of the candidate token.
        pending: True if the previous character was a lone backslash.

    Returns:
        A tuple ``(valid, end, pending)``: whether the token is allowed,
        the index of the closing quote inside the token (-1 if none) and
        whether a backslash is still waiting for its escaped character.
    """
    for i, char in enumerate(text):
        if pending:
            if char not in ESCAPABLE:
                return False, -1, False
            pending = False
        elif char == "\\":
            pending = True
        elif char == '"':
            return True, i, False
    return True, -1, pending


def _has_unusable_char(text: str) -> bool:
    """Tell whether a token holds control or partial-UTF-8 characters."""
    return any(ord(c) < 32 or c == "�" for c in text)


def precompute_token_sets(id_to_token: Dict[int, str]) -> TokenSets:
    """Classify every vocabulary token once, before generation starts.

    Args:
        id_to_token: Mapping from token id to its decoded text.

    Returns:
        The token sets used by the ``generate_*`` functions.
    """
    text_to_ids: Dict[str, List[int]] = {}
    number_ids: List[int] = []
    stop_ids: List[int] = []
    quote_ids: List[int] = []
    free_ids: List[int] = []
    escaped_ids: List[int] = []

    for tid, text in id_to_token.items():
        if not text:
            continue
        text_to_ids.setdefault(text, []).append(tid)
        stripped = text.lstrip()
        if stripped.startswith(",") or stripped.startswith("}"):
            stop_ids.append(tid)
        if text.startswith('"'):
            quote_ids.append(tid)
        if all(c in NUMBER_CHARS for c in text):
            number_ids.append(tid)
        if _has_unusable_char(text):
            continue
        if scan_token(text, False)[0]:
            free_ids.append(tid)
        if scan_token(text, True)[0]:
            escaped_ids.append(tid)

    return TokenSets(
        id_to_token=id_to_token,
        text_to_ids=text_to_ids,
        number_ids=number_ids,
        stop_ids=stop_ids,
        stop_set=set(stop_ids),
        quote_ids=quote_ids,
        free_ids=free_ids,
        escaped_ids=escaped_ids,
    )


def generate_choice(
    get_logits: GetLogits,
    input_ids: List[int],
    choices: List[str],
    sets: TokenSets,
    end_ids: Sequence[int],
) -> str:
    """Generate exactly one of the given strings.

    Only tokens that keep the generated text a prefix of one of the choices
    are allowed. When a choice is also the prefix of a longer one, the tokens
    in ``end_ids`` let the model decide to stop.

    Args:
        get_logits: Function returning next-token logits for token ids.
        input_ids: Token ids of the context.
        choices: Allowed output strings.
        sets: Pre-computed token sets.
        end_ids: Token ids that end the choice once it is complete.

    Returns:
        The chosen string.

    Raises:
        ValueError: If no choice can be generated.
    """
    generated = ""
    ids = list(input_ids)
    end_set = set(end_ids)
    while True:
        remaining = [c for c in choices if c.startswith(generated)]
        longer = [c for c in remaining if len(c) > len(generated)]
        if not remaining:
            raise ValueError("generation left every allowed choice")
        if not longer:
            return generated
        if len(remaining) == 1:
            return remaining[0]

        valid: Set[int] = set()
        for choice in longer:
            rest = choice[len(generated):]
            for length in range(1, len(rest) + 1):
                valid.update(sets.text_to_ids.get(rest[:length], ()))
        complete = generated in remaining
        if complete:
            valid.update(end_set)

        token_id = select_token(mask_logits(get_logits(ids), list(valid)))
        if complete and token_id in end_set:
            return generated
        generated += sets.id_to_token[token_id]
        ids.append(token_id)


def generate_number(
    get_logits: GetLogits,
    input_ids: List[int],
    sets: TokenSets,
    integer: bool = False,
    max_tokens: int = 24,
) -> str:
    """Generate a JSON number token by token.

    Only tokens that keep the text a valid prefix of a number are allowed,
    and the value can only be ended once it is a complete number.

    Args:
        get_logits: Function returning next-token logits for token ids.
        input_ids: Token ids of the context.
        sets: Pre-computed token sets.
        integer: Forbid a decimal part.
        max_tokens: Safety limit on the number of generated tokens.

    Returns:
        The generated number as text.
    """
    partial = _PARTIAL_INT if integer else _PARTIAL_FLOAT
    complete = _COMPLETE_INT if integer else _COMPLETE_FLOAT
    current = ""
    ids = list(input_ids)

    for _ in range(max_tokens):
        valid = [
            tid for tid in sets.number_ids
            if partial.match(current + sets.id_to_token[tid])
        ]
        can_stop = complete.match(current) is not None
        if can_stop:
            valid.extend(sets.stop_ids)
        if not valid:
            break
        token_id = select_token(mask_logits(get_logits(ids), valid))
        if can_stop and token_id in sets.stop_set:
            break
        current += sets.id_to_token[token_id]
        ids.append(token_id)
    return current


def parse_number(text: str, integer: bool) -> float:
    """Convert generated number text to a Python number.

    Args:
        text: Text produced by ``generate_number``.
        integer: Return an ``int`` instead of a ``float``.

    Returns:
        The parsed value.

    Raises:
        ValueError: If the text holds no digits.
    """
    cleaned = text.rstrip(".")
    if not any(c.isdigit() for c in cleaned):
        raise ValueError(f"the model produced no number (got {text!r})")
    if integer:
        return int(cleaned)
    return float(cleaned)


def generate_string(
    get_logits: GetLogits,
    input_ids: List[int],
    sets: TokenSets,
    max_tokens: int = 150,
) -> str:
    """Generate the body of a JSON string up to its closing quote.

    The model writes JSON, so escapes such as ``\\\\d`` are allowed but only
    the escapes JSON defines. A token containing the closing quote ends the
    string; whatever follows the quote inside that token is dropped.

    Args:
        get_logits: Function returning next-token logits for token ids.
        input_ids: Token ids of the context, ending right after the
            opening quote.
        sets: Pre-computed token sets.
        max_tokens: Safety limit on the number of generated tokens.

    Returns:
        The decoded string value.
    """
    raw = ""
    pending = False
    ids = list(input_ids)
    closed = False

    for _ in range(max_tokens):
        valid = sets.escaped_ids if pending else sets.free_ids
        token_id = select_token(mask_logits(get_logits(ids), valid))
        text = sets.id_to_token[token_id]
        _, end, pending = scan_token(text, pending)
        if end >= 0:
            raw += text[:end]
            closed = True
            break
        raw += text
        ids.append(token_id)

    if not closed and pending:
        raw = raw[:-1]
    try:
        value = json.loads('"' + raw + '"')
    except ValueError:
        return raw
    return str(value)
