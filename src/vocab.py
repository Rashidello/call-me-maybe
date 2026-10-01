"""Loading of the tokenizer vocabulary file."""

import json
from typing import Dict


def _build_byte_decoder() -> Dict[str, int]:
    """Build the reverse GPT-2 byte-to-unicode mapping.

    Returns:
        A mapping from the printable unicode stand-in to the raw byte value.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _decode_token(token_str: str, byte_decoder: Dict[str, int]) -> str:
    """Convert a byte-level BPE token back to the text it stands for.

    Args:
        token_str: Token as stored in the vocabulary file.
        byte_decoder: Mapping produced by ``_build_byte_decoder``.

    Returns:
        The decoded text (invalid UTF-8 becomes the replacement character).
    """
    raw = bytes(byte_decoder.get(c, ord(c) & 0xFF) for c in token_str)
    return raw.decode("utf-8", errors="replace")


def load_vocab(vocab_path: str) -> Dict[int, str]:
    """Load the vocabulary file.

    Args:
        vocab_path: Path to the ``vocab.json`` file.

    Returns:
        A mapping from token id to the decoded text of that token.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is not a valid vocabulary.
    """
    byte_decoder = _build_byte_decoder()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    if not isinstance(vocab_data, dict):
        raise ValueError(f"{vocab_path}: expected a JSON object")
    return {
        int(token_id): _decode_token(token_str, byte_decoder)
        for token_str, token_id in vocab_data.items()
    }
