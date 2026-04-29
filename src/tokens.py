"""Token counting utilities, model-aware."""

import tiktoken

_ENCODER_CACHE: dict[str, tiktoken.Encoding] = {}

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "claude-opus-4-7": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
}

# tiktoken doesn't know Claude models — cl100k_base is close enough for estimation
_TIKTOKEN_MODEL_MAP = {
    "claude-opus-4-7": "cl100k_base",
    "claude-sonnet-4-6": "cl100k_base",
    "claude-haiku-4-5": "cl100k_base",
}


def _encoder(model: str) -> tiktoken.Encoding:
    if model not in _ENCODER_CACHE:
        encoding_name = _TIKTOKEN_MODEL_MAP.get(model, model)
        try:
            enc = tiktoken.encoding_for_model(encoding_name)
        except KeyError:
            enc = tiktoken.get_encoding(encoding_name)
        _ENCODER_CACHE[model] = enc
    return _ENCODER_CACHE[model]


def count_tokens(text: str, model: str = "claude-sonnet-4-6") -> int:
    return len(_encoder(model).encode(text))


def count_tokens_batch(texts: list[str], model: str = "claude-sonnet-4-6") -> list[int]:
    enc = _encoder(model)
    return [len(enc.encode(t)) for t in texts]


def fits_in_context(text: str, model: str = "claude-sonnet-4-6", reserve: int = 4096) -> bool:
    limit = MODEL_CONTEXT_LIMITS.get(model, 128_000)
    return count_tokens(text, model) <= limit - reserve


def context_utilisation(text: str, model: str = "claude-sonnet-4-6") -> float:
    limit = MODEL_CONTEXT_LIMITS.get(model, 128_000)
    return count_tokens(text, model) / limit
