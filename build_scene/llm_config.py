"""Runtime LLM configuration for scene generation.

The pipeline hard-codes several different models for different roles
(object extraction / best-choice selection use ``gemini-2.5-pro``; the
placement-reasoning roles use ``config.MODEL``). ``GenConfig`` lets a caller
(the CLI) override the model, temperature, base URL and API key for *all*
chat-LLM roles at once, while preserving the legacy hard-coded defaults when
no override is supplied (``gen=None``).
"""

from dataclasses import dataclass
from typing import Optional

from config import API_KEY, BASE_URL, MODEL
from utils.llm import Llm

# Shared defaults for every chat-LLM call in the pipeline.
_MAX_TOKENS = 32768
_TIMEOUT = 600
_MAX_RETRIES = 5


@dataclass
class GenConfig:
    """Overrideable LLM parameters for a single generation run.

    Fields default to the legacy ``config`` values so a ``GenConfig()`` with
    no overrides reproduces the original placement behaviour; pass an explicit
    ``model``/``temperature``/``base_url``/``api_key`` to redirect every
    chat-LLM role in the pipeline.
    """

    model: str = MODEL
    temperature: Optional[float] = None
    base_url: Optional[str] = BASE_URL
    api_key: Optional[str] = API_KEY


def build_llm(gen: Optional[GenConfig], default_model: str) -> tuple[Llm, Optional[float]]:
    """Build an ``Llm`` client and return ``(llm, temperature)``.

    When ``gen`` is ``None`` the legacy per-role behaviour is preserved: the
    caller-supplied ``default_model`` is used and temperature is left unset
    (deterministic, as before). When ``gen`` is provided its fields override
    the defaults — a non-empty ``gen.model`` wins over ``default_model`` so a
    single ``--model`` flag reaches every chat-LLM role.
    """
    if gen is None:
        model = default_model
        temperature = None
        api_key = API_KEY
        base_url = BASE_URL
    else:
        model = gen.model if gen.model else default_model
        temperature = gen.temperature
        api_key = gen.api_key
        base_url = gen.base_url
    llm = Llm(
        model,
        max_tokens=_MAX_TOKENS,
        timeout=_TIMEOUT,
        max_retries=_MAX_RETRIES,
        api_key=api_key,
        base_url=base_url,
    )
    return llm, temperature
