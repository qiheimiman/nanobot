"""Vision (image understanding) helpers.

Both the ``describe_image`` tool and the inbound-message pipeline need to turn
local image files into text with a vision-capable model. The request logic lives
here so every caller behaves the same way (format detection, timeouts, and
per-image error handling).
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from loguru import logger

from nanobot.config.schema import Config, ProviderConfig
from nanobot.utils.helpers import detect_image_mime

DEFAULT_VISION_PROMPT = "描述这张图片"
DEFAULT_VISION_TIMEOUT_S = 120.0
_MAX_VISION_TOKENS = 2048


class ImageUnderstandingError(RuntimeError):
    """Raised when an image cannot be sent to a vision model."""


def image_understanding_provider_configs(config: Config) -> dict[str, ProviderConfig]:
    """Return every configured LLM provider usable for image understanding.

    Unlike image generation, any vision-capable chat provider works here
    (DeepSeek, OpenAI, Anthropic, ...), so the whole provider config is exposed.
    """
    providers_cfg = config.providers
    return {
        name: value
        for name in dir(providers_cfg)
        if not name.startswith("_")
        and isinstance(
            value := getattr(providers_cfg, name, None),
            ProviderConfig,
        )
    }


@dataclass(frozen=True, slots=True)
class ImageDescription:
    """One image plus the vision model's description of it."""

    path: str
    text: str

    def render(self) -> str:
        """Render as model-visible text that keeps the source path reference."""
        return f"[image: {self.path}]\n{self.text}"


def image_data_url(path: str | Path) -> str:
    """Return a ``data:`` URL for a supported local image file."""
    p = Path(path).expanduser()
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ImageUnderstandingError(f"cannot read image {p}: {exc}") from exc
    mime = detect_image_mime(raw)
    if mime is None:
        raise ImageUnderstandingError(f"unsupported image format: {p}")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def build_vision_client(provider_config: ProviderConfig) -> Any:
    """Build an OpenAI-compatible async client for a vision provider."""
    from openai import AsyncOpenAI

    kwargs: dict[str, Any] = {
        "api_key": provider_config.api_key or "not-needed",
        "timeout": DEFAULT_VISION_TIMEOUT_S,
        "max_retries": 1,
    }
    if provider_config.api_base:
        kwargs["base_url"] = provider_config.api_base
    if provider_config.extra_headers:
        kwargs["default_headers"] = dict(provider_config.extra_headers)
    if provider_config.extra_query:
        kwargs["default_query"] = dict(provider_config.extra_query)
    return AsyncOpenAI(**kwargs)


async def describe_images(
    image_paths: Sequence[str | Path],
    *,
    provider_config: ProviderConfig,
    model: str,
    prompt: str = DEFAULT_VISION_PROMPT,
) -> list[ImageDescription]:
    """Describe every image in *image_paths* with the configured vision model.

    Unreadable images and failed requests are logged and skipped so a single
    broken attachment never fails the caller's turn.
    """
    client = build_vision_client(provider_config)
    try:
        return await describe_images_with_client(
            client,
            image_paths,
            model=model,
            prompt=prompt,
        )
    finally:
        await _close_client(client)


async def describe_images_with_client(
    client: Any,
    image_paths: Sequence[str | Path],
    *,
    model: str,
    prompt: str = DEFAULT_VISION_PROMPT,
) -> list[ImageDescription]:
    """Describe images with an already-built OpenAI-compatible *client*."""
    descriptions: list[ImageDescription] = []
    for path in image_paths:
        try:
            data_url = image_data_url(path)
        except ImageUnderstandingError as exc:
            logger.warning("Skipping image for vision model: {}", exc)
            continue
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=_MAX_VISION_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001 - callers fall back per image
            logger.warning("Vision model failed for {}: {}", path, exc)
            continue
        text = _completion_text(completion)
        if not text:
            logger.warning("Vision model returned no text for {}", path)
            continue
        descriptions.append(ImageDescription(path=str(path), text=text))
    return descriptions


def render_image_descriptions(descriptions: Iterable[ImageDescription]) -> str:
    """Render descriptions as model-visible text blocks."""
    return "\n\n".join(description.render() for description in descriptions)


async def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if close is None:
        return
    try:
        await close()
    except Exception as exc:  # noqa: BLE001 - closing must never mask a result
        logger.debug("Failed to close vision client: {}", exc)


def _completion_text(completion: Any) -> str:
    raw_choices = getattr(completion, "choices", None)
    choices = cast(list[Any], raw_choices) if raw_choices else []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = _text_block_parts(cast(list[Any], content))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _text_block_parts(content: list[Any]) -> list[str]:
    """Collect ``text`` fields from text content blocks, ignoring other shapes."""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_data = cast(dict[str, Any], block)
        if block_data.get("type") != "text":
            continue
        text = block_data.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return parts
