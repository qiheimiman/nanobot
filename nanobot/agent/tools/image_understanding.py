"""Image understanding tool — calls a vision model to describe images."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.path_utils import resolve_workspace_path
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.schema import (
    ArraySchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.bus.events import (
    INBOUND_META_RUNTIME_CONTROL,
    RUNTIME_CONTROL_ACK,
    RUNTIME_CONTROL_IMAGE_UNDERSTANDING_RELOAD,
    InboundMessage,
)
from nanobot.config_base import Base
from nanobot.providers.image_understanding import (
    DEFAULT_VISION_PROMPT,
    describe_images,
    image_understanding_provider_configs,
    render_image_descriptions,
)
from nanobot.security.workspace_access import current_tool_workspace
from nanobot.security.workspace_policy import WorkspaceBoundaryError

if TYPE_CHECKING:
    from nanobot.agent.tools.context import ToolContext
    from nanobot.bus.queue import MessageBus


class ImageUnderstandingToolConfig(Base):
    """Image understanding tool configuration."""

    enabled: bool = False
    provider: str = "deepseek"  # Default to deepseek
    model: str = "deepseek-flash"
    prompt: str = DEFAULT_VISION_PROMPT  # Default prompt sent with each image


@tool_parameters(
    tool_parameters_schema(
        image_paths=ArraySchema(
            StringSchema(
                "Local path to an image file. Supports JPEG, PNG, GIF, WebP.",
                min_length=1,
            ),
            description="List of local image file paths to analyze.",
        ),
        prompt=StringSchema(
            "Prompt to send to the vision model. Defaults to the configured default prompt.",
        ),
        max_results=StringSchema(
            "Maximum number of results to return. Defaults to all.",
        ),
    )
)
class ImageUnderstandingTool(Tool):
    """Analyze images using a dedicated vision model and return text descriptions."""

    config_key = "image_understanding"

    @classmethod
    def config_cls(cls):
        return ImageUnderstandingToolConfig

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return ctx.config.image_understanding.enabled

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        return cls(
            workspace=ctx.workspace,
            config=ctx.config.image_understanding,
            provider_configs=ctx.image_understanding_provider_configs,
            restrict_to_workspace=bool(
                ctx.config.restrict_to_workspace or ctx.config.exec.sandbox
            ),
            sandbox_restricts_workspace=bool(ctx.config.exec.sandbox),
        )

    def __init__(
        self,
        *,
        workspace: str | Path,
        config: ImageUnderstandingToolConfig,
        provider_configs: dict[str, Any] | None = None,
        restrict_to_workspace: bool = False,
        sandbox_restricts_workspace: bool = False,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve(strict=False)
        self.config = config
        self.provider_configs = dict(provider_configs or {})
        self._restrict_to_workspace = restrict_to_workspace
        self._sandbox_restricts_workspace = sandbox_restricts_workspace

    @property
    def name(self) -> str:
        return "describe_image"

    @property
    def description(self) -> str:
        return (
            "Analyze images that are NOT part of the current message with a dedicated vision model "
            "and return text descriptions. Images attached to the current message are already "
            "described automatically and their descriptions are part of the conversation, so never "
            "call this tool for them. Use it for images outside the message: an older or newly "
            "generated file, or a path the user typed."
        )

    @property
    def read_only(self) -> bool:
        return True

    def _resolve_path(self, path: str) -> str:
        """Resolve and validate an image path against workspace restrictions.

        Relative paths are anchored to the current turn's project workspace (the
        workspace selected for this conversation), not to the agent home
        workspace, so images inside the selected project stay readable.
        """
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        access = current_tool_workspace(
            self.workspace,
            restrict_to_workspace=self._restrict_to_workspace,
            sandbox_restricts_workspace=self._sandbox_restricts_workspace,
        )
        allowed_root = access.allowed_root
        if allowed_root is None:
            return path
        resolved = resolve_workspace_path(
            path,
            workspace=access.project_path or self.workspace,
            allowed_dir=allowed_root,
            extra_allowed_dirs=[BUILTIN_SKILLS_DIR],
            include_media_dir=True,
        )
        return str(resolved)

    async def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        image_paths: list[str] | None = None,
        prompt: str | None = None,
        max_results: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Describe the requested images with the configured vision model."""
        raw_paths = [path for path in (image_paths or []) if path]
        if not raw_paths:
            return ToolResult.error("Error: image_paths is required")

        # Resolve and validate paths against workspace restrictions
        resolved_paths: list[str] = []
        for raw_path in raw_paths:
            try:
                resolved_paths.append(self._resolve_path(raw_path))
            except WorkspaceBoundaryError as exc:
                return ToolResult.error(str(exc))
            except OSError as exc:
                return ToolResult.error(
                    f"Error: could not resolve image path '{raw_path}': {exc}"
                )

        paths = resolved_paths
        max_count = _parse_positive_int(max_results)
        if max_count is not None:
            paths = paths[:max_count]

        provider_config = self.provider_configs.get(self.config.provider)
        if provider_config is None:
            return ToolResult.error(
                f"Error: provider '{self.config.provider}' is not configured"
            )

        effective_prompt = (prompt or self.config.prompt or DEFAULT_VISION_PROMPT).strip()
        descriptions = await describe_images(
            paths,
            provider_config=provider_config,
            model=self.config.model,
            prompt=effective_prompt or DEFAULT_VISION_PROMPT,
        )
        if not descriptions:
            return ToolResult.error(
                "Error: no image could be described (check the file paths and the "
                f"'{self.config.provider}' vision model configuration)"
            )
        return render_image_descriptions(descriptions)


def _parse_positive_int(value: str | None) -> int | None:
    """Parse a tool-supplied count, ignoring anything that is not a positive int."""
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


async def reload_image_understanding_tool(
    state: Any,
    registry: ToolRegistry,
) -> dict[str, Any]:
    """Apply the persisted image understanding configuration to the running agent."""
    try:
        from nanobot.config.loader import load_config, resolve_config_env_vars

        config = resolve_config_env_vars(load_config())
        tool_config = config.tools.image_understanding
        provider_configs = image_understanding_provider_configs(config)
    except Exception as exc:
        logger.warning("Image understanding hot reload could not read config: {}", exc)
        return {
            "ok": False,
            "message": "Could not reload image understanding config.",
            "requires_restart": True,
            "error": str(exc),
        }

    # The inbound-image pipeline reads this config off the live loop, so keep the
    # in-memory copy in sync with what was just written to disk.
    tools_config = getattr(state, "tools_config", None)
    if tools_config is not None:
        tools_config.image_understanding = tool_config
    state._image_understanding_provider_configs = provider_configs

    if tool_config.enabled:
        registry.register(
            ImageUnderstandingTool(  # pyright: ignore[reportAbstractUsage]
                workspace=state.workspace,
                config=tool_config,
                provider_configs=provider_configs,
                restrict_to_workspace=tools_config.restrict_to_workspace
                if tools_config is not None
                else False,
                sandbox_restricts_workspace=bool(
                    getattr(getattr(tools_config, "exec", None), "sandbox", False)
                    if tools_config is not None
                    else False
                ),
            )
        )
    else:
        registry.unregister("describe_image")

    logger.info(
        "Image understanding config reloaded: enabled={} provider={} model={}",
        tool_config.enabled,
        tool_config.provider,
        tool_config.model,
    )
    return {
        "ok": True,
        "message": "Image understanding settings applied without restarting nanobot.",
        "enabled": tool_config.enabled,
        "provider": tool_config.provider,
        "model": tool_config.model,
        "requires_restart": False,
    }


async def request_image_understanding_reload(
    bus: MessageBus,
    *,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Ask the running agent loop to refresh its image understanding tool."""
    loop = asyncio.get_running_loop()
    ack: asyncio.Future[dict[str, Any]] = loop.create_future()
    await bus.publish_inbound(
        InboundMessage(
            channel="system",
            sender_id="webui-settings",
            chat_id="runtime",
            content=RUNTIME_CONTROL_IMAGE_UNDERSTANDING_RELOAD,
            metadata={
                INBOUND_META_RUNTIME_CONTROL: RUNTIME_CONTROL_IMAGE_UNDERSTANDING_RELOAD,
                RUNTIME_CONTROL_ACK: ack,
            },
        )
    )
    try:
        result = await asyncio.wait_for(ack, timeout=timeout)
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "message": "Image understanding hot reload timed out.",
            "requires_restart": True,
        }
    if not isinstance(cast(object, result), dict):
        return {
            "ok": False,
            "message": "Image understanding hot reload returned an unexpected response.",
            "requires_restart": True,
        }
    return result


async def handle_runtime_control(
    state: Any,
    msg: InboundMessage,
    registry: ToolRegistry,
) -> bool:
    """Handle an in-process image understanding reload request."""
    metadata = msg.metadata
    if metadata.get(INBOUND_META_RUNTIME_CONTROL) != RUNTIME_CONTROL_IMAGE_UNDERSTANDING_RELOAD:
        return False

    ack = metadata.get(RUNTIME_CONTROL_ACK)
    try:
        result = await reload_image_understanding_tool(state, registry)
    except Exception as exc:
        logger.exception("Image understanding hot reload failed")
        result = {
            "ok": False,
            "message": "Image understanding hot reload failed.",
            "requires_restart": True,
            "error": str(exc),
        }
    if isinstance(ack, asyncio.Future) and not ack.done():
        cast(asyncio.Future[Any], ack).set_result(result)
    return True
