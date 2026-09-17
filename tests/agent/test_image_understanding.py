import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop, TurnContext, TurnKind
from nanobot.agent.tools.image_understanding import (
    ImageUnderstandingTool,
    ImageUnderstandingToolConfig,
    handle_runtime_control,
    reload_image_understanding_tool,
)
from nanobot.bus.events import (
    INBOUND_META_RUNTIME_CONTROL,
    RUNTIME_CONTROL_ACK,
    RUNTIME_CONTROL_IMAGE_UNDERSTANDING_RELOAD,
    InboundMessage,
)
from nanobot.bus.outbound_events import ProgressEvent
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config, ProviderConfig, ToolsConfig
from nanobot.events import EventSink
from nanobot.providers.base import LLMResponse
from nanobot.providers.image_understanding import (
    ImageDescription,
    image_understanding_provider_configs,
)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yF9kAAAAASUVORK5CYII="
)


def _make_loop(
    workspace: Path,
    *,
    image_understanding: ImageUnderstandingToolConfig | None = None,
    provider_configs: dict[str, ProviderConfig] | None = None,
) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content="ok"))
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=workspace,
        model="test-model",
        tools_config=ToolsConfig(
            image_understanding=image_understanding or ImageUnderstandingToolConfig(),
        ),
        image_understanding_provider_configs=provider_configs or {},
    )


def _turn_context(loop: AgentLoop, msg: InboundMessage) -> TurnContext:
    return TurnContext(
        msg=msg,
        session_key=f"{msg.channel}:{msg.chat_id}",
        turn_id="turn-1",
        runtime=loop.llm_runtime(),
        kind=TurnKind.USER,
        delivery=loop.turn_delivery_factory.create(msg, f"{msg.channel}:{msg.chat_id}"),
    )


def _progress_sink(events: list[ProgressEvent]) -> EventSink:
    """Capture the progress breadcrumbs one turn publishes."""

    async def publish(event: object) -> None:
        if isinstance(event, ProgressEvent):
            events.append(event)

    return EventSink(publish, accepts_type=lambda _event_type: True)


def _write_png(path: Path) -> Path:
    path.write_bytes(PNG_BYTES)
    return path


@pytest.mark.asyncio
async def test_inbound_image_is_described_before_the_chat_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = _write_png(tmp_path / "shot.png")
    loop = _make_loop(
        workspace,
        image_understanding=ImageUnderstandingToolConfig(
            enabled=True,
            provider="deepseek",
            model="deepseek-flash",
            prompt="what is in this image",
        ),
        provider_configs={"deepseek": ProviderConfig(api_key="key")},
    )
    captured: dict[str, object] = {}

    async def fake_describe(
        image_paths: list[str],
        *,
        provider_config: ProviderConfig,
        model: str,
        prompt: str,
    ) -> list[ImageDescription]:
        captured["paths"] = list(image_paths)
        captured["model"] = model
        captured["prompt"] = prompt
        captured["api_key"] = provider_config.api_key
        return [ImageDescription(path=str(image_paths[0]), text="a red square")]

    monkeypatch.setattr("nanobot.agent.loop.describe_images", fake_describe)

    msg = InboundMessage(
        channel="websocket",
        sender_id="u",
        chat_id="c",
        content="look at this",
        media=[str(image)],
    )
    ctx = _turn_context(loop, msg)

    await loop._restore_turn(ctx)

    assert captured["paths"] == [str(image.resolve())]
    assert captured["model"] == "deepseek-flash"
    assert captured["prompt"] == "what is in this image"
    assert captured["api_key"] == "key"

    assert isinstance(ctx.msg.content, str)
    assert "look at this" in ctx.msg.content
    assert "a red square" in ctx.msg.content
    assert f"[image: {image.resolve()}]" in ctx.msg.content
    # The block tells the chat model the attachment is already covered.
    assert "do not call describe_image" in ctx.msg.content
    # The chat model must not be asked to read the image itself.
    assert ctx.msg.media == []

    current = loop.context.build_current_message(ctx.msg.content, media=None)
    assert isinstance(current["content"], str)
    assert "a red square" in current["content"]


@pytest.mark.asyncio
async def test_disabled_image_understanding_keeps_native_image_blocks(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = _write_png(tmp_path / "shot.png")
    loop = _make_loop(
        workspace,
        provider_configs={"deepseek": ProviderConfig(api_key="key")},
    )
    msg = InboundMessage(
        channel="websocket",
        sender_id="u",
        chat_id="c",
        content="look at this",
        media=[str(image)],
    )
    ctx = _turn_context(loop, msg)

    await loop._restore_turn(ctx)

    assert ctx.msg.media == [str(image.resolve())]
    current = loop.context.build_current_message(ctx.msg.content, media=ctx.msg.media)
    blocks = current["content"]
    assert isinstance(blocks, list)
    assert any(block.get("type") == "image_url" for block in blocks)


@pytest.mark.asyncio
async def test_vision_failure_falls_back_to_native_image_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = _write_png(tmp_path / "shot.png")
    loop = _make_loop(
        workspace,
        image_understanding=ImageUnderstandingToolConfig(enabled=True, provider="deepseek"),
        provider_configs={"deepseek": ProviderConfig(api_key="key")},
    )

    async def failing_describe(*args: object, **kwargs: object) -> list[ImageDescription]:
        return []

    monkeypatch.setattr("nanobot.agent.loop.describe_images", failing_describe)

    msg = InboundMessage(
        channel="websocket",
        sender_id="u",
        chat_id="c",
        content="look at this",
        media=[str(image)],
    )
    ctx = _turn_context(loop, msg)

    await loop._restore_turn(ctx)

    assert ctx.msg.content == "look at this"
    assert ctx.msg.media == [str(image.resolve())]


@pytest.mark.asyncio
async def test_unconfigured_provider_keeps_native_image_blocks(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = _write_png(tmp_path / "shot.png")
    loop = _make_loop(
        workspace,
        image_understanding=ImageUnderstandingToolConfig(enabled=True, provider="deepseek"),
        provider_configs={},
    )
    msg = InboundMessage(
        channel="websocket",
        sender_id="u",
        chat_id="c",
        content="look at this",
        media=[str(image)],
    )
    ctx = _turn_context(loop, msg)

    await loop._restore_turn(ctx)

    assert ctx.msg.media == [str(image.resolve())]


@pytest.mark.asyncio
async def test_automatic_vision_call_publishes_tool_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WebUI must see a tool trace while images are described before the model call."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = _write_png(tmp_path / "shot.png")
    loop = _make_loop(
        workspace,
        image_understanding=ImageUnderstandingToolConfig(
            enabled=True,
            provider="deepseek",
            model="deepseek-flash",
            prompt="count the shapes",
        ),
        provider_configs={"deepseek": ProviderConfig(api_key="key")},
    )

    sent: dict[str, str] = {}

    async def fake_describe(
        image_paths: list[str],
        *,
        prompt: str,
        **kwargs: object,
    ) -> list[ImageDescription]:
        sent["prompt"] = prompt
        return [ImageDescription(path=str(image_paths[0]), text="a red square")]

    monkeypatch.setattr("nanobot.agent.loop.describe_images", fake_describe)

    msg = InboundMessage(
        channel="websocket",
        sender_id="u",
        chat_id="c",
        content="look at this",
        media=[str(image)],
    )
    ctx = _turn_context(loop, msg)
    events: list[ProgressEvent] = []
    ctx.events = _progress_sink(events)

    await loop._restore_turn(ctx)

    assert len(events) == 2
    start, end = events
    assert start.tool_hint is True
    assert start.tool_events is not None
    start_payload = start.tool_events[0]
    assert start_payload["phase"] == "start"
    assert start_payload["name"] == "describe_image"
    # The hint mirrors the tool event so chat clients group the breadcrumb with its
    # event and show the effective prompt instead of a bare tool name.
    assert start.content == (
        f'describe_image({json.dumps(start_payload["arguments"], ensure_ascii=False)})'
    )
    assert str(image.resolve()) in start.content
    # The configured prompt reaches the vision model and the trace.
    assert sent["prompt"] == "count the shapes"
    assert start_payload["arguments"] == {
        "image_paths": [str(image.resolve())],
        "prompt": "count the shapes",
    }
    assert '"prompt": "count the shapes"' in start.content
    assert end.tool_hint is False
    assert end.tool_events is not None
    end_payload = end.tool_events[0]
    assert end_payload["phase"] == "end"
    assert end_payload["call_id"] == start_payload["call_id"]
    assert "described 1 image(s)" in str(end_payload["result"])


@pytest.mark.asyncio
async def test_vision_failure_is_reported_as_an_error_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = _write_png(tmp_path / "shot.png")
    loop = _make_loop(
        workspace,
        image_understanding=ImageUnderstandingToolConfig(enabled=True, provider="deepseek"),
        provider_configs={"deepseek": ProviderConfig(api_key="key")},
    )

    async def failing_describe(*args: object, **kwargs: object) -> list[ImageDescription]:
        return []

    monkeypatch.setattr("nanobot.agent.loop.describe_images", failing_describe)

    msg = InboundMessage(
        channel="websocket",
        sender_id="u",
        chat_id="c",
        content="look at this",
        media=[str(image)],
    )
    ctx = _turn_context(loop, msg)
    events: list[ProgressEvent] = []
    ctx.events = _progress_sink(events)

    await loop._restore_turn(ctx)

    assert len(events) == 2
    end_payload = events[1].tool_events[0]
    assert end_payload["phase"] == "error"
    assert end_payload["error"]
    # The failed description still falls back to native image blocks.
    assert ctx.msg.media == [str(image.resolve())]


@pytest.mark.asyncio
async def test_disabled_image_understanding_publishes_no_progress(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = _write_png(tmp_path / "shot.png")
    loop = _make_loop(
        workspace,
        provider_configs={"deepseek": ProviderConfig(api_key="key")},
    )
    msg = InboundMessage(
        channel="websocket",
        sender_id="u",
        chat_id="c",
        content="look at this",
        media=[str(image)],
    )
    ctx = _turn_context(loop, msg)
    events: list[ProgressEvent] = []
    ctx.events = _progress_sink(events)

    await loop._restore_turn(ctx)

    assert events == []


@pytest.mark.asyncio
async def test_documents_and_images_are_split_before_description(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = _write_png(tmp_path / "shot.png")
    document = tmp_path / "report.csv"
    document.write_text("name,value", encoding="utf-8")
    loop = _make_loop(
        workspace,
        image_understanding=ImageUnderstandingToolConfig(enabled=True, provider="deepseek"),
        provider_configs={"deepseek": ProviderConfig(api_key="key")},
    )
    described: list[str] = []

    async def fake_describe(image_paths: list[str], **kwargs: object) -> list[ImageDescription]:
        described.extend(image_paths)
        return [ImageDescription(path=str(image_paths[0]), text="chart of values")]

    monkeypatch.setattr("nanobot.agent.loop.describe_images", fake_describe)

    msg = InboundMessage(
        channel="websocket",
        sender_id="u",
        chat_id="c",
        content="review these",
        media=[str(document), str(image)],
    )
    ctx = _turn_context(loop, msg)

    await loop._restore_turn(ctx)

    assert described == [str(image.resolve())]
    assert "chart of values" in ctx.msg.content
    assert f"[Attachment: {document.resolve()}]" in ctx.msg.content
    assert ctx.msg.media == []


@pytest.mark.asyncio
async def test_describe_image_tool_uses_shared_vision_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = _write_png(tmp_path / "shot.png")
    tool = ImageUnderstandingTool(
        workspace=workspace,
        config=ImageUnderstandingToolConfig(
            enabled=True,
            provider="deepseek",
            model="deepseek-flash",
            prompt="describe",
        ),
        provider_configs={"deepseek": ProviderConfig(api_key="key")},
    )

    async def fake_describe(
        image_paths: list[str],
        *,
        provider_config: ProviderConfig,
        model: str,
        prompt: str,
    ) -> list[ImageDescription]:
        assert model == "deepseek-flash"
        assert prompt == "name the colors"
        return [ImageDescription(path=str(image_paths[0]), text="a blue circle")]

    monkeypatch.setattr(
        "nanobot.agent.tools.image_understanding.describe_images",
        fake_describe,
    )

    result = await tool.execute(image_paths=[str(image)], prompt="name the colors")

    assert "a blue circle" in result
    assert f"[image: {image}]" in result


@pytest.mark.asyncio
async def test_describe_image_tool_reports_unconfigured_provider(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = _write_png(tmp_path / "shot.png")
    tool = ImageUnderstandingTool(
        workspace=workspace,
        config=ImageUnderstandingToolConfig(enabled=True, provider="deepseek"),
        provider_configs={},
    )

    result = await tool.execute(image_paths=[str(image)])

    assert result.is_error is True
    assert "not configured" in result


def test_image_understanding_provider_configs_covers_llm_providers() -> None:
    config = Config()

    providers = image_understanding_provider_configs(config)

    assert "deepseek" in providers
    assert "openai" in providers
    assert all(isinstance(value, ProviderConfig) for value in providers.values())


@pytest.mark.asyncio
async def test_reload_applies_saved_config_to_running_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fresh = Config()
    fresh.tools.image_understanding.enabled = True
    fresh.tools.image_understanding.provider = "deepseek"
    fresh.tools.image_understanding.model = "deepseek-flash"
    fresh.providers.deepseek.api_key = "key"
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda *a, **k: fresh)
    monkeypatch.setattr("nanobot.config.loader.resolve_config_env_vars", lambda config: config)

    loop = _make_loop(workspace)
    assert loop.tools_config.image_understanding.enabled is False

    result = await reload_image_understanding_tool(loop, loop.tools)

    assert result["ok"] is True
    assert result["requires_restart"] is False
    assert loop.tools_config.image_understanding.enabled is True
    assert loop._image_understanding_provider_configs.get("deepseek") is not None
    assert loop.tools.get("describe_image") is not None

    fresh.tools.image_understanding.enabled = False
    await reload_image_understanding_tool(loop, loop.tools)

    assert loop.tools.get("describe_image") is None
    assert loop.tools_config.image_understanding.enabled is False


@pytest.mark.asyncio
async def test_runtime_control_message_triggers_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fresh = Config()
    fresh.tools.image_understanding.enabled = True
    fresh.providers.deepseek.api_key = "key"
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda *a, **k: fresh)
    monkeypatch.setattr("nanobot.config.loader.resolve_config_env_vars", lambda config: config)
    loop = _make_loop(workspace)

    ack: asyncio.Future = asyncio.get_running_loop().create_future()
    handled = await handle_runtime_control(
        loop,
        InboundMessage(
            channel="system",
            sender_id="webui-settings",
            chat_id="runtime",
            content=RUNTIME_CONTROL_IMAGE_UNDERSTANDING_RELOAD,
            metadata={
                INBOUND_META_RUNTIME_CONTROL: RUNTIME_CONTROL_IMAGE_UNDERSTANDING_RELOAD,
                RUNTIME_CONTROL_ACK: ack,
            },
        ),
        loop.tools,
    )

    assert handled is True
    assert ack.done()
    assert ack.result()["ok"] is True
    assert loop.tools_config.image_understanding.enabled is True

    unrelated = InboundMessage(channel="system", sender_id="s", chat_id="c", content="x")
    assert await handle_runtime_control(loop, unrelated, loop.tools) is False
