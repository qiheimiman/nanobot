# Image Understanding

nanobot can route images that arrive in a conversation through a dedicated vision model, so a text-only
chat model can still answer questions about them.

When image understanding is enabled and a message carries an image, nanobot describes the image with the
configured vision model first. The chat model then receives the description as text instead of raw image
blocks, which keeps small or text-only models usable for "what is in this screenshot?" style questions.

The feature is disabled by default. Open **Settings → Image**, choose a vision-capable provider and model
under **Image understanding**, enable it, and save. The running gateway applies the change immediately.
If that screen is not available in your installed version, use the manual config below.

## Quick Setup

**WebUI**

1. Add the vision provider credential under **Settings → Models** if it is not already configured.
2. Open **Settings → Image**.
3. Under **Image understanding**, select the provider and model, optionally edit the prompt, then enable it.
4. Save and send an image. If the gateway cannot apply the change live, WebUI will prompt you to restart it.

**Manual config**

Any provider that accepts OpenAI-compatible `chat/completions` requests with `image_url` content works.
This snippet uses DeepSeek; replace `provider` and `model` with any vision-capable provider you have
configured.

```json
{
  "providers": {
    "deepseek": {
      "apiKey": "${DEEPSEEK_API_KEY}"
    }
  },
  "tools": {
    "imageUnderstanding": {
      "enabled": true,
      "provider": "deepseek",
      "model": "deepseek-flash",
      "prompt": "描述这张图片"
    }
  }
}
```

> [!TIP]
> Prefer environment variables for API keys. nanobot resolves `${VAR_NAME}` values from the environment at startup.

## What the Chat Model Sees

For each described image the chat model receives a text block in the user turn:

```
[image: /path/to/screenshot.png]
<description produced by the vision model>
(the images above are already described; do not call describe_image for them)
```

The same block is persisted with the user message, so follow-up questions keep working without calling the
vision model again. The original message text, `[Attachment: ...]` references for non-image files, and the
image path are preserved. The trailing note tells the chat model that the attachment is already covered, so
it answers from the description instead of calling `describe_image` for the same file again.

If the vision model is unavailable — the provider is not configured, the request fails, or the file is not a
supported image — nanobot falls back to sending the raw image blocks to the chat model, exactly as it did
before the feature was enabled.

While the vision request runs, the conversation shows a tool trace for it, closing when the description is
ready. The trace line is the same `describe_image(...)` form the chat model's own tool calls produce, with the
real arguments, for example:

```
describe_image({"image_paths": ["/path/to/screenshot.png"], "prompt": "描述这张图片"})
```

That breadcrumb appears for the automatic vision call just like any other tool call, so the pause before the
chat model answers is explained in the UI, and it spells out the prompt the running process used — which makes
it obvious whether the prompt from **Settings → Image** reached the vision model.

## Manual Tool Use

The `describe_image` tool remains available so the agent can describe images that were not attached to the
current message, such as a path the user typed, an older file, or a generated image it wants to inspect. It is
never needed for an attachment in the current message, which was already described automatically:

```json
{ "image_paths": ["/path/to/image.png"], "prompt": "read the text in this screenshot" }
```

Pass several paths to batch them, and use `prompt` to zoom in on a specific question instead of the default
description.

## Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tools.imageUnderstanding.enabled` | boolean | `false` | Describe inbound images with the vision model and register the `describe_image` tool |
| `tools.imageUnderstanding.provider` | string | `"deepseek"` | Provider used for vision requests. Reuses credentials from the `providers` section |
| `tools.imageUnderstanding.model` | string | `"deepseek-flash"` | Vision-capable model name for that provider |
| `tools.imageUnderstanding.prompt` | string | `"描述这张图片"` | Prompt sent with each image; empty falls back to the default. The effective value is shown in the tool trace of each automatic vision call |

## Provider Notes

- Image understanding reuses the credentials saved under **Settings → Models**; it does not keep its own
  API keys.
- Requests are sent as OpenAI-compatible chat completions with base64 `image_url` blocks, using the
  provider's `apiBase`, `extraHeaders`, and `extraQuery` settings.
- Supported image formats are PNG, JPEG, GIF, and WebP, detected from file bytes rather than the extension.
- DeepSeek documents inline base64 images up to 48 MiB per request and 32 MiB per image; it serves vision
  requests through `https://api.deepseek.com` with the `deepseek-flash` model.

## Troubleshooting

| Symptom | Check |
|---|---|
| Images still reach the chat model | `tools.imageUnderstanding.enabled` is `true`, the selected provider is configured, and the gateway restarted or accepted the hot reload |
| `describe_image` returns "provider is not configured" | Save the credential under **Settings → Models** for the provider named in `tools.imageUnderstanding.provider` |
| Description never changes when the prompt is edited | The vision trace in the conversation prints the `prompt` the running process used. A stale value means the gateway has not reloaded the setting — restart it |
| An attached image is described twice (automatic trace plus a model `describe_image` call) | The automatic block ends with "do not call describe_image for them", but the chat model can still ignore it and call the tool for the path it sees in `[image: ...]`. The second call wastes a request but does not change the answer — ask about the image again without repeating its path to keep the turn short |
| Description is missing for one image out of several | Only that image failed; the remaining images were still described and the failed one is passed through as a raw image block |
| Long replies about simple images | Set a shorter `tools.imageUnderstanding.prompt`, for example `"Describe this image in one sentence."` |

## Related

- [Image Generation](./image-generation.md) — the reverse direction: creating images
- [Configuration](./configuration.md#tools) — the full `tools` config reference
