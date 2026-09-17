---
name: image-understanding
description: Images attached to a conversation are described by a dedicated vision model before the chat model sees them. Never call describe_image for an attachment in the current message; use that tool only for images outside it.
---

# Image Understanding

When image understanding is enabled in settings, nanobot routes images the user attaches to a message
through the configured vision model first. The chat model receives the resulting text descriptions
instead of raw image blocks, so a text-only chat model can still answer questions about the image.

## What you see

An automatically described attachment appears in the user turn as:

```
[image: /path/to/image.png]
<description produced by the vision model>
(the images above are already described; do not call describe_image for them)
```

Treat that description as the content of the user's image and answer from it. No tool call is needed:
the `[image: ...]` line is only a reference to where the file lives, not a request to read it.

## When to use `describe_image`

Call the `describe_image` tool yourself only when the image was **not** described automatically, for example:

- The user names an image path that is not part of the current message (e.g. an older or newly created file)
- You generated or edited an image and need to inspect the result
- The user asks about a specific detail that the automatic description does not cover — pass a focused
  `prompt` (e.g. "read the text in this screenshot" or "list the values in the table")

Never call it for an image that arrived with the current message: it was already sent to the vision model
and the description is in the turn, so a second call only sends the same image again.

```
describe_image(image_paths=["/path/to/image1.png", "/path/to/image2.jpg"], prompt="描述这张图片")
```

The `prompt` parameter is optional — it defaults to the configured prompt (usually "描述这张图片").

## Tips

- Pass all image paths at once for batch processing
- The tool returns one description block per image
- If automatic description is unavailable (provider not configured or the request failed), the raw image
  blocks are still sent to the chat model, and the `[image: path]` reference tells you where the file is
- The vision model is configured separately from the chat model
