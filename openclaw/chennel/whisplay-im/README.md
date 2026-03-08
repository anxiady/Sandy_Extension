# whisplay-im Bridge

## Overview

Use `whisplay-im` to connect OpenClaw to a Whisplay device as a pure IM bridge.
The device pushes ASR text into the bridge. OpenClaw polls for new messages and
sends replies back for TTS playback.

## Inputs to collect

- Bridge base URL (host/port)
- Auth token for `Authorization: Bearer <token>`
- Optional `waitSec` for long-polling

## Actions

### Send device ASR text (inbox)

This api is called by the device to push ASR text into the bridge.

```bash
curl -X POST \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","messages":[{"role":"user","content":"hello"}]}' \
  http://<device-host>:18888/whisplay-im/inbox
```

### Poll for a new message

This api is called by OpenClaw to poll for new messages from the device. It supports long-polling with `waitSec` parameter.

```bash
curl -X GET \
  -H "Authorization: Bearer <token>" \
  "http://<device-host>:18888/whisplay-im/poll?waitSec=30"
```

### Send reply to device

This api is called by OpenClaw to send a reply back to the device for TTS playback. The `emoji` field is optional and can be used to control the device display.

```bash
curl -X POST \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"reply":"Hello from OpenClaw","emoji":"🦞"}' \
  http://<device-host>:18888/whisplay-im/send
```

## Notes

- `messages` is optional; use it for context routing.
- `poll` returns an empty payload if no messages are available.
- `send` supports optional `emoji` to control the device display.

