# Conversation Bundle Module

The `viam:conversation-bundle` module provides these models for conversational workflows:

1. **`viam:conversation-bundle:text-to-speech`** - A generic service that synthesises speech via Google Cloud Text-to-Speech and plays it through an `audio_out` component.

---

## Model: `viam:conversation-bundle:text-to-speech`

**API:** `rdk:service:generic`

Synthesises speech using the [Google Cloud Text-to-Speech API](https://cloud.google.com/text-to-speech) and plays the resulting audio through an `rdk:component:audio_out` component.

### Prerequisites

- A Google Cloud project with the Text-to-Speech API enabled.
- A service account key (JSON) with access to the API.
- A configured `audio_out` component on the same machine.

### Configuration

```json
{
  "audio_out": "<string>",
  "google_credentials_json": { ... },
  "language_code": "<string>",
  "voice_name": "<string>"
}
```

| Name                      | Type   | Required | Description                                                                                                 |
| ------------------------- | ------ | -------- | ----------------------------------------------------------------------------------------------------------- |
| `audio_out`               | string | Yes      | Name of the `audio_out` component dependency used for playback.                                             |
| `google_credentials_json` | object | Yes      | Google Cloud service account credentials as a JSON object (not a string).                                   |
| `language_code`           | string | No       | BCP-47 language code. Defaults to `"en-US"`.                                                                |
| `voice_name`              | string | No       | Specific Google voice name (e.g. `"en-US-Neural2-F"`). If omitted, Google picks a default for the language. |

### Example Configuration

```json
{
  "audio_out": "ao",
  "google_credentials_json": {
    "type": "service_account",
    "project_id": "my-project",
    "private_key_id": "abc123",
    "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
    "client_email": "tts@my-project.iam.gserviceaccount.com",
    "client_id": "123456789",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token"
  },
  "language_code": "en-US",
  "voice_name": "en-US-Neural2-F"
}
```

### DoCommand

**`say`** — Synthesise and play text. The call blocks until playback completes.

```json
{"say": "Hello, your espresso is ready!"}
```

Returns:

```json
{"text": "Hello, your espresso is ready!"}
```

**`say_async`** — Queue text for playback and return immediately without waiting for synthesis or playback to finish. A background worker drains the queue and plays items sequentially. Audio is only sent to the speaker when no other speech (sync or async) is currently playing, so queued messages will never overlap with an in-flight `say` call. Returns an error if the async queue is full (capacity 64).

```json
{"say_async": "Hello, your espresso is ready!"}
```

Returns:

```json
{"queued": "Hello, your espresso is ready!"}
```
