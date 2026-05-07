# Podcast Transcriber

Flask app that downloads podcast episodes from an RSS feed, transcribes them with
faster-whisper, and creates a ZIP containing transcripts plus episode metadata.

## Local Run

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://localhost:5000`.

## Live Streaming

To transcribe a live audio stream using Deepgram Nova-3, use:
`bash live_stream.sh` (Requires ffmpeg, websocat, and jq).

For a 4 GB GPU, these settings are a good balance:

```powershell
$env:WHISPER_MODEL="medium"
$env:WHISPER_BATCH_SIZE="8"
.\.venv\Scripts\python.exe app.py
```

## Koyeb

This project includes:

- `Procfile` for Koyeb/buildpack deployment
- `Dockerfile` for container deployment
- `$PORT` support
- `DATA_DIR` support for temporary or mounted storage

Recommended CPU/free settings:

```text
WHISPER_DEVICE=cpu
WHISPER_MODEL=small
WHISPER_COMPUTE_TYPE=int8
WHISPER_BATCH_SIZE=4
DATA_DIR=/tmp/transcripter-data
```

Recommended GPU settings:

```text
WHISPER_MODEL=medium
WHISPER_BATCH_SIZE=8
```

Live transcription is expensive on CPU. For free hosting, the practical workflow
is to transcribe locally and deploy only the completed transcripts/ZIP.
