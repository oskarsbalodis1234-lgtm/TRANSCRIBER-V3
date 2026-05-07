#!/bin/bash

# Deepgram Nova-3 Live Streaming Transcription
# Requires: ffmpeg, websocat, jq

DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY}
deepgram_api_endpoint="wss://api.deepgram.com/v1/listen?endpointing=false&language=en&model=nova-3&encoding=linear16&sample_rate=16000"

# Note: This is an English stream, update accordingly for other languages
stream_url="https://playerservices.streamtheworld.com/api/livestream-redirect/CSPANRADIOAAC.aac"

ffmpeg -loglevel error -i "$stream_url" -f s16le -ar 16000 -ac 1 - | \
  websocat -v -H "Authorization: Token $DEEPGRAM_API_KEY" \
    -b --base64-text "$deepgram_api_endpoint" | \
  {
    while read -r msg; do
      if [[ -n "$msg" ]]; then
        json=$(echo "$msg" | base64 -d)
        is_final=$(echo "$json" | jq -r '.is_final // empty')
        transcript=$(echo "$json" | jq -r '.channel?.alternatives?[0]?.transcript? // empty')
        if [[ -n "$transcript" ]]; then
          echo -e "${is_final:+[ FINAL ]}${is_final:-[Interim]} $transcript"
        fi
      fi
    done
  }