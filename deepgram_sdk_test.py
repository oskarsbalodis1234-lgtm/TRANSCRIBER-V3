from deepgram import DeepgramClient
import os

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

AUDIO_URL = "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"

def main():
    try:
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        response = deepgram.listen.v1.media.transcribe_url(
            url=AUDIO_URL,
            model="nova-3",
            language="en",
            smart_format=True,
        )

        # Print the full response object
        print(response)

        # Or access the transcript directly
        print("\nTranscript:")
        print(response.results.channels[0].alternatives[0].transcript)

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()