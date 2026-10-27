from src.speech.voice_input import VoiceInput


def main():
    voice = VoiceInput(model_size="base")

    print("\nSpeak a question when prompted.")
    question = voice.listen(duration=5)

    print("\n" + "=" * 40)
    print("Recognized question:", question)
    print("=" * 40)


if __name__ == "__main__":
    main()