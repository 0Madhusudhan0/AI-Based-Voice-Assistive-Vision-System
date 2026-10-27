import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel


class VoiceInput:
    def __init__(self, model_size="base", device="cpu"):
        print("[VOICE] Loading speech recognition model...")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8"
        )

        print("[VOICE] Speech recognition model loaded.")

    def record_audio(self, duration=5, sample_rate=16000):
        print("\n[VOICE] Listening... Speak now!")

        try:
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            return np.squeeze(audio)

        except Exception as error:
            print(f"[VOICE ERROR] Recording failed: {error}")
            return None

    def transcribe(self, audio, sample_rate=16000):
        if audio is None or len(audio) == 0:
            return ""

        print("[VOICE] Recognizing speech...")

        try:
            segments, _ = self.model.transcribe(
                audio,
                language="en",
                vad_filter=True,
                beam_size=1
            )

            text = " ".join(
                segment.text.strip() for segment in segments
            ).strip()

            if text:
                print(f"[VOICE] You said: {text}")
            else:
                print("[VOICE] No speech recognized.")

            return text

        except Exception as error:
            print(f"[VOICE ERROR] Transcription failed: {error}")
            return ""

    def listen(self, duration=5):
        audio = self.record_audio(duration=duration)

        if audio is None:
            return ""

        return self.transcribe(audio)