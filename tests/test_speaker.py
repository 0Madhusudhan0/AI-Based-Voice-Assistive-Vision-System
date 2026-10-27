
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from speech.speaker import Speaker


class TestSpeakerQueue(unittest.TestCase):

    def test_high_priority_replaces_pending_medium_alert(self):
        speaking_started = threading.Event()
        allow_speech_to_finish = threading.Event()
        spoken_messages = []

        def fake_speak(message):
            spoken_messages.append(message)

            if message == "currently speaking":
                speaking_started.set()
                if not allow_speech_to_finish.wait(timeout=5):
                    raise TimeoutError("Test speech release timed out.")

        with patch.object(Speaker, "_speak_windows", side_effect=fake_speak):
            speaker = Speaker(max_pending=2)

            try:
                speaker.speak("currently speaking", "medium")
                self.assertTrue(speaking_started.wait(timeout=3))

                speaker.speak("medium one", "medium")
                speaker.speak("medium two", "medium")
                speaker.speak("urgent alert", "high")

            finally:
                allow_speech_to_finish.set()
                speaker.close()

        self.assertEqual(
            spoken_messages,
            ["currently speaking", "urgent alert", "medium two"],
        )

    def test_equal_priority_preserves_arrival_order(self):
        speaking_started = threading.Event()
        allow_speech_to_finish = threading.Event()
        spoken_messages = []

        def fake_speak(message):
            spoken_messages.append(message)

            if message == "currently speaking":
                speaking_started.set()
                if not allow_speech_to_finish.wait(timeout=5):
                    raise TimeoutError("Test speech release timed out.")

        with patch.object(Speaker, "_speak_windows", side_effect=fake_speak):
            speaker = Speaker(max_pending=3)

            try:
                speaker.speak("currently speaking", "medium")
                self.assertTrue(speaking_started.wait(timeout=3))

                speaker.speak("first", "medium")
                speaker.speak("second", "medium")
                speaker.speak("third", "medium")

            finally:
                allow_speech_to_finish.set()
                speaker.close()

        self.assertEqual(
            spoken_messages,
            ["currently speaking", "first", "second", "third"],
        )

    def test_invalid_priority_is_rejected(self):
        with patch.object(Speaker, "_speak_windows"):
            speaker = Speaker()

            try:
                with self.assertRaises(ValueError):
                    speaker.speak("test message", "low")
            finally:
                speaker.close()


if __name__ == "__main__":
    unittest.main()
