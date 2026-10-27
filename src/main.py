import time
import threading

import cv2

from camera.webcam import Webcam
from detection.detector import ObjectDetector
from speech.speaker import Speaker
from speech.voice_input import VoiceInput
from llm.assistant import LlamaAssistant


# Objects that SightSpeak AI should announce automatically.
ANNOUNCE_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "chair",
    "couch",
    "dining table",
    "bottle",
    "cup",
    "backpack",
    "handbag",
    "suitcase",
    "cell phone",
    "laptop",
}


def describe_scene(detections):
    """Describe detected objects, positions, and estimated proximity."""

    objects = []
    seen = set()

    for detection in detections:
        name = detection.get("class_name")
        position = detection.get("position", "center")
        proximity = detection.get("proximity", "unknown")

        key = (name, position)

        if name in ANNOUNCE_CLASSES and key not in seen:
            seen.add(key)

            if proximity in {"near", "medium", "far"}:
                objects.append(
                    f"{name} on your {position}, appears {proximity}"
                )
            else:
                objects.append(f"{name} on your {position}")

        if len(objects) >= 4:
            break

    if not objects:
        return "I could not identify any familiar objects."

    return "I can see " + ", ".join(objects) + "."


def question_worker(
    voice_input,
    assistant,
    speaker,
    detections,
    question_active,
):
    """Listen, process a question, and wait for the answer to finish."""

    try:
        print("\n[VOICE] Speak your question after the listening prompt.")

        # Record and transcribe the user's question.
        question = voice_input.listen(duration=5)

        if not question:
            print("[VOICE] No question recognized.")
            return

        question = question.strip()

        # Keep exit commands from being sent to the LLM.
        if question.lower().strip(" .!?") in {
            "exit",
            "quit",
            "goodbye",
            "bye",
        }:
            print("[VOICE] Press Q to exit SightSpeak AI.")
            return

        print(f"\n[QUESTION] {question}")
        print("[LLAMA] Thinking...")

        # Ask Llama using the question and the saved detections.
        answer = assistant.ask(question, detections)

        if not answer:
            print("[LLAMA] No answer was returned.")
            return

        print(f"[ANSWER] {answer}\n")

        # Give the answer priority over routine announcements.
        speaker.speak(answer, priority="high")

        # Wait until the answer has finished playing.
        # This requires the updated Speaker class with wait_until_idle().
        finished = speaker.wait_until_idle(timeout=120)

        if not finished:
            print(
                "[VOICE] Timed out waiting for speech playback to finish."
            )

    except Exception as error:
        print(f"[VOICE/LLAMA ERROR] {error}")

    finally:
        # Release the question lock even if an error occurs.
        question_active.clear()
        print("[VOICE] Returning to camera controls.\n")


def main():
    # DroidCam stream URL.
    DROIDCAM_URL = "http://192.168.1.68:4747/video"
    # DROIDCAM_URL = "http://10.146.163.154:4747/video"

    camera = Webcam(stream_url=DROIDCAM_URL)
    detector = ObjectDetector()
    speaker = Speaker()
    assistant = LlamaAssistant()

    # Load Whisper speech recognition.
    print("[VOICE] Initializing speech recognition...")
    voice_input = VoiceInput(model_size="base", device="cpu")

    # Track when each object class was last detected.
    last_seen = {}

    # Prevent repeated announcements while a class remains visible.
    announced_classes = set()

    # An object must be absent for this long before it can be
    # announced again.
    disappearance_timeout = 3

    camera_opened = False

    # Protect detections shared between the camera and question thread.
    detections_lock = threading.Lock()
    latest_detections = []

    # Track the background question-processing thread.
    question_thread = None
    question_active = threading.Event()

    try:
        camera.open()
        camera_opened = True

        print("\n=== SightSpeak AI ===")
        print("Object detection and voice announcements are enabled.")

        print("\nControls:")
        print("  A - Ask a question by speaking")
        print("  S - Describe the current scene")
        print("  Q - Quit")

        print("\nMake sure Ollama is running before asking questions.")
        print("Press A, then speak within the 5-second recording window.\n")

        while True:

            # --------------------------------------------------
            # 1. Read a frame from the camera.
            # --------------------------------------------------
            try:
                frame = camera.read()

            except RuntimeError as error:
                print(f"[CAMERA] {error}")
                print("[CAMERA] Attempting to reconnect...")

                try:
                    camera.close()
                except Exception:
                    pass

                camera_opened = False
                time.sleep(2)

                try:
                    camera.open()
                    camera_opened = True
                    print("[CAMERA] Reconnected successfully.")

                except Exception as reconnect_error:
                    print(
                        f"[CAMERA] Reconnection failed: "
                        f"{reconnect_error}"
                    )
                    time.sleep(3)

                continue

            if frame is None:
                time.sleep(0.1)
                continue

            # --------------------------------------------------
            # 2. Run YOLO object detection.
            # --------------------------------------------------
            general_result, detections = detector.detect(frame)
            all_detections = detections

            # Save the latest detections for voice questions.
            with detections_lock:
                latest_detections = [
                    dict(item) for item in all_detections
                ]

            current_time = time.time()

            # --------------------------------------------------
            # 3. Collect visible objects, positions, and proximity.
            # --------------------------------------------------
            visible_objects = {}

            for detection in all_detections:
                name = detection.get("class_name")

                if name not in ANNOUNCE_CLASSES:
                    continue

                # Keep the first detection for each object class.
                if name not in visible_objects:
                    visible_objects[name] = {
                        "position": detection.get(
                            "position", "center"
                        ),
                        "proximity": detection.get(
                            "proximity", "unknown"
                        ),
                    }

            # --------------------------------------------------
            # 4. Announce newly visible objects.
            # --------------------------------------------------
            for name, details in visible_objects.items():
                last_seen[name] = current_time

                if name not in announced_classes:
                    position = details["position"]
                    proximity = details["proximity"]

                    if proximity in {"near", "medium", "far"}:
                        message = (
                            f"{name} on your {position}, "
                            f"appears {proximity}."
                        )
                    else:
                        message = f"{name} on your {position}."

                    print(f"[OBJECT] {message}")

                    # Do not queue routine announcements while
                    # a voice question is being processed.
                    if not question_active.is_set():
                        speaker.speak(
                            message,
                            priority="medium",
                        )

                    announced_classes.add(name)

            # --------------------------------------------------
            # 5. Allow announcements again after objects disappear.
            # --------------------------------------------------
            for name in list(announced_classes):
                if name not in visible_objects:
                    last_time = last_seen.get(name, 0)

                    if (
                        current_time - last_time
                        >= disappearance_timeout
                    ):
                        announced_classes.remove(name)

            # --------------------------------------------------
            # 6. Display the annotated camera frame.
            # --------------------------------------------------
            annotated_frame = general_result.plot(labels=False)
            cv2.imshow("SightSpeak AI", annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            # --------------------------------------------------
            # 7. S: Describe the current scene.
            # --------------------------------------------------
            if key == ord("s"):

                if question_active.is_set():
                    print(
                        "[VOICE] Please wait for the current "
                        "question to finish."
                    )

                else:
                    message = describe_scene(all_detections)

                    print(f"\n[SCENE] {message}")

                    speaker.speak(
                        message,
                        priority="medium",
                    )

            # --------------------------------------------------
            # 8. A: Ask a question using the microphone.
            # --------------------------------------------------
            elif key == ord("a"):

                if question_active.is_set():
                    print(
                        "[VOICE] A question is already being "
                        "processed. Please wait."
                    )
                    continue

                # Set the event before starting the worker thread.
                question_active.set()

                # Clear routine announcements waiting in the queue.
                speaker.clear_pending_medium()

                # Capture a stable snapshot of current detections.
                with detections_lock:
                    saved_detections = [
                        dict(item) for item in latest_detections
                    ]

                question_thread = threading.Thread(
                    target=question_worker,
                    args=(
                        voice_input,
                        assistant,
                        speaker,
                        saved_detections,
                        question_active,
                    ),
                    daemon=True,
                    name="sightspeak-question",
                )

                question_thread.start()

            # --------------------------------------------------
            # 9. Q: Exit SightSpeak AI.
            # --------------------------------------------------
            elif key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopping SightSpeak AI...")

    except Exception as error:
        print(
            f"\n[ERROR] SightSpeak AI encountered an error: {error}"
        )

    finally:
        # Release the camera.
        if camera_opened:
            try:
                camera.close()
            except Exception as error:
                print(f"[CAMERA] Error while closing camera: {error}")

        # Wait briefly for any question already in progress.
        if question_thread is not None and question_thread.is_alive():
            print("[VOICE] Waiting for the current question to finish...")
            question_thread.join(timeout=15)

            if question_thread.is_alive():
                print(
                    "[VOICE] Question processing is still running. "
                    "Shutting down speech."
                )

        # Close the speaker and release OpenCV resources.
        try:
            speaker.close()
        except Exception as error:
            print(f"[SPEECH] Error while closing speaker: {error}")

        cv2.destroyAllWindows()

        print("SightSpeak AI stopped.")


if __name__ == "__main__":
    main()