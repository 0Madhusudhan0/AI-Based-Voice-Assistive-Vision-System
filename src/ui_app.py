import sys
import time
import threading
from pathlib import Path

import cv2
import streamlit as st
import pandas as pd
from PIL import Image

# Ensure project modules can be imported when Streamlit starts.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from camera.webcam import Webcam
from detection.detector import ObjectDetector
from speech.speaker import Speaker
from speech.voice_input import VoiceInput
from llm.assistant import LlamaAssistant


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="SightSpeak AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b1120;
            color: #f1f5f9;
        }

        h1, h2, h3 {
            color: #67e8f9 !important;
        }

        div[data-testid="stMetric"] {
            background-color: #172033;
            border: 1px solid #263449;
            padding: 16px;
            border-radius: 12px;
        }

        div[data-testid="stButton"] button {
            border-radius: 9px;
            min-height: 42px;
        }

        .app-subtitle {
            color: #94a3b8;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# PROJECT SETTINGS
# ---------------------------------------------------------

DROIDCAM_URL = "http://192.168.1.68:4747/video"
# DROIDCAM_URL = "http://10.146.163.154:4747/video"


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
    """Create a concise description from current detections."""

    objects = []
    seen = set()

    for detection in detections:
        name = detection.get("class_name")
        position = detection.get("position", "center")
        proximity = detection.get("proximity", "unknown")

        key = (name, position)

        if name not in ANNOUNCE_CLASSES or key in seen:
            continue

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


# ---------------------------------------------------------
# BACKGROUND CAMERA AND DETECTION SERVICE
# ---------------------------------------------------------

class SightSpeakService:
    """Manage camera detection and voice processing in background threads."""

    def __init__(self):
        self.camera = Webcam(stream_url=DROIDCAM_URL)
        self.detector = ObjectDetector()
        self.speaker = Speaker()
        self.assistant = LlamaAssistant()

        # Load Whisper only when the user first asks a question.
        self.voice_input = None

        self.lock = threading.RLock()
        self.voice_lock = threading.Lock()

        self.stop_event = threading.Event()
        self.camera_thread = None

        self.frame = None
        self.detections = []

        self.camera_status = "Stopped"
        self.question_status = "Idle"
        self.last_error = ""

        self.auto_announcements = True

        self.conversation = []

        self.announced_classes = set()
        self.last_seen = {}
        self.disappearance_timeout = 3

    def start_camera(self):
        """Start the camera worker if it is not already running."""

        with self.lock:
            if (
                self.camera_thread is not None
                and self.camera_thread.is_alive()
            ):
                return

            self.stop_event.clear()
            self.camera_status = "Connecting"
            self.last_error = ""

            self.camera_thread = threading.Thread(
                target=self._camera_loop,
                name="sightspeak-camera",
                daemon=True,
            )

            self.camera_thread.start()

    def stop_camera(self):
        """Stop the camera worker and release the camera."""

        self.stop_event.set()

        thread = self.camera_thread

        if thread and thread.is_alive():
            thread.join(timeout=5)

        with self.lock:
            if thread is None or not thread.is_alive():
                self.camera_thread = None
                self.camera_status = "Stopped"
                self.frame = None
                self.detections = []

    def _camera_loop(self):
        """Continuously capture frames and run YOLO inference."""

        try:
            while not self.stop_event.is_set():

                try:
                    self.camera.open()

                    with self.lock:
                        self.camera_status = "Connected"

                except Exception as error:
                    with self.lock:
                        self.camera_status = "Disconnected"
                        self.last_error = str(error)

                    if self.stop_event.wait(3):
                        break

                    continue

                try:
                    frame = self.camera.read()

                    result, detections = self.detector.detect(frame)

                    annotated_frame = result.plot(labels=True)

                    annotated_frame = result.plot(labels=True)

                    # Original BGR-to-RGB frame processing.
                    annotated_frame = cv2.cvtColor(
                        annotated_frame,
                        cv2.COLOR_BGR2RGB,
                    )

                    current_time = time.time()

                    with self.lock:
                        self.frame = annotated_frame
                        self.detections = [
                            dict(item) for item in detections
                        ]

                        auto_announce = self.auto_announcements

                    if auto_announce:
                        self._announce_objects(
                            detections,
                            current_time,
                        )

                except Exception as error:
                    with self.lock:
                        self.camera_status = "Reconnecting"
                        self.last_error = str(error)

                    try:
                        self.camera.close()
                    except Exception:
                        pass

                    if self.stop_event.wait(2):
                        break

        finally:
            try:
                self.camera.close()
            except Exception:
                pass

            with self.lock:
                self.camera_status = "Stopped"
                self.frame = None

    def _announce_objects(self, detections, current_time):
        """Announce newly visible object classes."""

        visible = {}

        for detection in detections:
            name = detection.get("class_name")

            if name not in ANNOUNCE_CLASSES:
                continue

            if name not in visible:
                visible[name] = detection

        for name, detection in visible.items():
            self.last_seen[name] = current_time

            if name in self.announced_classes:
                continue

            position = detection.get("position", "center")
            proximity = detection.get("proximity", "unknown")

            if proximity in {"near", "medium", "far"}:
                message = (
                    f"{name} on your {position}, "
                    f"appears {proximity}."
                )
            else:
                message = f"{name} on your {position}."

            try:
                self.speaker.speak(message, priority="medium")
            except Exception as error:
                with self.lock:
                    self.last_error = str(error)

            self.announced_classes.add(name)

        for name in list(self.announced_classes):
            if name not in visible:
                last_time = self.last_seen.get(name, 0)

                if current_time - last_time >= self.disappearance_timeout:
                    self.announced_classes.remove(name)

    def set_auto_announcements(self, enabled):
        """Enable or disable routine object announcements."""

        with self.lock:
            self.auto_announcements = enabled

            if not enabled:
                self.speaker.clear_pending_medium()

    def get_snapshot(self):
        """Return a consistent snapshot for the dashboard."""

        with self.lock:
            frame = (
                self.frame.copy()
                if self.frame is not None
                else None
            )

            detections = [
                dict(item) for item in self.detections
            ]

            return (
                frame,
                detections,
                self.camera_status,
                self.question_status,
                self.last_error,
                list(self.conversation),
                self.auto_announcements,
            )

    def describe_current_scene(self):
        """Speak a description of the latest detections."""

        with self.lock:
            detections = [
                dict(item) for item in self.detections
            ]

            busy = self.question_status != "Idle"

        if busy:
            return "Please wait for the current voice question to finish."

        message = describe_scene(detections)

        self.speaker.speak(message, priority="medium")

        return message

    def ask_voice_question(self):
        """Start voice processing in a background thread."""

        if not self.voice_lock.acquire(blocking=False):
            return False

        with self.lock:
            if self.camera_status != "Connected":
                self.voice_lock.release()
                return False

            self.question_status = "Listening"
            saved_detections = [
                dict(item) for item in self.detections
            ]

        self.speaker.clear_pending_medium()

        thread = threading.Thread(
            target=self._question_worker,
            args=(saved_detections,),
            name="sightspeak-question",
            daemon=True,
        )

        thread.start()

        return True

    def _question_worker(self, saved_detections):
        """Record a question, ask Llama, and speak the answer."""

        try:
            if self.voice_input is None:
                with self.lock:
                    self.question_status = "Loading Whisper"

                self.voice_input = VoiceInput(
                    model_size="base",
                    device="cpu",
                )

            with self.lock:
                self.question_status = "Listening"

            question = self.voice_input.listen(duration=5).strip()

            if not question:
                with self.lock:
                    self.conversation.append({
                        "role": "assistant",
                        "text": "I couldn't recognize your question. Please try again.",
                    })
                return

            with self.lock:
                self.conversation.append({
                    "role": "user",
                    "text": question,
                })
                self.question_status = "Thinking"

            answer = self.assistant.ask(
                question,
                saved_detections,
            )

            with self.lock:
                self.conversation.append({
                    "role": "assistant",
                    "text": answer,
                })
                self.question_status = "Speaking"

            self.speaker.speak(answer, priority="high")
            self.speaker.wait_until_idle(timeout=120)

        except Exception as error:
            with self.lock:
                self.last_error = str(error)
                self.conversation.append({
                    "role": "assistant",
                    "text": f"An error occurred: {error}",
                })

        finally:
            with self.lock:
                self.question_status = "Idle"

            self.voice_lock.release()

    def close(self):
        """Stop camera and release speech resources."""

        self.stop_camera()

        try:
            self.speaker.close()
        except Exception:
            pass


# ---------------------------------------------------------
# INITIALIZE SHARED SERVICE
# ---------------------------------------------------------

@st.cache_resource
def get_service():
    return SightSpeakService()


service = get_service()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.title("👁️ SightSpeak AI")

    st.caption("Voice-Based Assistive Vision System")

    st.divider()

    st.subheader("Camera Controls")

    if st.button("▶️ Start Camera", use_container_width=True):
        service.start_camera()

    if st.button("⏹️ Stop Camera", use_container_width=True):
        service.stop_camera()

    st.divider()

    current_auto = service.get_snapshot()[-1]

    auto_announcements = st.toggle(
        "Automatic Voice Announcements",
        value=current_auto,
        key="auto_announcement_toggle",
    )

    service.set_auto_announcements(auto_announcements)

    st.divider()

    st.caption("YOLO11n • Whisper • Llama 3.2")

    st.caption("Local processing with Ollama")


# ---------------------------------------------------------
# DASHBOARD HEADER
# ---------------------------------------------------------

st.title("👁️ SightSpeak AI Dashboard")

st.markdown(
    '<p class="app-subtitle">'
    "Real-time object detection and voice-based scene assistance"
    "</p>",
    unsafe_allow_html=True,
)

st.divider()


# ---------------------------------------------------------
# LIVE DASHBOARD FRAGMENT
# ---------------------------------------------------------

@st.fragment(run_every=0.5)
def live_dashboard():

    (
        frame,
        detections,
        camera_status,
        question_status,
        last_error,
        conversation,
        auto_enabled,
    ) = service.get_snapshot()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Camera",
            camera_status,
        )

    with col2:
        st.metric(
            "Objects Detected",
            len(detections),
        )

    with col3:
        st.metric(
            "Voice Assistant",
            question_status,
        )

    camera_col, objects_col = st.columns(
        [1.8, 1],
        gap="large",
    )

    with camera_col:
        st.subheader("📹 Live Camera Feed")

        if frame is not None:
            st.image(
                Image.fromarray(frame),
                use_container_width=True,
            )
        else:
            st.info(
                "Camera is not running. "
                "Press Start Camera in the sidebar."
            )

    with objects_col:
        st.subheader("🎯 Detected Objects")

        if detections:
            rows = []

            for item in detections:
                rows.append({
                    "Object": item["class_name"],
                    "Confidence": (
                        f'{item["confidence"] * 100:.1f}%'
                    ),
                    "Position": item["position"].title(),
                    "Proximity": item["proximity"].title(),
                })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info("No objects detected yet.")

    st.divider()

    st.subheader("🗣️ Voice Assistant")

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        if st.button(
            "👁️ Describe Scene",
            use_container_width=True,
            disabled=(camera_status != "Connected"),
        ):
            message = service.describe_current_scene()
            st.toast(message)

    with action_col2:
        if st.button(
            "🎙️ Ask a Voice Question",
            use_container_width=True,
            disabled=(
                camera_status != "Connected"
                or question_status != "Idle"
            ),
        ):
            started = service.ask_voice_question()

            if started:
                st.toast(
                    "Listening for 5 seconds. Speak now!"
                )
            else:
                st.warning(
                    "The assistant is busy or the camera is disconnected."
                )

    st.caption(
        "Example questions: 'What is in front of me?' "
        "or 'Where is the chair?'"
    )

    st.divider()

    st.subheader("💬 Conversation History")

    if conversation:
        for message in conversation[-10:]:
            if message["role"] == "user":
                st.markdown(
                    f'**🎙️ You:** {message["text"]}'
                )
            else:
                st.markdown(
                    f'**🤖 SightSpeak AI:** {message["text"]}'
                )
    else:
        st.caption(
            "Your voice questions and assistant responses "
            "will appear here."
        )

    if last_error:
        with st.expander("Technical details"):
            st.code(last_error)

    st.caption(
        "Proximity is estimated visually; it is not a "
        "measurement of distance in metres."
    )


live_dashboard()