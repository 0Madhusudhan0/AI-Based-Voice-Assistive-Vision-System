import cv2


class Webcam:
    """Capture frames from a local camera or an IP camera stream."""

    def __init__(
        self,
        camera_index: int = 0,
        stream_url: str | None = None,
    ) -> None:
        self.camera_index = camera_index
        self.stream_url = stream_url
        self._capture = None

    def open(self) -> None:
        """Open the configured camera source."""
        if self._capture is not None and self._capture.isOpened():
            return

        if self.stream_url:
            # Connect to the phone's DroidCam video stream.
            capture = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        else:
            # Use the laptop's local webcam.
            capture = cv2.VideoCapture(
                self.camera_index,
                cv2.CAP_DSHOW,
            )

        if not capture.isOpened():
            capture.release()

            if self.stream_url:
                raise RuntimeError(
                    f"Unable to open camera stream: {self.stream_url}"
                )

            raise RuntimeError(
                f"Unable to open camera index {self.camera_index}"
            )

        self._capture = capture

    def read(self):
        """Read a frame from the active camera."""
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError("Camera is not open.")

        success, frame = self._capture.read()

        if not success or frame is None:
            raise RuntimeError("Failed to capture camera frame.")

        return frame

    def close(self) -> None:
        """Release the camera resource."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None