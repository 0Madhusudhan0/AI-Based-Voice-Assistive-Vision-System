from collections import deque
import subprocess
import threading
import time


class Speaker:
    """Offline Windows speech with priority and playback coordination."""

    PRIORITY_RANK = {
        "high": 0,
        "medium": 1,
    }

    def __init__(self, max_pending: int = 5) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1.")

        self._pending: deque[tuple[int, int, str]] = deque()
        self._max_pending = max_pending
        self._sequence = 0
        self._closed = False
        self._speaking = False

        self._condition = threading.Condition()

        self._worker = threading.Thread(
            target=self._run,
            name="sightspeak-speaker",
            daemon=True,
        )
        self._worker.start()

    def _speak_windows(self, message: str) -> None:
        safe_message = message.replace("'", "''")

        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Rate = 0; "
            f"$s.Speak('{safe_message}'); "
            "$s.Dispose()"
        )

        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or "Windows speech failed."
            )

    def speak(self, message: str, priority: str = "medium") -> None:
        """Queue a message for speech playback."""

        if priority not in self.PRIORITY_RANK:
            raise ValueError("priority must be 'high' or 'medium'.")

        if not message.strip():
            return

        priority_rank = self.PRIORITY_RANK[priority]

        with self._condition:
            if self._closed:
                raise RuntimeError("Speaker is already closed.")

            if len(self._pending) >= self._max_pending:
                if priority == "medium":
                    print("[SPEECH QUEUE] Dropped medium-priority alert.")
                    return

                medium_index = next(
                    (
                        index
                        for index, item in enumerate(self._pending)
                        if item[0] == self.PRIORITY_RANK["medium"]
                    ),
                    None,
                )

                if medium_index is None:
                    print(
                        "[SPEECH QUEUE] Queue full; "
                        "no lower-priority alert could be replaced."
                    )
                    return

                del self._pending[medium_index]
                print(
                    "[SPEECH QUEUE] Replaced a pending medium "
                    "alert with a high-priority alert."
                )

            self._pending.append(
                (priority_rank, self._sequence, message)
            )
            self._sequence += 1
            self._condition.notify_all()

    def clear_pending_medium(self) -> None:
        """Remove queued routine announcements without stopping current speech."""

        with self._condition:
            original_count = len(self._pending)

            self._pending = deque(
                item
                for item in self._pending
                if item[0] != self.PRIORITY_RANK["medium"]
            )

            removed_count = original_count - len(self._pending)

            if removed_count:
                print(
                    f"[SPEECH QUEUE] Cleared {removed_count} "
                    "pending routine announcement(s)."
                )

            self._condition.notify_all()

    def wait_until_idle(self, timeout: float = 60.0) -> bool:
        """Wait until the current speech and queued messages have finished."""

        deadline = time.monotonic() + timeout

        with self._condition:
            while self._speaking or self._pending:
                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    return False

                self._condition.wait(timeout=remaining)

            return True

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()

                if self._closed and not self._pending:
                    return

                best_index = min(
                    range(len(self._pending)),
                    key=lambda index: (
                        self._pending[index][0],
                        self._pending[index][1],
                    ),
                )

                _, _, message = self._pending[best_index]
                del self._pending[best_index]
                self._speaking = True

            try:
                self._speak_windows(message)

            except Exception as exc:
                print(f"[SPEECH ERROR] {exc}")

            finally:
                with self._condition:
                    self._speaking = False
                    self._condition.notify_all()

    def close(self) -> None:
        """Finish queued speech and shut down the speech worker."""

        with self._condition:
            if self._closed:
                return

            self._closed = True
            self._condition.notify_all()

        self._worker.join(timeout=60)

        if self._worker.is_alive():
            raise RuntimeError(
                "Speech worker did not shut down in time."
            )