import json
import urllib.error
import urllib.request


class LlamaAssistant:
    """Answer user questions using Ollama and current YOLO detections."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.url = f"{base_url}/api/chat"

    def ask(self, question: str, detections: list[dict]) -> str:
        # Summarize detections without including duplicate bounding boxes.
        objects = []
        seen = set()

        for detection in detections:
            name = detection["class_name"]
            position = detection["position"]
            key = (name, position)

            if key not in seen:
                seen.add(key)
                objects.append({
                    "object": name,
                    "position": position,
                })

            if len(objects) >= 15:
                break

        scene = json.dumps(objects, ensure_ascii=False)

        system_prompt = """
You are SightSpeak AI, a concise and helpful assistant.

The user may ask general questions or questions about their surroundings.

You receive object detections from a camera. These detections contain
object names and estimated horizontal positions.

Rules:
- Answer general questions normally.
- For questions about the current scene, use the supplied detections.
- Never claim to see objects or details that were not provided.
- If the detections cannot answer a question, say so honestly.
- Treat object positions as approximate.
- Do not claim to know actual distances.
- Use simple, natural language suitable for spoken responses.
- Keep answers to approximately one to three sentences.
"""

        user_message = (
            f"Current camera detections: {scene}\n\n"
            f"User's question: {question}"
        )

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "options": {
                "temperature": 0.2,
            },
        }

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))

            answer = result.get("message", {}).get("content", "").strip()

            if not answer:
                return "Sorry, I could not generate an answer."

            return answer

        except urllib.error.URLError:
            return (
                "I could not connect to Llama. "
                "Please check that Ollama is running."
            )

        except (TimeoutError, json.JSONDecodeError, KeyError, TypeError):
            return "Sorry, I encountered a problem answering your question."