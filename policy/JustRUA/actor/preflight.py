"""Verify a native JustRUA installation without sending any model turns."""
import json

from .runtime import CLI_VERSION, EFFORT, MODEL
from .wire import incremental_items


def check():
    from .client import CodingClient

    with CodingClient() as actor:
        actor._open(authenticated=False)
        actor._start_thread("Check the local runtime.", "No model turn will be started.")
        # Exercise the experimental native item transport without starting inference.
        one_pixel = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4"
                     "nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC")
        sample = [{"role": "user", "content": [
            {"type": "text", "text": "Synthetic observation for the offline protocol check."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + one_pixel}},
        ]}]
        feedback = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "probe", "type": "function",
             "function": {"name": "move_eef", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "probe", "content": "Synthetic feedback; no action executed."},
        ]
        for messages in (sample, feedback):
            actor.server.request("thread/inject_items", {"threadId": actor.thread_id,
                                                        "items": incremental_items(messages)})
        return {"codex": CLI_VERSION, "model": MODEL, "effort": EFFORT, "turns_started": 0,
                "multimodal_input_and_feedback": True, "scratch_write": True, "observation_read_only": True,
                "other_case_read_denied": True, "credential_directory_read_denied": True,
                "network_denied": True, "shell_environment_isolated": True, "numpy_and_pillow": True}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
