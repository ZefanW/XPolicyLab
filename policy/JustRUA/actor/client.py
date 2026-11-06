"""One persistent native Codex actor per Direct EEF episode."""
import atexit
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time

from .runtime import EFFORT, MODEL, PROFILE, native_command, server_environment
from .sandbox import verify_sandbox
from .transport import AppServer
from .wire import decode_answer, digest, incremental_items, output_schema, robot_contract, write_inputs


NATIVE_CALLS = {"exec", "wait", "exec_command", "write_stdin", "shell", "shell_command",
                "apply_patch", "view_image", "update_plan", "clock__curr_time"}


class CodingClient:
    """Pass the full append-only policy transcript to complete(); the wire receives only its new suffix."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="justrua-case-")).resolve()
        self.scratch = self.root / "scratch"
        self.inputs = self.root / "observations"
        for path in (self.scratch, self.inputs, self.scratch / ".tmp"):
            path.mkdir(mode=0o700)
        self.server = None
        self.thread_id = None
        self.contract = None
        self.prefix = []
        self.last_call_ids = set()
        self.decisions = 0
        self.closed = False
        self.failed = False
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _open(self, *, authenticated):
        self.python, argv = native_command(self.root, self.scratch, self.inputs)
        env = server_environment(self.root, authenticated)
        with (self.root / "app-server.stderr").open("wb") as log:
            self.server = AppServer(argv, env, self.scratch, log)
        self.server.request("initialize", {
            "clientInfo": {"name": "justrua", "version": "1"},
            "capabilities": {"experimentalApi": True},
        })
        self.server.send({"method": "initialized", "params": {}})

    def _start_thread(self, system, developer):
        cwd = str(self.scratch)
        reply = self.server.request("thread/start", {
            "cwd": cwd, "runtimeWorkspaceRoots": [cwd], "model": MODEL, "modelProvider": "openai",
            "config": {"model_reasoning_effort": EFFORT}, "baseInstructions": system,
            "developerInstructions": developer, "dynamicTools": [], "experimentalRawEvents": True,
            "ephemeral": False, "allowProviderModelFallback": False,
            "approvalPolicy": "never", "permissions": PROFILE,
        })
        profile = reply.get("activePermissionProfile") or {}
        if (reply.get("model") != MODEL or reply.get("reasoningEffort") != EFFORT
                or reply.get("modelProvider") != "openai" or reply.get("instructionSources") != []
                or profile.get("id") != PROFILE or profile.get("extends")
                or reply.get("approvalPolicy") != "never" or reply.get("runtimeWorkspaceRoots") != [cwd]
                or reply.get("cwd") != cwd):
            raise RuntimeError("Codex did not resolve the required model, instructions and case permissions")
        self.thread_id = reply["thread"]["id"]
        verify_sandbox(self)

    def complete(self, messages, tools):
        """Return one chat-shaped assistant response. A failed dispatch can never be replayed here."""
        with self._lock:
            if self.closed or self.failed:
                raise RuntimeError("This episode's Codex client is closed or held after a failure")
            try:
                return self._complete(messages, tools)
            except BaseException as error:
                self.failed = True
                self._shutdown()
                if isinstance(error, Exception):
                    raise RuntimeError(f"Codex episode held. Private evidence: {self.root}. {error}") from error
                raise

    def _complete(self, messages, tools):
        contract = robot_contract(messages, tools)
        if self.contract is not None and contract != self.contract:
            raise ValueError("System or robot tools changed within the episode")
        offset = len(self.prefix)
        fresh = messages[offset:]
        if not fresh or [digest(m) for m in messages[:offset]] != self.prefix:
            raise ValueError("Expected the original full history followed by new policy input")
        if not offset and any(m["role"] in ("assistant", "tool") for m in fresh):
            raise ValueError("A new episode must begin with its own initial observations")
        for message in fresh:
            if message["role"] == "assistant":
                if any(call["id"] not in self.last_call_ids for call in message.get("tool_calls", [])):
                    raise ValueError("Unexpected robot action alias from outside the previous native decision")
            if message["role"] == "tool" and message.get("tool_call_id") not in self.last_call_ids:
                raise ValueError("Robot feedback does not match the previous native decision")
        path = write_inputs(self.inputs / f"{self.decisions:04d}", fresh, tools)
        if self.server is None:
            self._open(authenticated=True)
            self._start_thread(*contract)
            self.contract = contract
        items = incremental_items(fresh)
        items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text":
            f"Coding input for this decision: {path}. Prior observation files remain readable. "
            f"Scratch directory: {self.scratch}. Python with NumPy and Pillow: {self.python}."
        }]})
        self.server.request("thread/inject_items", {"threadId": self.thread_id, "items": items})
        status_file = self.root / "decision-status.json"
        status = {"decision": self.decisions, "thread": self.thread_id, "turn": None, "stage": "dispatching"}
        status_file.write_text(json.dumps(status))
        reply = self.server.request("turn/start", {
            "threadId": self.thread_id, "model": MODEL, "effort": EFFORT, "input": [],
            "outputSchema": output_schema(), "approvalPolicy": "never", "permissions": PROFILE,
            "cwd": str(self.scratch), "runtimeWorkspaceRoots": [str(self.scratch)],
        })
        turn_id = reply["turn"]["id"]
        status.update(turn=turn_id, stage="in_progress")
        status_file.write_text(json.dumps(status))
        response = self._wait(turn_id)
        status_file.write_text(json.dumps({**status, "stage": "completed"}))
        answer = response["choices"][0]["message"]
        self.last_call_ids = {call["id"] for call in answer.get("tool_calls", [])}
        self.prefix.extend(digest(message) for message in fresh)
        self.decisions += 1
        return response

    def _wait(self, turn_id):
        final = None
        usage_by_id = {}
        native_calls = set()
        native_outputs = set()
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            event = self.server.next_event(deadline)
            kind, params = event.get("method"), event.get("params", {})
            if params.get("threadId") != self.thread_id:
                continue
            if kind == "turn/completed" and params["turn"]["id"] == turn_id:
                if params["turn"]["status"] != "completed" or final is None:
                    raise RuntimeError(f"Codex turn {turn_id} ended with {params['turn']['status']}: {params['turn'].get('error')}")
                if native_calls != native_outputs:
                    raise RuntimeError("Native turn ended with unresolved coding calls")
                response = decode_answer(final, turn_id)
                if usage_by_id:
                    response["usage"] = _usage(usage_by_id)
                return response
            if params.get("turnId") != turn_id:
                continue
            if kind == "error" and not params.get("willRetry"):
                raise RuntimeError(f"Codex turn failed: {params.get('error')}")
            if kind in ("item/started", "item/completed"):
                item = params["item"]
                if item["type"] in ("mcpToolCall", "dynamicToolCall", "collabAgentToolCall", "webSearch"):
                    raise RuntimeError(f"External tool outside the case profile: {item['type']}")
                if kind == "item/completed" and item["type"] == "agentMessage":
                    final = item["text"]
            if kind == "rawResponseItem/completed":
                item = params["item"]
                tag = item["type"]
                if tag in ("function_call", "custom_tool_call"):
                    call_id = item["call_id"]
                    if (not call_id or call_id in native_calls
                            or item["name"].removeprefix("functions.") not in NATIVE_CALLS):
                        raise RuntimeError("Unexpected or duplicate native coding call")
                    native_calls.add(call_id)
                elif tag in ("function_call_output", "custom_tool_call_output"):
                    call_id = item["call_id"]
                    if call_id not in native_calls or call_id in native_outputs:
                        raise RuntimeError("Unexpected native coding output")
                    native_outputs.add(call_id)
                elif tag not in ("reasoning", "message"):
                    raise RuntimeError(f"Unexpected native item: {tag}")
            if kind == "rawResponse/completed":
                identity, usage = params["responseId"], params.get("usage")
                if identity in usage_by_id and usage_by_id[identity] != usage:
                    raise RuntimeError("Conflicting usage for one native model response")
                usage_by_id[identity] = usage
            # The timeout is an inactivity check, independent of the number of coding calls.
            deadline = time.monotonic() + 600
        raise TimeoutError(f"Codex turn {turn_id} became inactive")

    def _shutdown(self):
        try:
            if self.server is not None:
                self.server.close()
                self.server = None
        finally:
            (self.root / "codex-home" / "auth.json").unlink(missing_ok=True)

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            try:
                self._shutdown()
            finally:
                atexit.unregister(self.close)
                if not self.failed:
                    shutil.rmtree(self.root)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None and self.thread_id is not None:
            self.failed = True
            if hasattr(exc, "add_note"):
                exc.add_note(f"JustRUA private Codex episode retained at {self.root}")
        self.close()


def _usage(receipts):
    """Only unique model-response deltas contribute. Leave unavailable usage unknown."""
    fields = ("inputTokens", "outputTokens", "totalTokens", "cachedInputTokens", "reasoningOutputTokens")
    totals = {}
    for field in fields:
        values = [row.get(field) if isinstance(row, dict) else None for row in receipts.values()]
        totals[field] = sum(values) if all(type(n) is int and n >= 0 for n in values) else None
    return {"prompt_tokens": totals["inputTokens"], "completion_tokens": totals["outputTokens"],
            "total_tokens": totals["totalTokens"],
            "prompt_tokens_details": {"cached_tokens": totals["cachedInputTokens"]},
            "completion_tokens_details": {"reasoning_tokens": totals["reasoningOutputTokens"]}}
