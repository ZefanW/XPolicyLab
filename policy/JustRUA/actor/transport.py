"""A single-owner stdio connection to the native Codex app-server."""
import json
import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque


class AppServer:
    def __init__(self, argv, env, cwd, stderr):
        self.process = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, start_new_session=True,
        )
        self.events = queue.Queue()
        self.pending = deque()
        self.serial = 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.events.put(json.loads(line))
        except (OSError, ValueError) as error:
            self.events.put(error)
        finally:
            self.events.put(EOFError("Codex app-server closed its output"))

    def send(self, message):
        self.process.stdin.write(json.dumps(message, ensure_ascii=False).encode() + b"\n")
        self.process.stdin.flush()

    def _receive(self, deadline):
        try:
            message = self.events.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty as error:
            raise TimeoutError("No response from Codex app-server") from error
        if isinstance(message, Exception):
            raise message
        if "method" in message and "id" in message:
            # The robot controller never runs model-selected RPCs or grants approvals.
            self.send({"id": message["id"], "error": {"code": -32601, "message": "Unavailable"}})
            raise RuntimeError(f"Unexpected app-server request: {message['method']}")
        return message

    def request(self, method, params, timeout=60):
        self.serial += 1
        self.send({"method": method, "id": self.serial, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            message = self._receive(deadline)
            if "method" in message:
                self.pending.append(message)
                continue
            if message.get("id") != self.serial:
                raise RuntimeError("Unexpected app-server response ID")
            if "error" in message:
                raise RuntimeError(f"{method}: {message['error']}")
            return message["result"]

    def next_event(self, deadline):
        return self.pending.popleft() if self.pending else self._receive(deadline)

    def close(self):
        # The case owns its entire process group, including interactive coding sessions.
        for sig, timeout in ((signal.SIGTERM, 3), (signal.SIGKILL, 3)):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                break
            except PermissionError:
                # On Darwin a child that has just closed stdout can already be a zombie.
                if self.process.poll() is not None:
                    break
                raise
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                continue
            if sig == signal.SIGTERM:
                # Some shell children can outlive the session leader.
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            break
        self.process.wait()
        self.process.stdin.close()
        self.reader.join(timeout=3)
        self.process.stdout.close()
