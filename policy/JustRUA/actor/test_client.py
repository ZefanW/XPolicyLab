"""Protocol regressions with an inert stdio child; no Codex account or network is used."""
import base64
import json
import shutil
import sys
import unittest
from unittest.mock import patch

from . import CodingClient
from .runtime import EFFORT, MODEL, PROFILE


FIXTURE = """
import json, sys
from pathlib import Path
def emit(value):
    print(json.dumps(value), flush=True)
def event(method, **params):
    emit({"method": method, "params": params})
turn = 0
for line in sys.stdin:
    request = json.loads(line)
    with Path("wire.jsonl").open("a") as record:
        record.write(line)
    if "id" not in request:
        continue
    method, params = request["method"], request["params"]
    result = {}
    if method == "thread/start":
        result = {k: params[k] for k in ("cwd","runtimeWorkspaceRoots","model","modelProvider","approvalPolicy")}
        result.update(thread={"id":"th"}, reasoningEffort=params["config"]["model_reasoning_effort"],
                      activePermissionProfile={"id": params["permissions"]}, instructionSources=[])
    if method == "turn/start":
        turn += 1
        tid = "t" + str(turn)
        p = {"threadId": "th", "turnId": tid}
        emit({"id": request["id"], "result": {"turn":{"id":tid}}})
        if sys.argv[-1] == "disconnect":
            sys.exit(0)
        event("rawResponseItem/completed", **p, item={"type":"function_call","name":"exec_command","call_id":"native"})
        event("rawResponseItem/completed", **p, item={"type":"function_call_output","call_id":"native","output":"42"})
        event("rawResponse/completed", **p, responseId=tid, usage={"inputTokens":12,"outputTokens":4,"totalTokens":16})
        event("rawResponse/completed", **p, responseId=tid, usage={"inputTokens":12,"outputTokens":4,"totalTokens":16})
        event("item/completed", **p, item={"type":"agentMessage","text":json.dumps({"content":"moving","tool_calls":[
            {"name":"move_eef","arguments":"{}"}]})})
        event("turn/completed", threadId="th", turn={"id":tid,"status":"completed"})
        continue
    emit({"id": request["id"], "result": result})
"""


class TestCodingClient(unittest.TestCase):
    def actor(self, scenario):
        actor = CodingClient()
        self.addCleanup(lambda: shutil.rmtree(actor.root, ignore_errors=True))
        self.addCleanup(actor.close)
        ctx = patch(__package__ + ".client.native_command",
                    return_value=(sys.executable, [sys.executable, "-Iu", "-c", FIXTURE, scenario]))
        ctx.start()
        self.addCleanup(ctx.stop)
        # Run only the fixture with an empty environment, including no credential paths.
        env = patch(__package__ + ".client.server_environment", return_value={})
        env.start()
        self.addCleanup(env.stop)
        probe = patch(__package__ + ".client.verify_sandbox")
        probe.start()
        self.addCleanup(probe.stop)
        return actor

    def setUp(self):
        self.tools = [{"type": "function", "function": {"name": n}} for n in ("move_eef", "give_up")]
        self.url = "data:image/png;base64," + base64.b64encode(b"supplied-pixels").decode()
        self.messages = [
            {"role": "system", "content": "absolute EEF"},
            {"role": "user", "content": [
                {"type": "text", "text": "eef measured at x=.1"},
                {"type": "image_url", "image_url": {"url": self.url}}]},
        ]

    def test_two_decisions_share_thread_and_scratch(self):
        actor = self.actor("ok")
        response = actor.complete(self.messages, self.tools)
        self.assertEqual(response["usage"]["total_tokens"], 16)
        message = response["choices"][0]["message"]
        self.messages.extend([message, {"role": "tool", "tool_call_id": "call_t1_0", "content": "arrived"},
                              {"role": "user", "content": "eef measured at x=.2"}])
        actor.complete(self.messages, self.tools)
        requests = [json.loads(line) for line in (actor.scratch / "wire.jsonl").read_text().splitlines()]
        starts = [r["params"] for r in requests if r["method"] == "thread/start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual((starts[0]["model"], starts[0]["permissions"]), (MODEL, PROFILE))
        self.assertEqual(starts[0]["config"]["model_reasoning_effort"], EFFORT)
        inputs = [r["params"]["items"] for r in requests if r["method"] == "thread/inject_items"]
        self.assertIn(self.url, json.dumps(inputs[0]))
        self.assertNotIn(self.url, json.dumps(inputs[1]))
        self.assertNotIn("moving", json.dumps(inputs[1]))
        self.assertEqual(inputs[1][0], {"type": "function_call", "name": "move_eef",
                                        "call_id": "call_t1_0", "arguments": "{}"})
        self.assertEqual(inputs[1][1]["output"], "arrived")
        self.assertEqual((actor.inputs / "0000/image-000.png").read_bytes(), b"supplied-pixels")
        self.assertTrue(all(r["params"]["threadId"] == "th" and not r["params"]["input"]
                            for r in requests if r["method"] == "turn/start"))
        actor.close()
        self.assertFalse(actor.root.exists())

    def test_lost_response_holds_the_case_without_redispatch(self):
        actor = self.actor("disconnect")
        with self.assertRaisesRegex(RuntimeError, "episode held"):
            actor.complete(self.messages, self.tools)
        with self.assertRaisesRegex(RuntimeError, "closed or held"):
            actor.complete(self.messages, self.tools)
        actor.close()
        self.assertTrue(actor.root.exists())
        requests = [json.loads(line) for line in (actor.scratch / "wire.jsonl").read_text().splitlines()]
        self.assertEqual(sum(r["method"] == "turn/start" for r in requests), 1)
        self.assertEqual(json.loads((actor.root / "decision-status.json").read_text())["turn"], "t1")


if __name__ == "__main__":
    unittest.main()
