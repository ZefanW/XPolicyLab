"""Check the agent, observation order and the exact prefix of a planned motion."""
import copy
import json
import unittest

import numpy as np

from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.policy import RoboDojoActionSpec
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.types import Observation
from XPolicyLab.policy.JustRUA.policy import CAMERAS, JustRUAPolicy


class RecordedTransport:
    def __init__(self):
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        name, arguments = (
            ("move_eef", {"targets": {"left_z": .91, "left_gripper": 0}, "note": "Approach."})
            if len(self.requests) == 1 else ("give_up", {"reason": "Finished the test."})
        )
        return {"choices": [{"message": {"role": "assistant", "content": "",
                "tool_calls": [{"id": f"call_{len(self.requests)}", "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


class ObservationFixture:
    guide = "Sensor guide"
    def enrich(self, observation):
        # Distinguishable from the inherited JPEG views.
        return "Sensor reading", [{"type": "image_url", "image_url": {"url": "depth:" + camera}}
                                  for camera in CAMERAS]


class ControlTest(unittest.TestCase):
    def test_next_decision_sees_the_executed_prefix_after_100_decisions(self):
        labels = tuple(f"{arm}_{label}" for arm in ("left", "right")
                       for label in ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"])
        spec = RoboDojoActionSpec(labels, np.full(14, -3.), np.full(14, 3.), 20., "Robot",
                                  max_step=(.02,) * 14)
        images = {camera: np.full((3, 4, 4), i * 40, np.uint8)
                  for i, camera in enumerate(reversed(CAMERAS))}
        state = {}
        for arm in ("left", "right"):
            state[f"{arm}_arm_joint_state"] = np.zeros(6)
            state[f"{arm}_ee_joint_state"] = np.ones(1)
            state[f"{arm}_ee_pose"] = np.array([0, 0, .9, 1, 0, 0, 0])
        observation = Observation(images, state, instruction="Lift the tube.", remaining_steps=400)
        def plan(**_):
            return {"status": "Success", "position": np.array([[0.] * 6, [.6] + [0.] * 5])}
        transport = RecordedTransport()
        policy = JustRUAPolicy(action_spec=spec, planner=plan, observer=ObservationFixture(), client=transport)
        policy._calls = 100
        chunk = policy.act(observation)
        self.assertEqual(len(chunk.actions), 20)
        self.assertGreater(chunk.meta["trace"]["planned_waypoints"], 20)
        self.assertTrue(all(a.data["left_ee_joint_state"][0] == 1. for a in chunk.actions))
        policy.confirm_executed(20)
        self.assertEqual(policy._gripper_goals["left"], 1.)
        policy.act(observation)
        self.assertEqual(policy.calls, 102)
        self.assertIsNone(policy.audit_config()["max_decisions"])
        messages = transport.requests[1]
        self.assertIn("executed 20 of", messages[-1]["content"][0]["text"])
        self.assertIn("Arrival check", messages[-1]["content"][0]["text"])
        observations = [m["content"] for m in messages if isinstance(m["content"], list)]
        self.assertEqual(len(observations), 2)
        for parts in observations:
            urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
            self.assertEqual(len(urls), 6)
            self.assertEqual(urls[3:], ["depth:" + camera for camera in CAMERAS])
            self.assertIn("Sensor reading", parts[0]["text"])
        self.assertEqual([m["role"] for m in messages[-3:]], ["assistant", "tool", "user"])
