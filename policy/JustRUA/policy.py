"""Absolute Cartesian control with observation-based replanning every 20 steps."""
from dataclasses import dataclass, replace
from math import inf

import numpy as np

from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.policy import EefAgentPolicy

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CONTROL_CAP = 20


@dataclass
class PendingMotion:
    planned_steps: int
    actions: tuple
    prior_gripper_goals: dict


class JustRUAPolicy(EefAgentPolicy):
    """One persistent coding conversation per native RoboDojo episode."""

    def __init__(self, *, action_spec, planner, observer, client):
        self.observer = observer
        self._sensor_text = ""
        super().__init__(
            planner=planner,
            action_spec=action_spec,
            client=client,
            env={"L4_INSPECT_KEEP_ALL_IMAGES": "1", "L4_INSPECT_USE_RECIPE": "1"},
        )

    def reset(self):
        # Direct EEF's act() compares an integer counter against this bound.
        self._max_llm_calls = inf
        self._pending = None
        self._execution_feedback = None
        super().reset()

    def _system_message(self):
        return (
            "You control a dual-arm ARX X5 robot. Each observation supplies three "
            "RGB views, predicted depth for those views, robot readings and camera "
            "calibration. The grasp-point state of both arms uses the same 14 "
            "dimensions as move_eef. Specify only absolute world-frame target "
            "dimensions you want to change. Arm joint angles provide read-only "
            "context. You can use native coding tools to inspect the supplied "
            "observations, calculate and keep files for this episode. End each "
            "decision with exactly one robot tool call. Only the robot action "
            "advances the environment. There is no decision-turn limit. Each "
            "move_eef executes at most 20 controls from its plan. Any suffix is "
            "discarded. Check the next measured state and arrival error before "
            "planning another move. A longer motion spends more physical steps. "
            "The environment's own remaining step budget "
            "is reported with each observation.\n\n"
            + self.observer.guide
            + "\n\nEmbodiment notes:\n"
            + self.action_spec.docs.strip()
        )

    def _state_block(self, observation):
        return super()._state_block(observation) + "\n\n" + self._sensor_text

    def _observation_message(self, observation):
        if set(observation.images) != set(CAMERAS):
            raise ValueError("JustRUA requires the head and both wrist RGB cameras.")
        self._sensor_text, depth_parts = self.observer.enrich(observation)
        # Isaac can return RGBA. The upstream policy's JPEG encoder takes RGB.
        observation = replace(observation, images={k: observation.images[k][..., :3] for k in CAMERAS})
        message = super()._observation_message(observation)
        message["content"].extend(depth_parts)
        if self._execution_feedback is not None:
            message["content"][0]["text"] += "\n" + self._execution_feedback
            self._execution_feedback = None
        return message

    def _handle_motion(self, name, arguments, observation):
        if self._pending is not None:
            raise RuntimeError("Confirm the executed motion before requesting a new one.")
        previous_goals = self._gripper_goals.copy()
        outcome = super()._handle_motion(name, arguments, observation)
        if outcome.chunk is None:
            return outcome
        planned = len(outcome.chunk.actions)
        if planned <= CONTROL_CAP:
            return outcome
        prefix = tuple(outcome.chunk.actions[:CONTROL_CAP])
        self._pending = PendingMotion(planned, prefix, previous_goals)
        trace = {**outcome.chunk.meta.get("trace", {}), "executed_prefix_limit": CONTROL_CAP}
        limited = replace(outcome.chunk, actions=prefix, meta={**outcome.chunk.meta, "trace": trace})
        return replace(
            outcome, chunk=limited,
            tool_result=f"Accepted: execute the first {CONTROL_CAP} of {planned} controls. "
                        "The rest will not execute, including any gripper change in that suffix.",
        )

    def confirm_executed(self, played):
        pending = self._pending
        if pending is not None and (type(played) is not int or not 0 <= played <= len(pending.actions)):
            raise ValueError("Executed control count is outside the offered prefix.")
        super().confirm_executed(played)
        if pending is None:
            return
        # Planning eagerly latches the requested jaw goal. A discarded gripper
        # phase must not quietly close/open the hand on the following decision.
        self._gripper_goals = pending.prior_gripper_goals
        if played:
            last = pending.actions[played - 1].data
            for arm in ("left", "right"):
                value = float(np.asarray(last[f"{arm}_ee_joint_state"]).item())
                if not np.isfinite(value):
                    raise ValueError("Executed gripper command must be finite.")
                self._gripper_goals[arm] = value
        self._execution_feedback = (
            f"Previous move_eef executed {played} of {pending.planned_steps} controls. "
            "The remaining controls were discarded. Arrival error uses the original target."
        )
        self._pending = None

    def audit_config(self):
        return {
            "policy": "JustRUA", "model": "gpt-6-astra", "effort": "medium",
            "scene": {"init_seed": self._layout_id}, "task": self._task_name,
            "actions": "absolute world-frame move_eef", "max_decisions": None,
            "max_controls_per_motion": CONTROL_CAP, "full_history": True,
            "observations": "three RGB, measured robot and camera state and RGB-predicted depth",
            "coding": "native",
        }
