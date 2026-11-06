"""WebSocket endpoint for the environment-side JustRUA policy."""

import os

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info


class Model(ModelTemplate):
    """Expose the transport in sim and echo measured joint positions in debug."""

    def __init__(self, model_cfg):
        super().__init__()
        self.debug = os.environ.get("EVAL_ENV_TYPE", "sim") == "debug"
        if model_cfg.get("action_type") != "joint":
            raise ValueError("JustRUA uses RoboDojo's joint transport.")
        if model_cfg.get("env_cfg_type") != "arx_x5":
            raise ValueError("JustRUA supports the dual-arm arx_x5 configuration.")
        if self.debug:
            self.dims = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        self.reset()

    def reset(self):
        self.holds = {}

    def update_obs(self, obs):
        if not self.debug:
            return
        # The WS server has already decoded each camera into an RGB array.
        for key in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            color = obs["vision"][key]["color"]
            if not isinstance(color, np.ndarray) or color.ndim != 3 or color.shape[-1] != 3:
                raise ValueError(f"Expected decoded RGB for {key}")
        state = obs["state"]
        hold = {}
        for index, side in enumerate(("left", "right")):
            joints = (
                (f"{side}_arm_joint_state", self.dims["arm_dim"][index]),
                (f"{side}_ee_joint_state", self.dims["ee_dim"][index]),
            )
            for key, dimension in joints:
                values = np.asarray(state[key], dtype=np.float32)
                if values.shape != (dimension,):
                    raise ValueError(f"Unexpected shape for {key}: {values.shape}")
                hold[key] = values.copy()
        self.holds[int(obs.get("env_idx", 0))] = hold

    def update_obs_batch(self, obs_list):
        for obs in obs_list:
            self.update_obs(obs)

    def _debug_hold(self, index):
        if not self.debug:
            raise RuntimeError("JustRUA plans and executes actions inside the RoboDojo client.")
        if index not in self.holds:
            raise RuntimeError(f"A debug observation is required for environment {index}.")
        return [self.holds[index]]

    def get_action(self):
        return self._debug_hold(0)

    def get_action_batch(self, env_idx_list=None):
        if not self.debug:
            raise RuntimeError("JustRUA runs one actor per RoboDojo environment.")
        indices = list(self.holds) if env_idx_list is None else env_idx_list
        return [self._debug_hold(index) for index in indices]
