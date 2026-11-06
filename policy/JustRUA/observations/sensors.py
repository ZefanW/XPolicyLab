"""Sample robot state, camera calibration and local contacts at one physics step."""
from __future__ import annotations

import math


def clock(env):
    from .finger_contact import _physics_dt
    native = getattr(env.sim, 'unwrapped', env.sim)
    step = int(native._sim_step_counter)
    dt = _physics_dt(env)
    if step < 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid native physics clock')
    return {'physics_step': step, 'simulation_time_s': step * dt}


def sample(env, observation, env_idx=0):
    from . import cameras, contact_history, kinematics
    if env.num_envs != 1 or env_idx != 0:
        raise ValueError('Robot sensing requires one environment per process')
    before = clock(env)
    history = getattr(env, '_justrua_kinematic_history', None)
    if history is None:
        history = {}
        env._justrua_kinematic_history = history
    camera_rows = cameras.sample(env, observation, before, env_idx)
    robot_state = kinematics.sample(env, env_idx=env_idx, history=history)
    contact = contact_history.sample(env, env_idx=env_idx)
    if clock(env) != before:
        raise RuntimeError('Sensor observation advanced physics or crossed a state boundary')
    statuses = [x['status'] for x in camera_rows] + [robot_state['status'], contact['status']]
    status = 'available' if all(x == 'available' for x in statuses) else (
        'unavailable' if all(x == 'unavailable' for x in statuses) else 'partial')
    return {'schema': 'justrua-robot-v1', 'status': status, 'sample': before,
            'cameras': camera_rows, 'kinematics': robot_state, 'contact': contact}
