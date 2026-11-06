"""Attach sensors to RoboDojo's existing capture and reset boundaries."""
import functools
from contextlib import ExitStack

from . import cameras, contact_history, finger_contact, finger_state, sensors
from .readings import calibration, render_fingers, render_robot


def _release(env):
    collector = getattr(env, '_justrua_contact_collector', None)
    with ExitStack() as cleanup:
        cleanup.callback(finger_contact._STATE['views'].clear)
        if collector is not None:
            subscription = collector.subscription
            collector.subscription = None
            collector.views.clear()
            collector.env = None
            env._justrua_contact_collector = None
            unsubscribe = getattr(subscription, 'unsubscribe', None)
            if callable(unsubscribe):
                cleanup.callback(unsubscribe)
        cleanup.callback(cameras.detach, env)


def attach(env):
    """Call once immediately after the native factory returns, before the first reset."""
    if getattr(env, '_justrua_attached', False):
        return env
    if env.num_envs != 1:
        raise ValueError('JustRUA uses one two-arm environment per process')
    original_reset = env.reset
    original_capture_reset = env.capture_manager.reset
    original_observe = env.get_obs_batch
    original_close = env.close
    simulator = getattr(env, 'sim', None)
    simulator = getattr(simulator, 'unwrapped', simulator)
    original_index_reset = getattr(simulator, '_reset_idx', None)
    hooks = ExitStack()

    @functools.wraps(original_reset)
    def reset(*args, **kwargs):
        env._justrua_kinematic_history = {}
        finger_contact._STATE['views'].clear()
        return original_reset(*args, **kwargs)

    @functools.wraps(original_capture_reset)
    def reset_capture(*args, **kwargs):
        result = original_capture_reset(*args, **kwargs)
        # Attach before the existing warm-up renders. Never step to collect a sensor.
        cameras.prepare(env)
        return result

    @functools.wraps(original_observe)
    def observe(*args, **kwargs):
        cameras.prepare(env)
        before = sensors.clock(env)
        rows = original_observe(*args, **kwargs)
        for row in rows:
            if type(row.get('env_idx')) is not int or row['env_idx'] != 0:
                raise ValueError('Unexpected environment in robot observations')
            state = row['state']
            if 'justrua_sensors' in state:
                raise ValueError('Unexpected existing JustRUA sensor record')
            fingers = finger_state.sample(env)
            robot = sensors.sample(env, row)
            if sensors.clock(env) != before or robot['sample'] != before:
                raise RuntimeError('RGB and robot sensors crossed a physics step')
            calibrated_sizes = {c['name']: c['value']['resolution_wh'] for c in robot['cameras']
                                if c['status'] == 'available'}
            text = '\n'.join((f"sample_physics_step={before['physics_step']}",
                              render_fingers(fingers), render_robot(robot), calibration(robot)))
            # Only compact, selected numeric measurements cross the policy boundary.
            row['state'] = {**state, 'justrua_sensors': {
                'physics_step': before['physics_step'], 'calibrated_sizes': calibrated_sizes, 'text': text}}
        return rows

    @functools.wraps(original_close)
    def close(*args, **kwargs):
        with ExitStack() as cleanup:
            # The native environment must still be closed if a sensor fails to detach.
            cleanup.callback(original_close, *args, **kwargs)
            hooks.close()

    # Release resources and restore native methods on close or partial setup failure.
    try:
        hooks.callback(setattr, env, '_justrua_attached', False)
        hooks.callback(setattr, env, 'close', original_close)
        hooks.callback(setattr, env, 'get_obs_batch', original_observe)
        hooks.callback(setattr, env.capture_manager, 'reset', original_capture_reset)
        hooks.callback(setattr, env, 'reset', original_reset)
        if callable(original_index_reset):
            hooks.callback(setattr, simulator, '_reset_idx', original_index_reset)
        hooks.callback(_release, env)
        env.reset = reset
        env.capture_manager.reset = reset_capture
        contact_history.attach(env)
        env.get_obs_batch = observe
        env.close = close
        env._justrua_attached = True
    except BaseException:
        hooks.close()
        raise
    return env


def install_after_app():
    """Temporarily install the native factory and X5 spawn hook before the first reset."""
    import importlib
    module = importlib.import_module('src.eval_client.eval_env')
    original = module.create_eval_env
    if getattr(original, '_justrua_factory', False):
        return ExitStack()
    robot_config = importlib.import_module('env.robot_manager.robot_config.x5')
    original_robot_factory = robot_config.get_robot_config

    @functools.wraps(original)
    def create(*args, **kwargs):
        env = original(*args, **kwargs)
        try:
            return attach(env)
        except BaseException:
            env.close()
            raise

    create._justrua_factory = True
    with ExitStack() as rollback:
        if not finger_contact._STATE['configured']:
            rollback.callback(finger_contact.clear)
            rollback.callback(setattr, robot_config, 'get_robot_config', original_robot_factory)
        result = finger_contact.install_before_create()
        if result['status'] != 'available':
            raise RuntimeError(f"Could not install the X5 contact reporter: {result.get('reason_code')}")
        rollback.callback(setattr, module, 'create_eval_env', original)
        module.create_eval_env = create
        return rollback.pop_all()
