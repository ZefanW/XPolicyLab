"""Opt-in X5 contact reporting; importing this module changes nothing.

The installed process-local spawn hook adds finger ContactReportAPI, sets its
report threshold to zero, and enables PhysX contact processing. It does not
change sleep thresholds, drive gains, collisions, actions or simulator timing.
Returned measurements contain no scene paths or contacted-object identities.
"""
from __future__ import annotations

import functools
import importlib
import math


CONTACT_THRESHOLD_N = 1.0
_STATE = {"configured": False, "spawn_calls": 0, "activations": {},
          "views": {}, "errors": [], "contact_processing_enabled": False}


def clear():
    """Release native handles and reset telemetry once the simulator client exits."""
    _STATE['activations'].clear()
    _STATE['views'].clear()
    _STATE['errors'].clear()
    _STATE.update(configured=False, spawn_calls=0, contact_processing_enabled=False)


def _failure(reason, exc=None):
    result = {"status": "unavailable", "reason_code": reason}
    if exc is not None:
        # Exception text can include private filesystem or scene/object paths.
        result["error_type"] = type(exc).__name__
    return result


def _status():
    return {"status": "available" if _STATE["configured"] else "unavailable",
            "configured": _STATE["configured"], "spawn_calls": _STATE["spawn_calls"],
            "report_body_count": len(_STATE["activations"]),
            "contact_processing_enabled": _STATE["contact_processing_enabled"],
            "errors": list(_STATE["errors"])}


def _finger_prims(root):
    from pxr import Usd, UsdPhysics
    stage = root.GetStage()
    descendants = list(Usd.PrimRange(root))
    found = {}
    for prim in descendants:
        name = prim.GetName()
        if name in ("joint7", "joint8") and prim.IsA(UsdPhysics.Joint):
            targets = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
            if len(targets) != 1:
                continue
            body = stage.GetPrimAtPath(targets[0])
            if body and body.HasAPI(UsdPhysics.RigidBodyAPI):
                found[name] = body
    # X5's fixed joint-to-link names are only a fallback for absent joint prims.
    for joint, link in (("joint7", "link7"), ("joint8", "link8")):
        if joint not in found:
            matches = [p for p in descendants if p.GetName() == link
                       and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(matches) == 1:
                found[joint] = matches[0]
    return found


def _activate(root):
    from pxr import PhysxSchema
    import carb
    found = _finger_prims(root)
    if set(found) != {"joint7", "joint8"} or any(
            prim.GetName() != "link" + joint[-1] for joint, prim in found.items()):
        raise RuntimeError("Both named X5 finger rigid bodies are required")
    settings = carb.settings.get_settings()
    settings.set_bool("/physics/disableContactProcessing", False)
    _STATE["contact_processing_enabled"] = not settings.get_as_bool("/physics/disableContactProcessing")
    if not _STATE["contact_processing_enabled"]:
        raise RuntimeError("Contact processing could not be enabled")
    for joint, prim in found.items():
        path = str(prim.GetPath())
        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        api.CreateThresholdAttr().Set(0.0)
        _STATE["activations"][path] = {
            "joint_name": joint, "body_name": prim.GetName(),
            "root_path": str(root.GetPath()),
        }


def install_before_create():
    """Install once after AppLauncher, before create_eval_env/first physics reset.

    No simulator module is imported until this explicit opt-in call. A missing
    factory returns unavailable so the caller can reject sensor setup.
    """
    if _STATE["configured"]:
        return _status()
    try:
        module = importlib.import_module("env.robot_manager.robot_config.x5")
        original_factory = module.get_robot_config
    except Exception as exc:
        return _failure("x5_factory_unavailable", exc)

    @functools.wraps(original_factory)
    def factory(*args, **kwargs):
        cfg = original_factory(*args, **kwargs)
        original_spawn = cfg.spawn.func

        @functools.wraps(original_spawn)
        def spawn(*spawn_args, **spawn_kwargs):
            prim = original_spawn(*spawn_args, **spawn_kwargs)
            _STATE["spawn_calls"] += 1
            try:
                if prim is None or not hasattr(prim, "GetStage"):
                    raise RuntimeError("X5 spawn did not return its root prim")
                _activate(prim)
            except Exception as exc:
                _STATE["errors"].append(_failure("contact_report_activation_failed", exc))
            return prim

        cfg.spawn.func = spawn
        return cfg

    module.get_robot_config = factory
    _STATE["configured"] = True
    return _status()


def _physics_dt(env):
    sim = getattr(env, "sim", None)
    candidates = (
        lambda: sim.get_physics_dt(),
        lambda: sim.physics_dt,
        lambda: sim.sim.get_physics_dt(),
        lambda: env.physics_dt,
    )
    for getter in candidates:
        try:
            dt = float(getter())
        except (AttributeError, TypeError):
            continue
        if math.isfinite(dt) and dt > 0:
            return dt
        raise ValueError("Invalid physics dt")
    raise RuntimeError("Physics dt unavailable; control dt is not a substitute")


def _physics_view(env):
    try:
        from isaacsim.core.simulation_manager import SimulationManager
        view = SimulationManager.get_physics_sim_view()
        if view is not None:
            return view
    except (ImportError, AttributeError):
        pass
    for art in getattr(env.robot_manager, "robot_key", []):
        view = getattr(art, "_physics_sim_view", None)
        if view is not None:
            return view
    raise RuntimeError("Initialized physics view unavailable")


def _root_path(art, env_idx):
    expression = str(getattr(getattr(art, "cfg", None), "prim_path", ""))
    # Resolve only the native env namespace convention; never widen a regex.
    path = expression.replace("env_.*", "env_" + str(env_idx))
    if path and not any(char in path for char in "*[]{}()+?|\\"):
        if expression != path or env_idx == 0:
            return path
    view = getattr(art, "root_physx_view", None)
    paths = getattr(view, "prim_paths", None)
    if paths is not None:
        path = str(paths[env_idx])
        # ArticulationRootAPI may live on a child link rather than asset root.
        roots = {row["root_path"] for row in _STATE["activations"].values()
                 if path == row["root_path"] or path.startswith(row["root_path"] + "/")}
        if len(roots) == 1:
            return roots.pop()
    raise ValueError("Exact activated robot root path unavailable")


def _body_path(root, joint):
    matches = [path for path, row in _STATE["activations"].items()
               if row["root_path"] == root and row["joint_name"] == joint]
    if len(matches) != 1:
        raise RuntimeError("No unique pre-physics contact report activation")
    return matches[0]


def _force_vector(view, dt):
    values = view.get_net_contact_forces(dt=dt)
    for method in ("clone", "detach", "cpu", "numpy", "tolist"):
        if callable(getattr(values, method, None)):
            values = getattr(values, method)()
    if len(values) != 1 or len(values[0]) != 3:
        raise ValueError("Expected one exact finger and one force vector")
    vector = [float(x) for x in values[0]]
    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Non-finite contact force")
    return vector


def unavailable_sample(reason, exc=None):
    """An explicit unknown reading with the complete, fixed finger identities."""
    result = {"status": "unavailable", "frame": "world", "units": "N",
              "sample_window": "latest_physics_step", "physics_dt": None,
              "contact_threshold_n": CONTACT_THRESHOLD_N,
              "contact_threshold_rule": "resultant_magnitude >= 1.0",
              "quantity": "normal contact resultant on finger; friction excluded",
              "interpretation": "Threshold flag is not proof of grasp, object identity, or holding stability.",
              "fingers": []}
    for arm in ("left_arm", "right_arm"):
        for joint, body in (("joint7", "link7"), ("joint8", "link8")):
            row = {"arm_name": arm, "joint_name": joint, "body_name": body,
                   "status": "unavailable", "normal_force_w": None,
                   "resultant_magnitude": None, "contact_above_threshold": None}
            result["fingers"].append(row)
    result.update(_failure(reason, exc))
    return result


def sample(env, env_idx=0):
    """Read latest native-physics-step normal resultants, without advancing time."""
    result = unavailable_sample("contact_reporting_unavailable")
    rows = {(row["arm_name"], row["joint_name"]): row for row in result["fingers"]}
    if isinstance(env_idx, bool) or not isinstance(env_idx, int) or env_idx < 0:
        result["reason_code"] = "invalid_environment_index"
        return result
    if not _STATE["configured"]:
        result["reason_code"] = "contact_reporting_not_installed"
        return result
    try:
        import carb
        if carb.settings.get_settings().get_as_bool("/physics/disableContactProcessing"):
            raise RuntimeError("Contact processing disabled")
        dt, physics_view = _physics_dt(env), _physics_view(env)
        result["physics_dt"] = dt
    except Exception as exc:
        result.update(_failure("contact_backend_unavailable", exc))
        return result
    rm = env.robot_manager
    for arm in ("left_arm", "right_arm"):
        indices = [i for i, robot in enumerate(rm.robot_list) if getattr(robot, "arm_name", None) == arm]
        for joint in ("joint7", "joint8"):
            row = rows[(arm, joint)]
            try:
                if len(indices) != 1:
                    raise RuntimeError("Exactly one robot is required per arm")
                art = rm.robot_key[indices[0]]
                path = _body_path(_root_path(art, env_idx), joint)
                activation = _STATE["activations"][path]
                row["body_name"] = activation["body_name"]
                # A new initialized physics view cannot reuse an old contact view.
                cached = _STATE["views"].get(path)
                if cached is None or cached[0] is not physics_view:
                    cached = (physics_view, physics_view.create_rigid_contact_view(path))
                    _STATE["views"][path] = cached
                vector = _force_vector(cached[1], dt)
                magnitude = math.hypot(*vector)
                if not math.isfinite(magnitude):
                    raise ValueError("Non-finite contact resultant")
                row.update(status="available", normal_force_w=vector,
                           resultant_magnitude=magnitude,
                           contact_above_threshold=magnitude >= CONTACT_THRESHOLD_N)
            except Exception as exc:
                row.update(_failure("finger_contact_unavailable", exc))
    count = sum(row["status"] == "available" for row in result["fingers"])
    result["status"] = "available" if count == 4 else "partial" if count else "unavailable"
    if count:
        result.pop("reason_code", None)
    return result
