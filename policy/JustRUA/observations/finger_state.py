"""Measured X5 finger motion, wrist loads and contact forces.

No simulator imports or hooks run on import. Call install_before_create() after AppLauncher and before constructing the native environment. Then
sample() reads the state refreshed by the original observation path; it never
steps, resets, refreshes the scene, changes targets, or writes files.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path

from . import finger_contact


SCHEMA = "justrua-gripper-v1"
ARMS = ("left_arm", "right_arm")
_CALIBRATIONS = {}
SEMANTICS = {
    "joint_position": "Measured gripper joint coordinates; translation in metres, rotation in radians.",
    "control_coordinate": "Unclipped actual first joint position mapped through the controller scale/sign, not a physical opening percentage or fingertip gap. Zero is the controller target, which may lie beyond the mechanical limit.",
    "target_error": "Measured position minus last target; a nonzero error can result from an empty-gripper joint limit and is not proof of grasp.",
    "projected_joint_force": "PhysX incoming joint force projected along the joint motion axis, including transmitted loads; N for translation, N*m for rotation.",
    "end_effector_wrench": "Force and torque transmitted by the incoming joint of the named end link, in the child joint frame. Not motor torque, isolated object contact, or an installed hardware wrist sensor.",
    "contact": "Latest-physics-step world-frame normal contact resultant on each finger, excluding friction. Magnitude >= 1 N is a threshold flag, not proof of grasp or holding stability.",
    "excluded": "No controller-computed PD torque, object identity, object pose, contact point, or independent grasp-success inference.",
}


def install_before_create():
    """Explicit opt-in process-local contact instrumentation; see finger_contact."""
    return finger_contact.install_before_create()


def _plain(value):
    # OmegaConf is supplied by native RoboDojo, not required for offline import.
    if type(value).__module__.startswith("omegaconf"):
        from omegaconf import OmegaConf
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite reading")
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    converted = False
    for method in ("detach", "cpu", "numpy", "tolist"):
        if callable(getattr(value, method, None)):
            value = getattr(value, method)()
            converted = True
    if converted:
        return _plain(value)
    raise TypeError("Unsupported reading type")


def _unavailable(reason, exc=None, **metadata):
    result = {"status": "unavailable", "value": None, "reason_code": reason, **metadata}
    if exc is not None:
        result["error_type"] = type(exc).__name__
    return result


def _read(getter, reason, **metadata):
    try:
        value = _plain(getter())
        if value is None:
            return _unavailable(reason, **metadata)
        return {"status": "available", "value": value, **metadata}
    except Exception as exc:
        return _unavailable(reason, exc, **metadata)


def _number(value):
    value = _plain(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected a numeric reading")
    if not math.isfinite(value):
        raise ValueError("Non-finite reading")
    return float(value)


def _vector(value, length):
    value = _plain(value)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("Unexpected reading shape")
    return [_number(x) for x in value]


def _status(readings):
    count = sum(reading.get("status") == "available" for reading in readings)
    return "available" if count == len(readings) and count else "partial" if count else "unavailable"


def _empty_robot(arm, reason):
    row = {"arm_name": arm, "gripper_name": arm.replace("_arm", "_ee"),
           "status": "unavailable", "gripper_joint_names": ["joint7", "joint8"]}
    for key in ("gripper_dof_types", "measured_joint_position", "measured_joint_velocity",
                "previous_control", "command_scale", "actual_position_in_command_coordinates",
                "commanded_position_in_command_coordinates", "measured_minus_target",
                "projected_joint_force", "end_effector_wrench"):
        row[key] = _unavailable(reason)
    return row


def _normalization(raw, scale, sign):
    raw = _number(raw)
    low, high = _vector(scale, 2)
    if high <= low or sign not in (-1, 1):
        raise ValueError("Invalid controller scale/sign")
    return (raw - low) / (high - low) if sign == 1 else (high - raw) / (high - low)


def _previous(rm, robot, env_idx):
    entry = rm.control_manager.prev_control[env_idx].get(rm.process_name(robot.gripper_name))
    if entry is None:
        return None
    entry = _plain(entry)
    if not isinstance(entry, dict):
        raise ValueError("Invalid previous control")
    position = _vector(entry["position"], 2)
    velocity = entry.get("velocity")
    if velocity is not None:
        velocity = _plain(velocity)
        velocity = [_number(velocity)] * 2 if isinstance(velocity, (int, float)) else _vector(velocity, 2)
    # Explicit allowlist: no arbitrary controller metadata reaches model input.
    return {"position": position, "velocity": velocity}


def _dof_types(view, env_idx, indices):
    all_types = _plain(view.get_dof_types()[env_idx])
    types = [all_types[i] for i in indices]
    if any(isinstance(t, bool) or t != 1 for t in types):
        raise ValueError("X5 fingers must be prismatic")
    return types


def _wrench(art, view, env_idx, body):
    names = list(art.body_names)
    if names.count(body) != 1:
        raise ValueError("Unique end link not found")
    values = view.get_link_incoming_joint_force()[env_idx]
    if len(values) != len(names):
        raise ValueError("Body and wrench counts differ")
    return _vector(values[names.index(body)], 6)


def _geometry(robot, names, q):
    unknown = {"status": "unavailable", "reason_code": "gripper_geometry_unavailable",
               "joint_names": list(names), "joint_positions_m": None,
               "joint_limits_m": None, "joint_position_from_lower_m": None,
               "joint_limit_fractions": None, "joint_position_within_limits": None,
               "opening_travel_from_joint_zero_m": None, "opening_travel_status": "unavailable",
               "gap_m": None, "gap_status": "unknown", "usd_consistency": "unverified"}
    try:
        from .finger_geometry import calibration_from_urdf, aperture_from_positions
        path = Path(robot.urdf_path)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size, tuple(names))
        calibration = _CALIBRATIONS.get(key)
        if calibration is None:
            calibration = calibration_from_urdf(path, names)
            _CALIBRATIONS[key] = calibration
        if calibration.get("status") != "available":
            return unknown
        return _plain(aperture_from_positions(q, calibration))
    except Exception as exc:
        return {**unknown, "error_type": type(exc).__name__}


def _projected(view, env_idx, indices):
    forces = view.get_dof_projected_joint_forces()[env_idx]
    return [_number(forces[i]) for i in indices]


def _sample_robot(rm, robot, art, env_idx, arm):
    result = _empty_robot(arm, "reading_unavailable")
    indices = [_number(x) for x in _plain(robot.gripper_joint_indices)]
    if len(indices) != 2 or any(i < 0 or int(i) != i for i in indices) or indices[0] == indices[1]:
        raise ValueError("Expected two distinct gripper indices")
    indices = [int(i) for i in indices]
    names = [art.joint_names[i] for i in indices]
    if names != ["joint7", "joint8"]:
        raise ValueError("Finger contact sensing supports X5 joint7/joint8")
    result["gripper_joint_names"] = list(names)
    view = getattr(art, "root_physx_view", None)
    types = _read(lambda: _dof_types(view, env_idx, indices), "gripper_dof_types_unavailable")
    result["gripper_dof_types"] = types
    pos_units = ["m" if t == 1 else "rad" for t in types["value"]] if types["status"] == "available" else None
    vel_units = ["m/s" if t == 1 else "rad/s" for t in types["value"]] if pos_units else None
    force_units = ["N" if t == 1 else "N*m" for t in types["value"]] if pos_units else None
    q = _read(lambda: [_number(art.data.joint_pos[env_idx][i]) for i in indices],
              "gripper_position_unavailable", units=pos_units)
    qd = _read(lambda: [_number(art.data.joint_vel[env_idx][i]) for i in indices],
               "gripper_velocity_unavailable", units=vel_units)
    result.update(measured_joint_position=q, measured_joint_velocity=qd)
    if types["status"] != "available":
        q = result["measured_joint_position"] = _unavailable("gripper_units_unverified")
        qd = result["measured_joint_velocity"] = _unavailable("gripper_units_unverified")
    previous = _read(lambda: _previous(rm, robot, env_idx), "previous_control_unavailable")
    result["previous_control"] = previous
    scale = _read(lambda: _vector(robot.gripper_scale, 2), "controller_scale_unavailable", units=pos_units[0] if pos_units else None)
    sign_reading = _read(lambda: _number(_plain(robot.gripper_move)["sign"]), "controller_sign_unavailable")
    scale["sign"] = sign_reading["value"]
    if (sign_reading["status"] != "available" or scale["sign"] not in (-1, 1)
            or (scale["value"] is not None and scale["value"][1] <= scale["value"][0])):
        scale = _unavailable("controller_scale_or_sign_invalid")
    result["command_scale"] = scale
    if scale["status"] == "available" and sign_reading["status"] == "available":
        if q["status"] == "available":
            result["actual_position_in_command_coordinates"] = _read(
                lambda: _normalization(q["value"][0], scale["value"], scale["sign"]),
                "actual_control_coordinate_unavailable")
        if previous["status"] == "available":
            result["commanded_position_in_command_coordinates"] = _read(
                lambda: _normalization(previous["value"]["position"][0], scale["value"], scale["sign"]),
                "commanded_control_coordinate_unavailable")
    if q["status"] == previous["status"] == "available":
        result["measured_minus_target"] = _read(
            lambda: [a - b for a, b in zip(q["value"], previous["value"]["position"])],
            "target_difference_unavailable", units=pos_units)
    result["projected_joint_force"] = _read(
        lambda: _projected(view, env_idx, indices),
        "projected_joint_force_unavailable", units=force_units)
    if types["status"] != "available":
        result["projected_joint_force"] = _unavailable("gripper_units_unverified")
    body = getattr(robot, "ee_link_name", None)
    if body == "link6":
        result["end_effector_wrench"] = _read(lambda: _wrench(art, view, env_idx, body),
            "end_link_wrench_unavailable", body_name="link6", frame="child_joint", force_unit="N", torque_unit="N*m")
    result["gripper_geometry"] = _geometry(robot, names, q["value"])
    result["status"] = _status([result[k] for k in ("gripper_dof_types", "measured_joint_position",
        "measured_joint_velocity", "previous_control", "actual_position_in_command_coordinates",
        "projected_joint_force", "end_effector_wrench")])
    return result


def sample(env, env_idx=0):
    """Return a JSON-safe justrua-gripper-v1 record. Sample without moving the robot."""
    result = {"schema": SCHEMA, "status": "unavailable", "env_idx": env_idx,
              "semantics": dict(SEMANTICS), "robots": []}
    if isinstance(env_idx, bool) or not isinstance(env_idx, int) or env_idx < 0:
        raise ValueError("env_idx must be a nonnegative integer")
    for arm in ARMS:
        try:
            rm = env.robot_manager
            matches = [(i, robot) for i, robot in enumerate(rm.robot_list)
                       if getattr(robot, "arm_name", None) == arm]
            if len(matches) != 1:
                row = _empty_robot(arm, "robot_missing_or_ambiguous")
            else:
                index, robot = matches[0]
                row = _sample_robot(rm, robot, rm.robot_key[index], env_idx, arm)
        except Exception as exc:
            row = _empty_robot(arm, "robot_sensor_unavailable")
            row["error_type"] = type(exc).__name__
        result["robots"].append(row)
    try:
        result["contact"] = finger_contact.sample(env, env_idx)
    except Exception as exc:
        result["contact"] = finger_contact.unavailable_sample("contact_sensor_unavailable", exc)
    statuses = [row["status"] for row in result["robots"]] + [result["contact"]["status"]]
    result["status"] = "available" if all(s == "available" for s in statuses) else (
        "partial" if any(s in ("available", "partial") for s in statuses) else "unavailable")
    return _plain(result)
