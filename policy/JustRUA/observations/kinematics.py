"""Robot kinematics; explicit sampling only, no simulator side effects.

No scene objects, task metadata, rewards, depth, or object identities are read.
The caller owns history and clears it on reset. Simulation time, not wall time,
defines finite differences. All transforms are column-vector homogeneous T.
"""
from __future__ import annotations

from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np

from . import robot_geometry

SCHEMA = "justrua-kinematics-v1"
TCP_OFFSET_M = .1501
CHARACTERISTIC_LENGTH_M = .5
_URDF_CACHE = {}


def unavailable(reason, source="robot_self"):
    return {"status": "unavailable", "value": None, "reason_code": reason, "source": source}


def available(value, source, frame=None, units=None):
    result = {"status": "available", "value": value, "source": source}
    if frame is not None:
        result["frame"] = frame
    if units is not None:
        result["units"] = units
    return result


def _read(func, reason, source, frame=None, units=None):
    try:
        return available(func(), source, frame, units)
    except Exception:
        return unavailable(reason, source)


def _array(value, shape=None):
    return robot_geometry.array(value, shape)


def _rotation_quat(quat):
    q = _array(quat, (4,))
    if abs(np.linalg.norm(q)-1) > 1e-4:
        raise ValueError("unnormalized quaternion")
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def transform_from_pose(pose, env_origin=None):
    pose = _array(pose, (7,))
    result = np.eye(4)
    result[:3, :3] = _rotation_quat(pose[3:])
    result[:3, 3] = pose[:3]
    if env_origin is not None:
        result[:3, 3] -= _array(env_origin, (3,))
    return result


def tcp_transform(link_transform):
    result = _array(link_transform, (4, 4)).copy()
    result[:3, 3] += result[:3, :3] @ np.array([TCP_OFFSET_M, 0., 0.])
    return result


def shift_twist(twist, displacement):
    """Move reference point: v_new = v_old + omega cross (p_new-p_old)."""
    twist = _array(twist, (6,)).copy()
    twist[:3] += np.cross(twist[3:], _array(displacement, (3,)))
    return twist


def _axis_angle(axis, angle):
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross @ cross)


def _xyz(node, name, default="0 0 0"):
    return _array([float(v) for v in (node.get(name, default) if node is not None else default).split()], (3,))


def _origin(node):
    result = np.eye(4)
    result[:3, 3] = _xyz(node, "xyz")
    r, p, y = _xyz(node, "rpy")
    result[:3, :3] = (_axis_angle(np.array([0., 0., 1.]), y) @
                      _axis_angle(np.array([0., 1., 0.]), p) @
                      _axis_angle(np.array([1., 0., 0.]), r))
    return result


def load_robot_chain(path, target="link6"):
    """Parse only the robot URDF chain, never task/scene files."""
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, target)
    if key in _URDF_CACHE:
        return _URDF_CACHE[key]
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError("not a robot URDF")
    by_child = {}
    for joint in root.findall("joint"):
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        kind = joint.get("type")
        if child in by_child or kind not in ("fixed", "revolute", "continuous", "prismatic"):
            raise ValueError("unsupported robot chain")
        axis = _xyz(joint.find("axis"), "xyz", "1 0 0")
        if np.linalg.norm(axis) < 1e-10:
            raise ValueError("zero axis")
        axis /= np.linalg.norm(axis)
        limit = joint.find("limit")
        bounds = None
        if kind in ("revolute", "prismatic") and limit is not None:
            bounds = _array([float(limit.get("lower")), float(limit.get("upper"))], (2,))
            if bounds[1] <= bounds[0]:
                raise ValueError("invalid joint bounds")
        by_child[child] = {"name": joint.get("name"), "parent": parent, "child": child,
                           "kind": kind, "origin": _origin(joint.find("origin")), "axis": axis,
                           "bounds": bounds}
    chain, current, seen = [], target, set()
    while current in by_child:
        if current in seen:
            raise ValueError("URDF cycle")
        seen.add(current)
        joint = by_child[current]
        chain.append(joint)
        current = joint["parent"]
    chain.reverse()
    if not chain:
        raise ValueError("missing robot chain")
    result = {"root_link": current, "joints": chain}
    _URDF_CACHE[key] = result
    return result


def forward_kinematics(chain, q, root_transform):
    """URDF FK and geometric TCP Jacobian in the same env/world axes."""
    transform = _array(root_transform, (4, 4)).copy()
    frames = {chain["root_link"]: transform.copy()}
    axes, origins, names, kinds = [], [], [], []
    for joint in chain["joints"]:
        transform = transform @ joint["origin"]
        kind = joint["kind"]
        if kind != "fixed":
            angle = float(q[joint["name"]])
            if not np.isfinite(angle):
                raise ValueError("nonfinite joint")
            names.append(joint["name"])
            kinds.append(kind)
            axes.append(transform[:3, :3] @ joint["axis"])
            origins.append(transform[:3, 3].copy())
            motion = np.eye(4)
            if kind == "prismatic":
                motion[:3, 3] = joint["axis"]*angle
            else:
                motion[:3, :3] = _axis_angle(joint["axis"], angle)
            transform = transform @ motion
        frames[joint["child"]] = transform.copy()
    tcp = tcp_transform(transform)
    columns = []
    for axis, origin, kind in zip(axes, origins, kinds):
        columns.append(np.r_[axis, [0., 0., 0.]] if kind == "prismatic" else
                       np.r_[np.cross(axis, tcp[:3, 3]-origin), axis])
    return frames, tcp, np.asarray(columns).T, names


def normalized_singularity(jacobian, joint_ranges):
    """Dimensionless inverse condition score for normalized joint coordinates.

    Jn = diag(1/L,1/L,1/L,1,1,1) J diag(q_upper-q_lower), L=.5m.
    Sigma_min/sigma_max in [0,1], with zero for rank-deficient 6D control. It is
    a local kinematic condition index, not IK feasibility or collision safety.
    """
    matrix = _array(jacobian)
    ranges = _array(joint_ranges)
    if matrix.ndim != 2 or matrix.shape[0] != 6 or ranges.shape != (matrix.shape[1],) or np.any(ranges <= 0):
        raise ValueError("invalid Jacobian normalization")
    scaled = matrix * ranges[None, :]
    scaled[:3] /= CHARACTERISTIC_LENGTH_M
    values = np.linalg.svd(scaled, compute_uv=False)
    low = float(values[-1]) if len(values) == 6 else 0.
    high = float(values[0]) if len(values) else 0.
    score = max(0., min(1., low/high)) if high > 1e-12 else 0.
    return {"inverse_condition": score, "singular_values": values.tolist(),
            "characteristic_length_m": CHARACTERISTIC_LENGTH_M,
            "normalization": "diag(1/L,1/L,1/L,1,1,1) J diag(joint_upper-joint_lower)",
            "meaning": "0 singular or deficient for 6D TCP control; 1 isotropic under this normalization; not reachability"}


def _time(env):
    sim = env.sim
    sim = getattr(sim, "unwrapped", sim)
    steps, dt = sim._sim_step_counter, float(sim.physics_dt)
    if isinstance(steps, bool) or int(steps) != steps or steps < 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("simulation clock unavailable")
    stamp = float(steps)*dt
    if not np.isfinite(stamp):
        raise ValueError("nonfinite simulation time")
    return stamp


def _acceleration(history, key, time_s, twist):
    source = "finite_difference_of_measured_tcp_twist"
    if history is None or time_s is None:
        return unavailable("simulation_time_or_history_unavailable", source)
    old = history.get(key)
    current = {"time": time_s, "twist": twist.tolist()}
    if old is not None and time_s == old["time"]:
        return old.get("acceleration", unavailable("no_positive_time_interval", source))
    result = unavailable("first_sample_or_clock_reset", source)
    if old is not None and time_s > old["time"]:
        interval = time_s-old["time"]
        value = ((twist-_array(old["twist"], (6,)))/interval).tolist()
        result = available({"linear_m_s2": value[:3], "angular_rad_s2": value[3:],
                            "interval_s": interval, "kind": "interval_average"}, source, "env", "m/s^2,rad/s^2")
    current["acceleration"] = result
    history[key] = current
    return result


def _empty(arm):
    return {"arm_name": arm, "status": "unavailable", **{name: unavailable("robot_state_unavailable") for name in
            ("joints", "frames", "fk_validation", "tcp_twist", "tcp_acceleration", "jacobian", "singularity", "tool_geometry", "wrist_wrench_axes")}}


def wrench_axes(raw, T_env_from_link, T_link_from_joint):
    """Re-express incoming joint wrench; torque reference stays joint anchor."""
    raw = _array(raw, (6,))
    link, local = _array(T_env_from_link, (4, 4)), _array(T_link_from_joint, (4, 4))
    joint = link @ local
    force_tool, torque_tool = local[:3, :3] @ raw[:3], local[:3, :3] @ raw[3:]
    force_env, torque_env = link[:3, :3] @ force_tool, link[:3, :3] @ torque_tool
    return {"force_env_n": force_env.tolist(), "torque_env_nm": torque_env.tolist(),
            "force_tool_n": force_tool.tolist(), "torque_tool_nm": torque_tool.tolist(),
            "joint_anchor_env_m": joint[:3, 3].tolist(), "T_link6_from_child_joint": local.tolist(),
            "force_along_approach_n": float(force_tool[0]), "force_along_closing_axis_n": float(force_tool[1]),
            "torque_about_approach_nm": float(torque_tool[0]), "torque_about_closing_axis_nm": float(torque_tool[1]),
            "approach_axis_link6": [1., 0., 0.], "closing_axis_link6": [0., 1., 0.],
            "torque_reference": "incoming_joint_anchor_not_tcp",
            "meaning": "total_incoming_joint_wrench_not_isolated_external_contact_or_motor_torque"}


def _sample_robot(env, robot, art, env_idx, time_s, history):
    row = _empty(robot.arm_name)
    names = list(art.joint_names)
    indices = [int(i) for i in robot.arm_joint_indices]
    arm_names = [names[i] for i in indices]
    q = _array(art.data.joint_pos[env_idx])
    if any(i < 0 or i >= len(q) for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("invalid arm indices")
    chain = load_robot_chain(robot.urdf_path, robot.ee_link_name)
    joints = [j for j in chain["joints"] if j["kind"] != "fixed"]
    if [j["name"] for j in joints] != arm_names or any(j["kind"] not in ("revolute", "continuous") for j in joints):
        raise ValueError("unsupported arm chain")
    # Read runtime physical bounds when present; URDF bounds are a labelled
    # fallback, never silently called the active simulator/controller limits.
    try:
        bounds = _array(art.data.joint_pos_limits[env_idx])[indices]
        limits_source = "measured_articulation_joint_pos_limits"
    except AttributeError:
        bounds = np.asarray([j["bounds"] for j in joints], dtype=float)
        limits_source = "robot_urdf_limits_unverified_in_simulator"
    if bounds.shape != (len(indices), 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("arm limits unavailable")
    q_arm = q[indices]
    try:
        qd = _array(art.data.joint_vel[env_idx], q.shape)[indices].tolist()
    except Exception:
        qd = None
    try:
        velocity_limits = _array(art.data.joint_vel_limits[env_idx], q.shape)[indices]
        if np.any(velocity_limits <= 0):
            raise ValueError("invalid velocity limits")
        velocity_margin = (velocity_limits-np.abs(qd)).tolist() if qd is not None else None
        velocity_limits = velocity_limits.tolist()
    except Exception:
        velocity_limits, velocity_margin = None, None
    lo_margin, hi_margin = q_arm-bounds[:, 0], bounds[:, 1]-q_arm
    row["joints"] = available({"names": arm_names, "position_rad": q_arm.tolist(), "velocity_rad_s": qd,
        "velocity_status": "available" if qd is not None else "unavailable", "limits_rad": bounds.tolist(), "limits_source": limits_source,
        "velocity_limits_rad_s": velocity_limits, "velocity_margin_rad_s": velocity_margin,
        "velocity_limits_source": "articulation_joint_vel_limits" if velocity_limits is not None else "unavailable",
        "lower_margin_rad": lo_margin.tolist(), "upper_margin_rad": hi_margin.tolist(),
        "nearest_margin_rad": np.minimum(lo_margin, hi_margin).tolist(),
        "limit_fraction_unclipped": (lo_margin/(bounds[:, 1]-bounds[:, 0])).tolist()},
        "measured_joint_state_and_robot_urdf", units="rad,rad/s")
    origin = _array(env.robot_manager.scene.env_origins[env_idx], (3,))
    body_names = list(art.body_names)
    poses = _array(art.data.body_link_pose_w[env_idx], (len(body_names), 7))
    transforms = {name: transform_from_pose(pose, origin) for name, pose in zip(body_names, poses)}
    tcp = tcp_transform(transforms[robot.ee_link_name])
    row["frames"] = available({"links": [{"body_name": name, "T_env_from_body": transforms[name].tolist()} for name in body_names],
                                "T_env_from_tcp": tcp.tolist(), "tcp_offset_link6_m": [TCP_OFFSET_M, 0., 0.]},
                               "measured_robot_body_link_pose", "env", "m,rotation_matrix")
    fk, computed_tcp, jacobian, jac_names = forward_kinematics(chain, dict(zip(names, q)), transforms[chain["root_link"]])
    errors = [np.max(np.abs(t-transforms[name])) for name, t in fk.items() if name in transforms]
    max_error = float(max(errors))
    matched = max_error <= 1e-4 and np.max(np.abs(computed_tcp-tcp)) <= 1e-4
    row["fk_validation"] = available({"matches_measured_links": bool(matched), "max_transform_abs_error": max_error,
                                      "tolerance": 1e-4}, "robot_urdf_vs_measured_body_link_poses")
    if matched:
        row["jacobian"] = available({"matrix": jacobian.tolist(), "joint_names": jac_names,
                                      "rows": ["vx", "vy", "vz", "wx", "wy", "wz"], "point": "grasp_point"},
                                     "urdf_geometric_jacobian_validated_by_fk", "env", "m/rad,rad/rad")
        row["singularity"] = available(normalized_singularity(jacobian, bounds[:, 1]-bounds[:, 0]),
                                         "normalized_urdf_jacobian_svd", units="dimensionless")
    else:
        row["jacobian"] = unavailable("urdf_usd_fk_mismatch")
        row["singularity"] = unavailable("urdf_usd_fk_mismatch")
    body_idx = body_names.index(robot.ee_link_name)
    try:
        try:
            link_twist = _array(art.data.body_link_vel_w[env_idx][body_idx], (6,))
            velocity_source = "measured_body_link_vel_w"
        except (AttributeError, NotImplementedError):
            com_twist = _array(art.data.body_com_vel_w[env_idx][body_idx], (6,))
            com_pose = _array(art.data.body_com_pose_w[env_idx][body_idx], (7,))
            link_twist = shift_twist(com_twist, poses[body_idx, :3]-com_pose[:3])
            velocity_source = "measured_body_com_vel_w_shifted_to_link_origin"
        offset = tcp[:3, 3]-transforms[robot.ee_link_name][:3, 3]
        twist = shift_twist(link_twist, offset)
        row["tcp_twist"] = available({"linear_m_s": twist[:3].tolist(), "angular_rad_s": twist[3:].tolist(),
                                      "point": "grasp_point", "formula": "v_tcp=v_link+omega_cross_R_link*[0.1501,0,0]"},
                                     velocity_source, "env", "m/s,rad/s")
        row["tcp_acceleration"] = _acceleration(history, (id(env), env_idx, robot.arm_name), time_s, twist)
    except Exception:
        row["tcp_twist"] = unavailable("measured_velocity_unavailable")
    row["tool_geometry"] = robot_geometry.sample(robot, art, env_idx, transforms)
    try:
        joint_transform = robot_geometry.incoming_joint_frame(art, env_idx, robot.ee_link_name)
        raw_wrench = art.root_physx_view.get_link_incoming_joint_force()[env_idx][body_idx]
        row["wrist_wrench_axes"] = available(wrench_axes(raw_wrench, transforms[robot.ee_link_name], joint_transform),
            "incoming_joint_wrench_with_self_usd_joint_frame", "env_and_link6_axes", "N,N*m")
    except Exception:
        row["wrist_wrench_axes"] = unavailable("incoming_joint_frame_or_wrench_unverified")
    statuses = [value["status"] for key, value in row.items() if isinstance(value, dict) and "status" in value]
    row["status"] = "available" if all(s == "available" for s in statuses) else "partial" if any(s == "available" for s in statuses) else "unavailable"
    # Refuse non-finite derived arithmetic as well as non-finite native arrays.
    json.dumps(row, allow_nan=False)
    return row


def sample(env, env_idx=0, history=None):
    """Return JSON-safe robot-only observations; caller retains history per episode."""
    if isinstance(env_idx, bool) or not isinstance(env_idx, int) or env_idx < 0:
        raise ValueError("env_idx must be a nonnegative integer")
    try:
        time_s = _time(env)
    except Exception:
        time_s = None
    result = {"schema": SCHEMA, "status": "unavailable", "sample_time_s": time_s, "robots": [],
              "bimanual_relative_pose": unavailable("both_tcp_poses_required")}
    for arm in ("left_arm", "right_arm"):
        row = _empty(arm)
        try:
            rm = env.robot_manager
            candidates = [(i, robot) for i, robot in enumerate(rm.robot_list) if getattr(robot, "arm_name", None) == arm]
            if len(candidates) != 1:
                raise ValueError("robot missing or ambiguous")
            index, robot = candidates[0]
            row = _sample_robot(env, robot, rm.robot_key[index], env_idx, time_s, history)
        except Exception:
            pass
        result["robots"].append(row)
    a, b = result["robots"]
    if a["frames"]["status"] == b["frames"]["status"] == "available":
        left = _array(a["frames"]["value"]["T_env_from_tcp"], (4, 4))
        right = _array(b["frames"]["value"]["T_env_from_tcp"], (4, 4))
        try:
            relative = _array(np.linalg.inv(left) @ right, (4, 4))
            distance = float(np.linalg.norm(relative[:3, 3]))
            if not np.isfinite(distance):
                raise ValueError("nonfinite relative distance")
            result["bimanual_relative_pose"] = available({"T_left_tcp_from_right_tcp": relative.tolist(),
                "tcp_distance_m": distance}, "measured_robot_tcp_transforms", "left_tcp", "m,rotation_matrix")
        except Exception:
            result["bimanual_relative_pose"] = unavailable("relative_transform_unavailable")
    statuses = [r["status"] for r in result["robots"]]
    result["status"] = "available" if all(s == "available" for s in statuses) else "partial" if any(s in ("available", "partial") for s in statuses) else "unavailable"
    return result
