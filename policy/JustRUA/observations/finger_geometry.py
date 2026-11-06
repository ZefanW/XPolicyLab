"""Pure, conservative gripper geometry derived from an external URDF.

Joint travel is not fingertip aperture. Neither a collision mesh bounding box nor
the distance between child-link origins establishes the inner working surfaces.
This module consequently leaves gap_m unknown. It imports no simulator and does
not open mesh assets, fetch resources, alter controls, or infer a grasp outcome.
"""

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET


_REASONS = frozenset({
    "invalid_joint_names", "urdf_unreadable", "urdf_invalid", "joint_not_found",
    "unsupported_joint_type", "invalid_joint_geometry", "invalid_joint_limits",
    "calibration_unavailable", "invalid_positions", "incomplete_positions",
})


def _number(value):
    if value is None or isinstance(value, (bool, str, bytes)):
        raise ValueError("not a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("not a finite number")
    return value


def _vector(text, default):
    result = [float(x) for x in (text if text is not None else default).split()]
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        raise ValueError("invalid vector")
    return result


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _rotate_rpy(v, rpy):
    """URDF fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    x, yy, z = v
    yy, z = cr*yy-sr*z, sr*yy+cr*z
    x, z = cp*x+sp*z, -sp*x+cp*z
    return [cy*x-sy*yy, sy*x+cy*yy, z]


def _pair_geometry(joints):
    if len(joints) != 2 or joints[0]["parent"] != joints[1]["parent"]:
        return None
    first, second = joints
    a, b = first["axis_parent"], second["axis_parent"]
    if abs(_dot(a, b) + 1.0) > 1e-8:
        return None
    delta = [x-y for x, y in zip(first["origin_xyz_m"], second["origin_xyz_m"])]
    projection = _dot(delta, a)
    if abs(projection) < 1e-10:
        # Without separation along the motion axis, the outward sign is unknown.
        return None
    axis = [x * (1.0 if projection > 0 else -1.0) for x in a]
    return {
        "parent": first["parent"], "opening_axis_parent": axis,
        "travel_coefficients": [_dot(a, axis), -_dot(b, axis)],
        "reference_origin_separation_m": _dot(delta, axis),
        "meaning": "projected child-frame origin separation; not inner working-surface gap",
    }


def calibration_from_urdf(path, joint_names):
    """Read joint geometry/limits, returning JSON data without paths or exceptions.

    The caller must bind this URDF to the actual runtime asset separately. An
    available result establishes only URDF kinematics, never USD consistency or
    calibrated inner-pad geometry. Both measured joints are retained even when a
    mimic relation is present. Unsupported or missing information fails closed.
    """
    result = {"status": "unavailable", "reason_code": None, "joint_names": [],
              "joints": [], "pair_geometry": None, "usd_consistency": "unverified",
              "inner_working_surfaces": "uncalibrated"}
    if isinstance(joint_names, (str, bytes)):
        result["reason_code"] = "invalid_joint_names"
        return result
    try:
        names = list(joint_names)
        if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError()
    except (TypeError, ValueError):
        result["reason_code"] = "invalid_joint_names"
        return result
    result["joint_names"] = names
    try:
        raw = Path(path).read_bytes()
    except (OSError, TypeError, ValueError):
        result["reason_code"] = "urdf_unreadable"
        return result
    try:
        root = ET.fromstring(raw)
        if root.tag != "robot":
            raise ValueError()
    except (ET.ParseError, ValueError):
        result["reason_code"] = "urdf_invalid"
        return result
    joints = []
    for name in names:
        matches = [n for n in root.findall("joint") if n.get("name") == name]
        if len(matches) != 1:
            result["reason_code"] = "joint_not_found"
            return result
        node = matches[0]
        if node.get("type") != "prismatic":
            result["reason_code"] = "unsupported_joint_type"
            return result
        try:
            parent, child = node.find("parent").get("link"), node.find("child").get("link")
            if not parent or not child:
                raise ValueError()
            origin, axis_node = node.find("origin"), node.find("axis")
            xyz = _vector(origin.get("xyz") if origin is not None else None, "0 0 0")
            rpy = _vector(origin.get("rpy") if origin is not None else None, "0 0 0")
            axis = _vector(axis_node.get("xyz") if axis_node is not None else None, "1 0 0")
            norm = math.sqrt(_dot(axis, axis))
            if not math.isfinite(norm) or norm < 1e-12:
                raise ValueError()
            axis_parent = _rotate_rpy([x/norm for x in axis], rpy)
        except (AttributeError, TypeError, ValueError, OverflowError):
            result["reason_code"] = "invalid_joint_geometry"
            return result
        try:
            limits = node.find("limit")
            lower, upper = float(limits.get("lower")), float(limits.get("upper"))
            if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
                raise ValueError()
        except (AttributeError, TypeError, ValueError):
            result["reason_code"] = "invalid_joint_limits"
            return result
        joints.append({"name": name, "type": "prismatic", "parent": parent, "child": child,
                       "origin_xyz_m": xyz, "origin_rpy_rad": rpy, "axis_parent": axis_parent,
                       "lower_m": lower, "upper_m": upper,
                       "mimic": dict(node.find("mimic").attrib) if node.find("mimic") is not None else None})
    result.update(status="available", joints=joints, pair_geometry=_pair_geometry(joints),
                  urdf_sha256=hashlib.sha256(raw).hexdigest())
    return result


def aperture_from_positions(qs, calibration):
    """Compute URDF-limit fractions and relative opening travel, leaving gap unknown.

    qs is a name->position mapping or a sequence in calibration joint order. The
    limit fraction is (q-lower)/(upper-lower), un-clipped: 0 is the URDF lower
    limit and 1 the upper limit, NOT a jaw-gap percentage or gripper command scale.
    Relative travel uses both measured positions and URDF parent-frame axes. For
    opposed outward X5 joints this is q7+q8, never an assumption of equal motion.
    """
    result = {"status": "unavailable", "reason_code": None, "joint_names": [],
              "joint_positions_m": [], "joint_limits_m": [], "joint_position_from_lower_m": [],
              "joint_limit_fractions": [], "joint_position_within_limits": [],
              "opening_travel_from_joint_zero_m": None, "opening_travel_status": "unavailable",
              "gap_m": None, "gap_status": "unknown",
              "gap_reason_code": "inner_working_surfaces_not_calibrated", "usd_consistency": "unverified"}
    if not isinstance(calibration, Mapping) or calibration.get("status") != "available":
        reason = calibration.get("reason_code") if isinstance(calibration, Mapping) else None
        result["reason_code"] = reason if isinstance(reason, str) and reason in _REASONS else "calibration_unavailable"
        return result
    try:
        joints = calibration["joints"]
        names = [j["name"] for j in joints]
        limits = [[_number(j["lower_m"]), _number(j["upper_m"])] for j in joints]
        if not joints or any(hi <= lo for lo, hi in limits):
            raise ValueError()
    except (KeyError, TypeError, ValueError, OverflowError):
        result["reason_code"] = "calibration_unavailable"
        return result
    result.update(joint_names=names, joint_limits_m=limits)
    try:
        if isinstance(qs, Mapping):
            values = [qs.get(name) for name in names]
        elif isinstance(qs, (str, bytes)):
            raise ValueError()
        else:
            values = list(qs)
            if len(values) != len(names):
                raise ValueError()
    except (TypeError, ValueError):
        values = [None] * len(names)
        result["reason_code"] = "invalid_positions"
    for value, (lo, hi) in zip(values, limits):
        try:
            q = _number(value)
        except (TypeError, ValueError, OverflowError):
            q = None
        result["joint_positions_m"].append(q)
        result["joint_position_from_lower_m"].append(q-lo if q is not None else None)
        result["joint_limit_fractions"].append((q-lo)/(hi-lo) if q is not None else None)
        result["joint_position_within_limits"].append(lo <= q <= hi if q is not None else None)
    count = sum(q is not None for q in result["joint_positions_m"])
    result["status"] = "available" if count == len(names) else "partial" if count else "unavailable"
    if count != len(names):
        result["reason_code"] = result["reason_code"] or "incomplete_positions"
        return result
    pair = calibration.get("pair_geometry")
    if pair is not None:
        try:
            coefficients = [_number(x) for x in pair["travel_coefficients"]]
            if len(coefficients) != len(names):
                raise ValueError()
            travel = _dot(coefficients, result["joint_positions_m"])
            if not math.isfinite(travel):
                raise ValueError()
            result.update(opening_travel_from_joint_zero_m=travel, opening_travel_status="urdf_kinematic_estimate")
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
    return result
