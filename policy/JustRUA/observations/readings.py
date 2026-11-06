"""Compact, unit-labelled projections of measured robot and camera data."""
from collections.abc import Mapping
import math

GRIPPER_GUIDE = """Gripper readings:
All finger pairs are [joint7, joint8]. Position and velocity are actual prismatic joint readings; drive_target is the previous controller position target, not measured motion. The base gripper value is the controller command. Finger position is joint displacement, NOT fingertip gap or opening percentage.
Finger joint force is incoming joint force projected on the prismatic motion axis, in N, not motor torque. normal_N is each finger's normal-contact resultant magnitude, without friction; contact_ge1N is yes iff its unrounded magnitude is >=1 N. These are latest-physics-step samples. Below threshold does not prove no contact: forces can cancel or contact can occur between samples. Contact does not establish grasp success, no slip or a stable hold.
Wrist total force/torque are norms of the incoming link6 joint wrench (N, N*m), including dynamics and transmitted loads, NOT isolated external forces. Torque is about the incoming joint anchor, not the TCP. Direction is omitted.
unknown means unavailable or invalid, never zero. Display rounding: position 0.01 mm, velocity 0.01 mm/s, force 0.01 N, torque 0.001 N*m. Extra normal-force digits are retained when rounding would cross the 1 N threshold; flags use unrounded readings. Actions, images and feedback are unchanged."""


def _finite(value):
    if type(value) not in (int, float):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except (OverflowError, ValueError):
        return None


def _vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return [None] * length
    return [_finite(item) for item in value]


def _reading(robot, key, length, *, units=None):
    reading = robot.get(key)
    if not isinstance(reading, Mapping) or reading.get('status') != 'available':
        return [None] * length
    if units is not None and reading.get('units') != units:
        return [None] * length
    return _vector(reading.get('value'), length)


def _number(value, precision=2, scale=1):
    value = _finite(value)
    if value is None:
        return 'unknown'
    value = _finite(value * scale)
    if value is None:
        return 'unknown'
    rounded = round(value, precision)
    if rounded == 0:
        return '0'
    text = f'{rounded:.{precision}f}'
    return text.rstrip('0').rstrip('.') if '.' in text else text


def _numbers(values, *, precision=2, scale=1):
    return '[' + ', '.join(_number(x, precision, scale) for x in values) + ']'


def _normal_number(value):
    value = _finite(value)
    if value is not None and value < 1.0 and round(value, 2) == 1.0:
        # Do not show "1 N, no": retain the fewest additional decimal places
        # that keep this below-threshold reading visibly below the threshold.
        for precision in range(3, 18):
            text = _number(value, precision)
            if float(text) < 1.0:
                return text
        return repr(value)
    return _number(value)


def _norm(values):
    if any(value is None for value in values):
        return None
    return _finite(math.hypot(*values))


def _robot_map(payload):
    result = {}
    for robot in payload.get('robots', []):
        if not isinstance(robot, Mapping):
            continue
        arm = robot.get('arm_name')
        if arm not in ('left_arm', 'right_arm'):
            continue
        if arm in result:
            raise ValueError('Duplicate arm identity')
        if robot.get('gripper_joint_names') != ['joint7', 'joint8']:
            raise ValueError('Finger ordering must be joint7, joint8')
        result[arm] = robot
    return result


def _contact_map(payload):
    result = {}
    contact = payload.get('contact')
    if not isinstance(contact, Mapping):
        return result
    if (contact.get('frame') != 'world' or contact.get('units') != 'N'
            or contact.get('sample_window') != 'latest_physics_step'):
        return result
    for finger in contact.get('fingers', []):
        if not isinstance(finger, Mapping):
            continue
        key = (finger.get('arm_name'), finger.get('joint_name'))
        if key[0] not in ('left_arm', 'right_arm') or key[1] not in ('joint7', 'joint8'):
            continue
        if key in result:
            raise ValueError('Duplicate finger identity')
        magnitude = None
        if finger.get('status') == 'available':
            vector = _vector(finger.get('normal_force_w'), 3)
            stated = _finite(finger.get('resultant_magnitude'))
            computed = _norm(vector)
            if (stated is not None and stated >= 0 and computed is not None
                    and math.isclose(stated, computed, rel_tol=1e-6, abs_tol=1e-8)):
                magnitude = stated
        result[key] = magnitude
    return result


def render_fingers(grippers) -> str:
    """Format selected original readings; missing/non-finite values stay unknown.

    Wrong schema or ambiguous identities are explicit errors. Unknown fields and
    free-form source/error strings never enter the output.
    """
    if not isinstance(grippers, Mapping) or grippers.get('schema') != 'justrua-gripper-v1':
        raise ValueError('Expected a JustRUA gripper observation')
    robots = _robot_map(grippers)
    contacts = _contact_map(grippers)
    lines = ['Gripper readings:']
    for side in ('left', 'right'):
        robot = robots.get(side + '_arm', {})
        position = _reading(robot, 'measured_joint_position', 2, units=['m', 'm'])
        velocity = _reading(robot, 'measured_joint_velocity', 2, units=['m/s', 'm/s'])
        joint_force = _reading(robot, 'projected_joint_force', 2, units=['N', 'N'])
        previous = robot.get('previous_control')
        target = [None, None]
        if isinstance(previous, Mapping) and previous.get('status') == 'available':
            previous_value = previous.get('value')
            if isinstance(previous_value, Mapping):
                target = _vector(previous_value.get('position'), 2)
        magnitudes = [contacts.get((side + '_arm', joint)) for joint in ('joint7', 'joint8')]
        flags = ['unknown' if value is None else 'yes' if value >= 1.0 else 'no'
                 for value in magnitudes]
        wrench = _reading(robot, 'end_effector_wrench', 6)
        wrench_meta = robot.get('end_effector_wrench', {})
        if (not isinstance(wrench_meta, Mapping)
                or wrench_meta.get('body_name') != 'link6'
                or wrench_meta.get('frame') != 'child_joint'
                or wrench_meta.get('force_unit') != 'N'
                or wrench_meta.get('torque_unit') != 'N*m'):
            wrench = [None] * 6
        lines.extend([
            f'{side}_finger_position_mm=' + _numbers(position, scale=1000),
            f'{side}_finger_velocity_mm_s=' + _numbers(velocity, scale=1000),
            f'{side}_drive_target_mm=' + _numbers(target, scale=1000),
            f'{side}_finger_joint_force_N=' + _numbers(joint_force),
            f'{side}_normal_N=[' + ', '.join(_normal_number(x) for x in magnitudes) + ']'
            + '; contact_ge1N=[' + ', '.join(flags) + ']',
            f'{side}_wrist_total_force_N=' + _number(_norm(wrench[:3])),
            f'{side}_wrist_total_torque_Nm=' + _number(_norm(wrench[3:]), precision=3),
        ])
    return '\n'.join(lines)


GEOMETRY_GUIDE = """Robot and camera geometry:
env is the environment-local, world-aligned frame used by the end-effector targets. All xyz vectors use its axes. Camera right/down/forward are unit vectors in env; pixel centers are (column+0.5,row+0.5). Calibration [fx,fy,cx,cy] is in pixels; it is not depth. Camera poses below belong to the current RGB capture. Calibration belongs to the RGB cameras. The depth predictor uses its own learned field of view.
inner_gap_mm is the signed separation of the robot's inner finger mesh surfaces, NOT a verified collision gap. TCP speeds are measured at the grasp point. Finger pairs are [joint7,joint8]. friction_N is the tangential resultant magnitude, not a coefficient. contact_centers_env_m uses absolute normal-force weights, only for fingers with >=1 N contact and valid details; otherwise unknown. It gives no object identity or pose.
contact_age_ms is the observed uninterrupted >=1 N contact duration: 0 means currently below threshold; >= means the start was not observed and this is a lower bound. recent_peak_normal_N is over the stated rolling window (at most 512 physics steps), not the whole episode. Unknown values remain unknown. Contact and force do not imply grasp success or absence of slip. Values are display-rounded."""


def number(x, digits=2, scale=1):
    if type(x) not in (int, float) or not math.isfinite(x):
        return 'unknown'
    x = round(x * scale, digits)
    if x == 0:
        return '0'
    rendered = f'{x:.{digits}f}'
    return rendered.rstrip('0').rstrip('.') if '.' in rendered else rendered


def vector(xs, digits=2, scale=1):
    if not isinstance(xs, (list, tuple)):
        return 'unknown'
    return '[' + ', '.join(number(x, digits, scale) for x in xs) + ']'


def value(robot, key):
    r = robot.get(key, {})
    return (r.get('value') or {}) if r.get('status') == 'available' else {}


def calibration(robot_state):
    lines = ['Camera calibration for current RGB:']
    for c in robot_state['cameras']:
        v = c.get('value') if c['status'] == 'available' else None
        if not v:
            lines.append(c['name'] + '=unknown'); continue
        k = v['intrinsics_px']
        lines.append(c['name'] + ': resolution=' + vector(v['resolution_wh'], 0)
                     + '; fx_fy_cx_cy_px=' + vector([k[0][0], k[1][1], k[0][2], k[1][2]], 4)
                     + '; clip_m=' + vector(v['clipping_range_m'], 4) + '; pinhole')
    return '\n'.join(lines)


def center(finger, robot):
    # The full internal centroid is computed before per-point output truncation.
    # Its source is finger-local, so T_env_from_body handles the env translation.
    p = finger.get('patch_migration', {}).get('centroid_finger_m')
    if (p is None or finger.get('contact_above_threshold') is not True
            or finger.get('detail_status') != 'available'
            or finger.get('buffer_saturated') is not False):
        return None
    links = value(robot, 'frames').get('links', [])
    matches = [x for x in links if x['body_name'] == finger['body_name']]
    if len(matches) != 1:
        return None
    t = matches[0]['T_env_from_body']
    return [sum(t[i][j] * p[j] for j in range(3)) + t[i][3] for i in range(3)]


def age(f):
    active = f.get('contact_above_threshold')
    if active is False:
        return '0'
    s = f.get('statistics', {})
    interval = s.get('latest_contact_interval')
    if (active is not True or s.get('status') != 'available' or not interval
            or not interval.get('ongoing')):
        return 'unknown'
    return ('>=' if interval['start_censored'] else '') + number(interval['duration_s'], 1, 1000)


def render_robot(robot_state):
    if robot_state.get('schema') != 'justrua-robot-v1':
        raise ValueError('Unexpected robot sensor schema')
    lines = ['Robot and camera geometry:']
    robots = {x['arm_name']: x for x in robot_state['kinematics']['robots']}
    fingers = {(x['arm_name'], x['joint_name']): x for x in robot_state['contact']['fingers']}
    duration = robot_state['contact']['window']['duration_s']
    lines.append('contact_window_s=' + number(duration, 3))
    for side in ('left', 'right'):
        robot = robots[side + '_arm']
        twist = value(robot, 'tcp_twist')
        pair = [fingers[(side + '_arm', joint)] for joint in ('joint7', 'joint8')]
        friction = [f.get('friction_magnitude_n') if f.get('friction_status') == 'available' else None for f in pair]
        peaks = [f['statistics'].get('peak_normal_n') if f['statistics'].get('status') == 'available' else None for f in pair]
        lines.extend([
            side + '_inner_gap_mm=' + number(value(robot, 'tool_geometry').get('inner_working_surface_gap_m'), 2, 1000),
            side + '_tcp_velocity_mm_s=' + vector(twist.get('linear_m_s'), 2, 1000),
            side + '_tcp_angular_velocity_deg_s=' + vector(twist.get('angular_rad_s'), 2, 180 / math.pi),
            side + '_friction_N=' + vector(friction),
            side + '_contact_age_ms=[' + ', '.join(age(f) for f in pair) + ']',
            side + '_recent_peak_normal_N=' + vector(peaks),
            side + '_contact_centers_env_m=[' + ', '.join(vector(center(f, robot), 4) for f in pair) + ']',
        ])
    for c in robot_state['cameras']:
        v = c.get('value') if c['status'] == 'available' else None
        if not v:
            lines.append(c['name'] + '_pose=unknown'); continue
        t = v['T_env_from_camera_cv']
        lines.append(c['name'] + '_position_env_m=' + vector([t[i][3] for i in range(3)], 4))
        lines.append(c['name'] + '_right_down_forward_env=[' + ', '.join(
            vector([t[i][j] for i in range(3)], 5) for j in range(3)) + ']')
    return '\n'.join(lines)
