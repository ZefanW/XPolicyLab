"""Optional, identity-free finger contact telemetry sampled after physics steps.

No simulator imports or changes occur on import.  Contact points and friction
anchors are different PhysX buffers, not corresponding measurements.  Point
migration is a robot-local contact-patch estimate, never a holding/slip label.
"""
from __future__ import annotations

from collections import deque
import copy
import functools
import importlib
import math

from . import finger_contact

SCHEMA = "justrua-contact-v1"
HISTORY_STEPS = 512
CONTACT_CAPACITY = 256
OUTPUT_POINTS = 32
CONTACT_THRESHOLD_N = 1.0
IDENTITIES = tuple((arm, joint, "link" + joint[-1])
                   for arm in ("left_arm", "right_arm")
                   for joint in ("joint7", "joint8"))


def install_before_create():
    """Reuse the opt-in report hook; never change drives, sleep or collisions."""
    return finger_contact.install_before_create()


def _plain(value):
    for method in ("clone", "detach", "cpu", "numpy", "tolist"):
        fn = getattr(value, method, None)
        if callable(fn):
            value = fn()
    return value


def _number(value):
    if isinstance(value, bool):
        raise ValueError("Expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Expected finite number")
    return result


def _vector(value):
    if len(value) != 3:
        raise ValueError("Expected three-vector")
    return [_number(v) for v in value]


def _scalar(value):
    return _number(value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value)


def _indices(counts, starts, capacity):
    """Consume opaque pair offsets; never keep/return the other-body identity."""
    counts, starts = _plain(counts), _plain(starts)
    if (len(counts) != 1 or len(starts) != 1 or not counts[0]
            or len(counts[0]) != len(starts[0])):
        raise ValueError("Expected exactly one sensor")
    indices = set()
    for count, start in zip(counts[0], starts[0]):
        count = _number(count)
        if count != int(count) or count < 0:
            raise ValueError("Invalid contact offsets")
        if count == 0:
            # Empty-pair offsets are unspecified; do not dereference them.
            continue
        start = _number(start)
        if start != int(start) or start < 0:
            raise ValueError("Invalid contact offsets")
        count, start = int(count), int(start)
        if start + count > capacity:
            raise OverflowError("Contact buffer overflow")
        indices.update(range(start, start + count))
    return sorted(indices), len(indices) >= capacity


def _finger_transform(art, env_idx, body):
    """Raw PhysX link transform is xyz + xyzw; only robot state is accessed."""
    index = list(art.body_names).index(body)
    transform = _plain(art.root_physx_view.get_link_transforms())[env_idx][index]
    if len(transform) != 7:
        raise ValueError("Expected PhysX transform")
    pos = _vector(transform[:3])
    quat = [_number(x) for x in transform[3:]]
    norm = math.sqrt(sum(x * x for x in quat))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Invalid quaternion")
    return pos, [x / norm for x in quat]


def _local(point, transform):
    if transform is None:
        return None
    pos, (x, y, z, w) = transform
    v = [point[i] - pos[i] for i in range(3)]
    # Conjugate rotation, q^-1 v q; no scene/object transforms are accessed.
    x, y, z = -x, -y, -z
    cross = [y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]]
    cross2 = [y * cross[2] - z * cross[1], z * cross[0] - x * cross[2],
              x * cross[1] - y * cross[0]]
    return _vector([v[i] + 2 * w * cross[i] + 2 * cross2[i] for i in range(3)])


def _empty_finger(identity, reason):
    arm, joint, body = identity
    return {"arm_name": arm, "joint_name": joint, "body_name": body,
            "status": "unavailable", "reason_code": reason,
            "normal_resultant_w_n": None, "normal_magnitude_n": None,
            "friction_resultant_w_n": None, "friction_magnitude_n": None,
            "contact_above_threshold": None, "contacts": [], "friction_anchors": [],
            "detail_status": "unavailable", "friction_status": "unavailable",
            "buffer_saturated": None, "detail_output_truncated": False,
            "friction_output_truncated": False}


def _read_finger(view, dt, identity, transform=None):
    row = _empty_finger(identity, "finger_contact_unavailable")
    centroid = None
    saturated = False
    force_mismatch = False
    try:
        normal = finger_contact._force_vector(view, dt)
        magnitude = _number(math.hypot(*normal))
        row.update(normal_resultant_w_n=normal, normal_magnitude_n=magnitude,
                   contact_above_threshold=magnitude >= CONTACT_THRESHOLD_N)
    except Exception:
        pass
    try:
        forces, points, normals, distances, counts, starts = view.get_contact_data(dt=dt)
        forces, points, normals, distances = map(_plain, (forces, points, normals, distances))
        capacity = min(len(forces), len(points), len(normals), len(distances), CONTACT_CAPACITY)
        indices, full = _indices(counts, starts, capacity)
        if not indices and row["normal_magnitude_n"] is not None and row["normal_magnitude_n"] > 1e-8:
            force_mismatch = True
            raise ValueError("Net force contradicts empty contact details")
        saturated |= full
        contacts = []
        for index in indices:
            point = _vector(points[index])
            contacts.append({"point_w_m": point, "normal_w": _vector(normals[index]),
                             "normal_force_n": _scalar(forces[index]),
                             "separation_m": _scalar(distances[index]),
                             "point_finger_m": _local(point, transform)})
        row.update(contacts=contacts[:OUTPUT_POINTS], detail_status="partial" if full else "available",
                   detail_output_truncated=len(contacts) > OUTPUT_POINTS)
        if not full and row["normal_resultant_w_n"] is not None:
            aggregate = _vector([sum(c["normal_force_n"] * c["normal_w"][i] for c in contacts) for i in range(3)])
            discrepancy = math.hypot(*[aggregate[i] - row["normal_resultant_w_n"][i] for i in range(3)])
            force_mismatch = discrepancy > 1e-4 + 1e-3 * row["normal_magnitude_n"]
            if force_mismatch:
                row["detail_status"] = "partial"
        if contacts and transform is not None and not full and not force_mismatch:
            weights = [abs(c["normal_force_n"]) for c in contacts]
            if sum(weights) == 0:
                weights = [1.0] * len(contacts)
            centroid = _vector([sum(c["point_finger_m"][i] * w for c, w in zip(contacts, weights)) / sum(weights)
                                for i in range(3)])
    except OverflowError:
        saturated = True
        row.update(detail_status="unavailable", contacts=[], detail_output_truncated=False)
        centroid = None
    except Exception:
        row.update(detail_status="unavailable", contacts=[], detail_output_truncated=False)
        centroid = None
    try:
        forces, points, counts, starts = view.get_friction_data(dt=dt)
        forces, points = _plain(forces), _plain(points)
        indices, full = _indices(counts, starts, min(len(forces), len(points), CONTACT_CAPACITY))
        saturated |= full
        anchors = [{"point_w_m": _vector(points[i]), "force_w_n": _vector(forces[i])} for i in indices]
        row.update(friction_anchors=anchors[:OUTPUT_POINTS],
                   friction_output_truncated=len(anchors) > OUTPUT_POINTS,
                   friction_status="partial" if full else "available")
        # A truncated physical buffer cannot supply a complete resultant.
        if not full:
            vector = _vector([sum(a["force_w_n"][i] for a in anchors) for i in range(3)])
            magnitude = _number(math.hypot(*vector))
            row.update(friction_resultant_w_n=vector, friction_magnitude_n=magnitude)
    except OverflowError:
        saturated = True
        row.update(friction_status="unavailable", friction_anchors=[], friction_output_truncated=False,
                   friction_resultant_w_n=None, friction_magnitude_n=None)
    except Exception:
        row.update(friction_status="unavailable", friction_anchors=[], friction_output_truncated=False,
                   friction_resultant_w_n=None, friction_magnitude_n=None)
    if force_mismatch and row["friction_status"] != "unavailable":
        # Anonymous normal/detail disagreement means the coverage check failed.
        # Individual anchor values remain readings; their sum is not complete.
        row.update(friction_status="partial", friction_resultant_w_n=None, friction_magnitude_n=None)
    row["buffer_saturated"] = saturated if row["detail_status"] != "unavailable" or row["friction_status"] != "unavailable" or saturated else None
    complete = row["normal_magnitude_n"] is not None and row["detail_status"] == row["friction_status"] == "available"
    any_data = row["normal_magnitude_n"] is not None or row["detail_status"] != "unavailable" or row["friction_status"] != "unavailable"
    row.update(status="available" if complete else "partial" if any_data else "unavailable",
               reason_code=None if complete else "contact_buffer_saturated" if saturated else
               "contact_force_mismatch" if force_mismatch else "contact_channel_unavailable")
    return row, centroid


def _registration_paths(env):
    """Resolve collision actors for registration; never read object state.

    Tensor wildcards do not recursively enumerate USD descendants. Traverse the
    composed stage, including instance proxies, and map each enabled collision
    shape to its nearest enabled rigid actor. A static shape has no enabled
    rigid actor and is itself the exact contact-filter path. The caller discards
    this local list immediately after binding; identities are not telemetry.
    """
    from pxr import Usd, UsdPhysics
    stage = getattr(env, "stage", None)
    if stage is None:
        stage = importlib.import_module("omni.usd").get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage unavailable")
    paths = set()
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        target = prim
        current = prim
        while current and str(current.GetPath()) != "/":
            if current.HasAPI(UsdPhysics.RigidBodyAPI):
                if UsdPhysics.RigidBodyAPI(current).GetRigidBodyEnabledAttr().Get() is not False:
                    target = current
                break
            current = current.GetParent()
        paths.add(str(target.GetPath()))
    if not paths:
        raise RuntimeError("Collision registration unavailable")
    return sorted(paths)


def _contact_view(backend, path, registration_paths):
    """Bind one exact sensor to exact collision actor/shape filters."""
    paths = sorted(set(registration_paths) - {path})
    if not paths or any(not p.startswith("/") or any(c in p for c in "*[]{}()+?|\\") for p in paths):
        raise ValueError("Exact collision registration paths unavailable")
    view = backend.create_rigid_contact_view(path, filter_patterns=paths,
                                             max_contact_data_count=CONTACT_CAPACITY)
    if int(view.sensor_count) != 1 or int(view.filter_count) != len(paths):
        raise ValueError("Contact registration coverage mismatch")
    if not view.check():
        raise ValueError("Invalid contact view")
    return view


def _integral(samples, key):
    valid = [s for s in samples if s[key] is not None]
    return [sum(s[key][i] * s["dt"] for s in valid) for i in range(3)] if valid else None


def _statistics(samples):
    normal = [s for s in samples if s["normal"] is not None]
    friction = [s for s in samples if s["friction"] is not None]
    duration = sum(s["dt"] for s in normal)
    friction_duration = sum(s["dt"] for s in friction)
    result = {"status": "available" if samples and len(normal) == len(friction) == len(samples) else "partial" if normal or friction else "unavailable",
              "observed_duration_s": duration, "missing_sample_count": len(samples) - len(normal),
              "peak_normal_n": max((s["normal_mag"] for s in normal), default=None),
              "mean_normal_n": sum(s["normal_mag"] * s["dt"] for s in normal) / duration if duration else None,
              "normal_impulse_w_ns": _integral(samples, "normal"),
              "friction_observed_duration_s": friction_duration,
              "friction_missing_sample_count": len(samples) - len(friction),
              "peak_friction_n": max((s["friction_mag"] for s in friction), default=None),
              "mean_friction_n": sum(s["friction_mag"] * s["dt"] for s in friction) / friction_duration if friction_duration else None,
              "friction_impulse_w_ns": _integral(samples, "friction"),
              "contact_active_duration_s": sum(s["dt"] for s in samples if s["active"] is True) if normal else None,
              "latest_contact_interval": None}
    active_indices = [i for i, s in enumerate(samples) if s["active"] is True]
    if active_indices:
        end = active_indices[-1]
        start = end
        while start > 0 and samples[start - 1]["active"] is True:
            start -= 1
        segment = samples[start:end + 1]
        elapsed = sum(s["dt"] for s in segment)
        result["latest_contact_interval"] = {
            "start_s": segment[0]["t"] - segment[0]["dt"], "end_s": segment[-1]["t"],
            "duration_s": elapsed, "ongoing": end == len(samples) - 1,
            "start_censored": (start == 0) or samples[start - 1]["active"] is None,
            "end_censored": end < len(samples) - 1 and samples[end + 1]["active"] is None,
            "peak_normal_n": max(s["normal_mag"] for s in segment),
            "mean_normal_n": sum(s["normal_mag"] * s["dt"] for s in segment) / elapsed,
            "normal_impulse_w_ns": _integral(segment, "normal")}
    return result


def _migration(samples):
    latest = samples[-1] if samples else None
    result = {"status": "unavailable", "reason_code": "insufficient_contact_history", "inference_only": True,
              "centroid_finger_m": latest["centroid"] if latest else None,
              "delta_finger_m": None, "distance_m": None, "speed_m_s": None}
    if len(samples) < 2:
        return result
    previous, latest = samples[-2:]
    if not all(s["active"] is True and s["centroid"] is not None for s in (previous, latest)):
        result["reason_code"] = "contact_patch_unavailable"
        return result
    delta = [latest["centroid"][i] - previous["centroid"][i] for i in range(3)]
    distance = math.hypot(*delta)
    result.update(status="available", reason_code=None, delta_finger_m=delta,
                  distance_m=distance, speed_m_s=distance / latest["dt"])
    return result


def _asymmetry(first, second):
    valid = first is not None and second is not None
    return {"status": "available" if valid else "unavailable",
            "difference_n": first - second if valid else None,
            "normalized_difference": (first - second) / (first + second) if valid and first + second else 0.0 if valid else None}


def _native(env):
    sim = getattr(env, "sim", None)
    return getattr(sim, "unwrapped", sim)


def _token(env):
    native = _native(env)
    for candidate in (getattr(native, "sim", None), native):
        value = getattr(candidate, "current_time_step_index", None)
        if value is not None:
            return int(value)
    return None


class _Collector:
    def __init__(self, env):
        self.env = env
        self.views = {}
        self.backend = None
        self.subscription = None
        self.reason = "post_step_subscription_unavailable"
        self.last_token = None
        self.resetting = False
        self.reset_depth = 0
        self.env_count = int(getattr(env, "num_envs", getattr(_native(env), "num_envs", 1)))
        self.records = [self._record(0) for _ in range(self.env_count)]

    @staticmethod
    def _record(epoch):
        return {"epoch": epoch, "t": 0.0, "dt": None, "dropped": 0,
                "history": deque(maxlen=HISTORY_STEPS), "latest": None}

    def reset(self, env_ids=None):
        ids = range(self.env_count) if env_ids is None else _plain(env_ids)
        if isinstance(ids, int):
            ids = [ids]
        reset_indices = set()
        for index in ids:
            index = int(index)
            if 0 <= index < self.env_count:
                self.records[index] = self._record(self.records[index]["epoch"] + 1)
                reset_indices.add(index)
        self.last_token = None
        self.views.clear()
        if len(reset_indices) == self.env_count:
            # A rebuilt physics backend belongs to this same explicit reset,
            # not a second episode boundary on the next callback.
            self.backend = None

    def _read(self, env_idx, dt, backend):
        rows, centroids = [], []
        rm = self.env.robot_manager
        registration_paths = None
        registration_failed = False
        for identity in IDENTITIES:
            arm, joint, body = identity
            path = None
            try:
                indices = [i for i, robot in enumerate(rm.robot_list) if getattr(robot, "arm_name", None) == arm]
                if len(indices) != 1:
                    raise ValueError("Expected one arm")
                art = rm.robot_key[indices[0]]
                path = finger_contact._body_path(finger_contact._root_path(art, env_idx), joint)
                if path not in self.views:
                    # Cache both success and failure until reset/backend change.
                    # No path list or contacted-object metadata is retained.
                    self.views[path] = None
                    if registration_paths is None and not registration_failed:
                        try:
                            registration_paths = _registration_paths(self.env)
                        except Exception:
                            registration_failed = True
                    if registration_paths is not None:
                        self.views[path] = _contact_view(backend, path, registration_paths)
                if self.views[path] is None:
                    raise RuntimeError("Contact registration unavailable")
                try:
                    transform = _finger_transform(art, env_idx, body)
                except Exception:
                    transform = None
                row, centroid = _read_finger(self.views[path], dt, identity, transform)
            except Exception:
                reason = "contact_registration_unavailable" if path in self.views and self.views[path] is None else "finger_contact_unavailable"
                row, centroid = _empty_finger(identity, reason), None
            rows.append(row)
            centroids.append(centroid)
        return rows, centroids

    def on_step(self, dt):
        """Post-physics callback. Never step/reset/write controls from here."""
        if self.resetting:
            return
        try:
            dt = _number(dt)
            if dt <= 0:
                raise ValueError("Invalid dt")
            token = _token(self.env)
            try:
                backend = finger_contact._physics_view(self.env)
            except Exception:
                backend = None
            backend_changed = backend is not None and self.backend is not None and self.backend is not backend
            clock_regressed = False
            if token is not None and self.last_token is not None:
                if token == self.last_token and not backend_changed:
                    return
                clock_regressed = token < self.last_token
            if clock_regressed or backend_changed:
                # Clock restart and replacement backend can describe the same
                # physical boundary, and must increment the epoch only once.
                self.reset()
            self.last_token = token
            if backend is None:
                raise RuntimeError("Physics view unavailable")
            self.backend = backend
            # Reading a disabled reporter must not silently produce zero contacts.
            carb = importlib.import_module("carb")
            if carb.settings.get_settings().get_as_bool("/physics/disableContactProcessing"):
                raise RuntimeError("Contact processing disabled")
            backend_available = True
        except Exception:
            backend_available = False
            try:
                dt = finger_contact._physics_dt(self.env)
            except Exception:
                self.reason = "physics_dt_unavailable"
                return
        for env_idx, record in enumerate(self.records):
            try:
                if not backend_available:
                    raise RuntimeError("Backend unavailable")
                rows, centroids = self._read(env_idx, dt, backend)
            except Exception:
                rows, centroids = ([_empty_finger(identity, "contact_backend_unavailable")
                                    for identity in IDENTITIES], [None] * 4)
            record["t"] += dt
            record["dt"] = dt
            condensed = []
            for row, centroid in zip(rows, centroids):
                condensed.append({"t": record["t"], "dt": dt,
                                  "normal": row["normal_resultant_w_n"], "normal_mag": row["normal_magnitude_n"],
                                  "friction": row["friction_resultant_w_n"], "friction_mag": row["friction_magnitude_n"],
                                  "active": row["contact_above_threshold"], "centroid": centroid})
            if len(record["history"]) == HISTORY_STEPS:
                record["dropped"] += 1
            record["history"].append(condensed)
            record["latest"] = rows
        self.reason = None if backend_available else "contact_backend_unavailable"

    def sample(self, env_idx):
        record = self.records[env_idx]
        history = list(record["history"])
        rows = copy.deepcopy(record["latest"] or [_empty_finger(i, "no_physics_samples") for i in IDENTITIES])
        for index, row in enumerate(rows):
            samples = [step[index] for step in history]
            row["statistics"] = _statistics(samples)
            row["patch_migration"] = _migration(samples)
        statuses = [row["status"] for row in rows]
        status = "available" if all(s == "available" for s in statuses) else "partial" if any(s != "unavailable" for s in statuses) else "unavailable"
        asymmetry = []
        for index, arm in ((0, "left_arm"), (2, "right_arm")):
            asymmetry.append(dict(arm_name=arm, **_asymmetry(rows[index]["normal_magnitude_n"], rows[index + 1]["normal_magnitude_n"])))
        totals = [sum(row["normal_magnitude_n"] for row in rows[i:i + 2])
                  if all(row["normal_magnitude_n"] is not None for row in rows[i:i + 2]) else None for i in (0, 2)]
        first = history[0][0] if history else None
        result = {"schema": SCHEMA, "status": status, "reason_code": self.reason if history else "no_physics_samples",
                  "source": "physx_contact_tensors", "frame": "world", "physics_dt_s": record["dt"], "epoch": record["epoch"],
                  "contact_threshold_n": CONTACT_THRESHOLD_N, "contact_threshold_rule": "normal_resultant_magnitude >= 1.0",
                  "window": {"mode": "rolling_physics_steps", "capacity_steps": HISTORY_STEPS, "sample_count": len(history),
                             "start_s": first["t"] - first["dt"] if first else None, "end_s": record["t"] if history else None,
                             "duration_s": sum(step[0]["dt"] for step in history), "dropped_steps": record["dropped"],
                             "coverage_complete": bool(history) and record["dropped"] == 0 and all(r["statistics"]["status"] == "available" for r in rows)},
                  "fingers": rows, "arm_asymmetry": asymmetry, "cross_arm_asymmetry": _asymmetry(*totals)}
        return result


def attach(env):
    """Idempotently register one post-physics subscription and reset observer.

    The reset observer calls the original reset exactly once and only clears this
    module's telemetry. A retained subscription is required for callback life.
    """
    existing = getattr(env, "_justrua_contact_collector", None)
    if existing is not None:
        return {"status": "available" if existing.subscription is not None else "unavailable",
                "reason_code": existing.reason}
    collector = _Collector(env)
    setattr(env, "_justrua_contact_collector", collector)
    native = _native(env)
    original_index_reset = getattr(native, "_reset_idx", None)
    original_env_reset = getattr(env, "reset", None)
    if not callable(original_index_reset) and not callable(original_env_reset):
        collector.reason = "reset_observer_unavailable"
        return {"status": "unavailable", "reason_code": collector.reason}
    try:
        physx = importlib.import_module("omni.physx")
        collector.subscription = physx.get_physx_interface().subscribe_physics_on_step_events(
            collector.on_step, pre_step=False, order=100)
        if collector.subscription is None:
            raise RuntimeError("Subscription missing")
    except Exception:
        return {"status": "unavailable", "reason_code": collector.reason}

    def observe_reset(original, whole_environment):
        @functools.wraps(original)
        def observed_reset(*args, **kwargs):
            collector.reset_depth += 1
            collector.resetting = True
            try:
                return original(*args, **kwargs)
            finally:
                collector.reset_depth -= 1
                if collector.reset_depth == 0:
                    ids = None if whole_environment else kwargs.get("env_ids", args[0] if args else None)
                    collector.reset(ids)
                    collector.resetting = False
        return observed_reset

    if callable(original_env_reset):
        # EvalEnv creates its simulator lazily, and performs 200 warm-up steps
        # after the inner reset. Observe the entire existing reset call, including
        # any previously installed kinematics observer, without changing it.
        env.reset = observe_reset(original_env_reset, True)
    if callable(original_index_reset):
        # Native-only partial resets are also boundaries. Nested resets clear
        # once when the outermost call returns, never once per wrapper.
        native._reset_idx = observe_reset(original_index_reset, False)
    collector.reason = None
    return {"status": "available", "reason_code": None}


def sample(env, env_idx=0):
    """Return a detached JSON-safe snapshot; never sample/step/reset on read."""
    collector = getattr(env, "_justrua_contact_collector", None)
    if collector is None:
        collector = _Collector(env)
        collector.reason = "collector_not_attached"
    if isinstance(env_idx, bool) or not isinstance(env_idx, int) or not 0 <= env_idx < collector.env_count:
        # Never expose another environment's observations for an invalid index.
        empty = _Collector(None)
        result = empty.sample(0)
        result.update(status="unavailable", reason_code="invalid_environment_index")
        return result
    result = collector.sample(env_idx)
    if collector.subscription is None:
        result["reason_code"] = collector.reason
    if not _all_finite(result):
        empty = _Collector(None)
        unknown = empty.sample(0)
        unknown.update(epoch=result["epoch"], reason_code="nonfinite_derived_value")
        return unknown
    return result


def _all_finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(v) for v in value)
    return True
