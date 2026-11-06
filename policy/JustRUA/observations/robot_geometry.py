"""Robot-only X5 tool geometry. No simulator imports until explicit sampling.

Working surfaces are selected from the two actual USD finger meshes, expressed
in their rigid-link frames. This measures a visual/geometric gap, not the gap of
PhysX convex/contact-offset colliders and never an object/grasp-success signal.
"""
from __future__ import annotations

import numpy as np

TCP_OFFSET_M = 0.1501
_MESH_CACHE = {}
_JOINT_CACHE = {}


def array(value, shape=None):
    for method in ("detach", "cpu", "numpy"):
        if callable(getattr(value, method, None)):
            value = getattr(value, method)()
    out = np.asarray(value, dtype=float)
    if (shape is not None and out.shape != shape) or not np.isfinite(out).all():
        raise ValueError("invalid numeric array")
    return out


def unknown(reason):
    return {"status": "unavailable", "value": None, "reason_code": reason,
            "source": "robot_self_usd_mesh", "frame": "env", "units": "m"}


def select_working_pad(triangles, inward_sign):
    """Select the terminal, inward planar surface of an X5 finger in link axes.

    X5 fingers extend along link +x and open along +/-y. Require an actual
    terminal plane, a nonzero x/z patch, and planarity. No URDF q-sum fallback.
    """
    triangles = array(triangles)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError("invalid finger triangles")
    if inward_sign not in (-1, 1):
        raise ValueError("invalid inward direction")
    tip = triangles[:, :, 0].max()
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    parallel = np.abs(normals[:, 1]) >= .99999 * np.maximum(norm, 1e-30)
    terminal = np.abs(triangles[:, :, 0].max(axis=1)-tip) < 1e-6
    candidates = triangles[parallel & terminal & (norm > 1e-12)]
    if len(candidates) < 2:
        raise ValueError("no terminal planar pad")
    # Of terminal y-planes, select the side facing the other finger.
    planes = candidates[:, :, 1].mean(axis=1)
    target = planes.min() if inward_sign < 0 else planes.max()
    pad = candidates[np.abs(planes-target) < 1e-6].reshape(-1, 3)
    lo, hi = pad.min(axis=0), pad.max(axis=0)
    if hi[1]-lo[1] > 1e-6 or min(hi[0]-lo[0], hi[2]-lo[2]) < .001:
        raise ValueError("invalid planar pad extent")
    corners = np.array([[x, target, z] for x in (lo[0], hi[0]) for z in (lo[2], hi[2])])
    return {"corners": corners, "center": (lo+hi)/2,
            "normal": np.array([0., float(inward_sign), 0.])}


def geometry_from_meshes(meshes, transforms):
    """Pure geometry path used by runtime and CPU tests; frames are column T."""
    pads = []
    for name, inward in (("link7", -1), ("link8", 1)):
        pad = select_working_pad(meshes[name], inward)
        transform = array(transforms[name], (4, 4))
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("invalid finger rotation")
        corners = pad["corners"] @ rotation.T + transform[:3, 3]
        pads.append({"body_name": name, "center_env_m": (rotation @ pad["center"]+transform[:3, 3]).tolist(),
                     "inward_normal_env": (rotation @ pad["normal"]).tolist(),
                     "corners_env_m": corners.tolist()})
    a, b = pads
    n = array(a["inward_normal_env"], (3,))
    if np.dot(n, b["inward_normal_env"]) > -.99999:
        raise ValueError("finger working planes not opposed")
    delta = array(b["center_env_m"])-array(a["center_env_m"])
    gap = float(delta @ n)
    # Check the projected rectangular pads overlap in both tangential axes.
    corners_a, corners_b = array(a["corners_env_m"]), array(b["corners_env_m"])
    tangent = array(transforms["link7"], (4, 4))[:3, 0]
    other = np.cross(n, tangent)
    overlap = [float(min((corners_a @ axis).max(), (corners_b @ axis).max()) -
                     max((corners_a @ axis).min(), (corners_b @ axis).min())) for axis in (tangent, other)]
    if min(overlap) <= 0:
        raise ValueError("working surfaces do not overlap")
    return {"working_surfaces": pads, "inner_working_surface_gap_m": gap,
            "working_surface_overlap_m": overlap,
            "gap_definition": "signed separation of opposed terminal inner USD mesh planes over their overlap",
            "gap_source": "actual_robot_usd_mesh_and_measured_body_link_poses",
            "collision_gap_m": None, "collision_geometry_status": "not_verified",
            "tool_axis_link6": [1., 0., 0.], "jaw_axis_link6": [0., 1., 0.],
            "tcp_offset_link6_m": [TCP_OFFSET_M, 0., 0.],
            "object_clearance_or_grasp_success": "not_inferred"}


def _root_path(art, env_idx):
    root = str(art.cfg.prim_path).replace("env_.*", "env_"+str(env_idx))
    if not root.startswith("/") or any(c in root for c in "*[]{}()+?|\\"):
        raise ValueError("robot root not exact")
    return root


def _usd_meshes(art, env_idx):
    # Traverse the exact robot subtree only, never scene objects or layout data.
    from pxr import Usd, UsdGeom, UsdPhysics
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    root_path = _root_path(art, env_idx)
    key = (id(stage), root_path)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    if not np.isclose(float(UsdGeom.GetStageMetersPerUnit(stage)), 1., atol=1e-9):
        raise ValueError("stage length unit unsupported")
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError("robot root missing")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    result = {}
    for body in ("link7", "link8"):
        matches = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.GetName() == body and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(matches) != 1:
            raise ValueError("finger rigid body ambiguous")
        rigid = matches[0]
        inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(rigid), dtype=float).T)
        triangles = []
        for prim in Usd.PrimRange(rigid, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
                continue
            # A nested rigid body is a different physical link; do not include it.
            ancestor = rigid if prim == rigid else prim.GetParent()
            while ancestor.IsValid() and ancestor != rigid and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                ancestor = ancestor.GetParent()
            if ancestor != rigid:
                continue
            mesh = UsdGeom.Mesh(prim)
            points = array(mesh.GetPointsAttr().Get())
            counts = list(mesh.GetFaceVertexCountsAttr().Get())
            indices = list(mesh.GetFaceVertexIndicesAttr().Get())
            transform = inverse @ np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
            points = points @ transform[:3, :3].T + transform[:3, 3]
            cursor = 0
            for count in counts:
                polygon = indices[cursor:cursor+count]
                cursor += count
                if count < 3:
                    continue
                for i in range(1, count-1):
                    triangles.append(points[[polygon[0], polygon[i], polygon[i+1]]])
            if cursor != len(indices):
                raise ValueError("invalid USD mesh faces")
        result[body] = array(triangles)
        select_working_pad(result[body], -1 if body == "link7" else 1)
    _MESH_CACHE[key] = result
    return result


def sample(robot, art, env_idx=0, body_transforms=None):
    """Read robot-self USD geometry only; no scene lookup, step, reset or write."""
    try:
        if getattr(robot, "ee_link_name", None) != "link6":
            return unknown("unsupported_tool")
        meshes = _usd_meshes(art, env_idx)
        value = geometry_from_meshes(meshes, body_transforms)
        return {"status": "available", "value": value,
                "source": "robot_self_usd_mesh", "frame": "env", "units": "m"}
    except Exception:
        return unknown("working_surfaces_not_calibrated")


def incoming_joint_frame(art, env_idx=0, child_name="link6"):
    """Return T_child_link_from_child_joint from that robot's USD joint only.

    Do not assume a zero URDF origin or child-joint axes equal link axes. The
    incoming wrench is at this child joint anchor, not automatically at TCP.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    root_path = _root_path(art, env_idx)
    key = (id(stage), root_path, child_name)
    if key in _JOINT_CACHE:
        return _JOINT_CACHE[key].copy()
    if not np.isclose(float(UsdGeom.GetStageMetersPerUnit(stage)), 1., atol=1e-9):
        raise ValueError("stage units unsupported")
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError("robot root missing")
    bodies = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.GetName() == child_name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise ValueError("child rigid body ambiguous")
    child_path = bodies[0].GetPath()
    matches = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            if list(joint.GetBody1Rel().GetTargets()) == [child_path]:
                matches.append(joint)
    if len(matches) != 1:
        raise ValueError("incoming child joint ambiguous")
    joint = matches[0]
    q = joint.GetLocalRot1Attr().Get()
    quaternion = array([q.GetReal(), *q.GetImaginary()], (4,))
    if abs(np.linalg.norm(quaternion)-1) > 1e-5:
        raise ValueError("child joint rotation invalid")
    w, x, y, z = quaternion/np.linalg.norm(quaternion)
    transform = np.eye(4)
    transform[:3, :3] = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    transform[:3, 3] = array(joint.GetLocalPos1Attr().Get(), (3,))
    _JOINT_CACHE[key] = transform
    return transform.copy()
