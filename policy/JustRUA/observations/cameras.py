"""Camera calibration from the same render products as the original RGB.

Only calibration is read: no depth, labels, object transforms, or scene queries.
Attaching an annotator never renders or advances physics. Native capture owns
the render fence. Dependencies on Isaac/Replicator are imported only on opt-in.
"""
from __future__ import annotations

import math
from contextlib import ExitStack

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def _array(value):
    import numpy as np
    for name in ('detach', 'cpu', 'numpy'):
        method = getattr(value, name, None)
        if callable(method):
            value = method()
    return np.asarray(value)


def decode_render_calibration(raw, resolution_wh, origin, actual_camera_usd=None):
    """Pure CPU conversion; pixel centers are (column+.5, row+.5)."""
    import numpy as np
    width, height = resolution_wh
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError('invalid_resolution')
    if tuple(_array(raw['renderProductResolution']).reshape(-1)) != (width, height):
        raise ValueError('render_resolution_mismatch')
    model = _array(raw['cameraModel']).reshape(-1)
    if len(model) != 1 or str(model[0]) != 'pinhole':
        raise ValueError('unsupported_lens_model')
    scale = float(raw['metersPerSceneUnit'])
    if not math.isfinite(scale) or not np.isclose(scale, 1., atol=1e-9, rtol=0):
        raise ValueError('unsupported_scene_unit')
    projection = _array(raw['cameraProjection']).astype(float).reshape(4, 4)
    view = _array(raw['cameraViewTransform']).astype(float).reshape(4, 4)
    if not np.isfinite(projection).all() or not np.isfinite(view).all():
        raise ValueError('nonfinite_camera_matrix')
    if not np.allclose(projection[:, 3], [0, 0, -1, 0], atol=1e-6):
        raise ValueError('unsupported_projection')

    def project(points):
        usd = np.c_[points[:, 0], -points[:, 1], -points[:, 2], np.ones(len(points))]
        clip = usd @ projection
        ndc = clip[:, :2] / clip[:, 3:4]
        return np.c_[(ndc[:, 0]+1)*width/2, (1-ndc[:, 1])*height/2]

    uv = project(np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]]))
    k = np.array([[uv[1, 0]-uv[0, 0], uv[2, 0]-uv[0, 0], uv[0, 0]],
                  [uv[1, 1]-uv[0, 1], uv[2, 1]-uv[0, 1], uv[0, 1]], [0., 0., 1.]])
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('invalid_intrinsics')
    if abs(k[0, 1])+abs(k[1, 0]) > 1e-6:
        raise ValueError('unsupported_skew')
    check = np.array([[.013, -.021, .4], [-.13, .1, .7], [.2, -.1, 1.3]])
    homogeneous = check @ k.T
    projection_error = float(np.max(np.abs(project(check)-homogeneous[:, :2]/homogeneous[:, 2:3])))
    if projection_error > 1e-5:
        raise ValueError('projection_roundtrip_failed')
    # USD row-vector world-to-camera -> column-vector camera-to-world.
    t_usd = np.linalg.inv(view.T)
    if not np.allclose(t_usd[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('invalid_camera_transform')
    r = t_usd[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1., atol=1e-5):
        raise ValueError('nonrigid_camera_transform')
    pose_error = None
    if actual_camera_usd is not None:
        pose_error = float(np.max(np.abs(t_usd-_array(actual_camera_usd))))
        if not math.isfinite(pose_error) or pose_error > 1e-4:
            raise ValueError('camera_rgb_state_mismatch')
    origin = _array(origin).astype(float).reshape(3)
    if not np.isfinite(origin).all():
        raise ValueError('invalid_robot_environment_origin')
    t_cv = t_usd @ np.diag([1., -1., -1., 1.])
    t_cv[:3, 3] -= origin
    near_far = _array(raw['cameraNearFar']).astype(float).reshape(2)
    if not np.isfinite(near_far).all() or not 0 < near_far[0] < near_far[1]:
        raise ValueError('invalid_camera_clipping')
    fov = [math.degrees(math.atan(k[0, 2]/k[0, 0])+math.atan((width-k[0, 2])/k[0, 0])),
           math.degrees(math.atan(k[1, 2]/k[1, 1])+math.atan((height-k[1, 2])/k[1, 1]))]
    return {'resolution_wh': [width, height], 'intrinsics_px': k.tolist(),
            'T_env_from_camera_cv': t_cv.tolist(), 'fov_xy_deg': fov,
            'clipping_range_m': near_far.tolist(),
            'distortion': {'model': 'pinhole', 'coefficients': []},
            'validation': {'projection_error_px': projection_error,
                           'camera_pose_max_abs_error': pose_error}}


def detach(env):
    current = getattr(env, '_justrua_camera', None)
    if current is None:
        return
    env._justrua_camera = None
    with ExitStack() as cleanup:
        for annotator, path in reversed(list(zip(current['annotators'], current['paths']))):
            cleanup.callback(annotator.detach, [path])


def prepare(env):
    """Attach to initialized RGB render products. Does not render or step."""
    import omni.replicator.core as rep
    names = list(env.camera_manager.camera_names[0])
    if len(names) != 3 or set(names) != set(CAMERAS) or env.num_envs != 1:
        raise ValueError('JustRUA requires the three original cameras in one environment')
    tiled = list(env.capture_manager.tiled_cameras)
    if len(tiled) < len(names):
        raise RuntimeError('camera_render_products_not_initialized')
    # Native TiledCaptureManager.reset/init_cameras appends new views, whereas
    # its step() reads tiled_cameras[cam_id] for cam_id in range(num_cameras).
    # Bind those same RGB-producing objects, not the newly appended tail. Do
    # not clear or otherwise change the native capture manager's lifecycle.
    tiled = tiled[:len(names)]
    paths = tuple(str(t._render_product_path) for t in tiled)
    # Retain the objects: an integer id can be recycled after native reset.
    products = tuple(tiled)
    current = getattr(env, '_justrua_camera', None)
    if (current is not None and current['paths'] == paths
            and len(current['products']) == len(products)
            and all(old is new for old, new in zip(current['products'], products))):
        return
    detach(env)
    annotators = []
    with ExitStack() as rollback:
        for path in paths:
            annotator = rep.AnnotatorRegistry.get_annotator('CameraParams', device='cpu')
            annotator.attach([path])
            rollback.callback(annotator.detach, [path])
            annotators.append(annotator)
        env._justrua_camera = {'paths': paths, 'products': products,
                              'annotators': annotators, 'names': names}
        rollback.pop_all()


def sample(env, observation, stamp, env_idx=0):
    """Read calibration after original RGB capture; no implicit extra capture."""
    if env_idx != 0:
        raise ValueError('Camera calibration requires env_idx=0')
    state = getattr(env, '_justrua_camera', None)
    names = list(env.camera_manager.camera_names[0])
    result = []
    for i, name in enumerate(names):
        row = {'name': name, 'status': 'unavailable', 'source': 'actual_rgb_render_product',
               'frame': 'env', 'camera_axes': '+X_right,+Y_down,+Z_forward',
               'matrix_convention': 'column_vectors; destination_from_source',
               'pixel_convention': 'u=column+0.5,v=row+0.5',
               'sample': dict(stamp), 'value': None}
        try:
            from pxr import Usd, UsdGeom
            from isaacsim.core.utils.stage import get_current_stage
            from env.camera_manager.capture.render_sync import dispatcher_time, zero_delay_active
            if state is None or state['names'] != names:
                raise ValueError('camera_annotator_not_prepared')
            if not zero_delay_active() or dispatcher_time() is None:
                raise ValueError('camera_sync_not_verified')
            if getattr(env, 'use_fabric', True):
                raise ValueError('camera_prim_verification_requires_usd_updates')
            rgb = observation['vision'][name]['color']
            shape = tuple(rgb.shape)
            if len(shape) != 3 or shape[2] not in (3, 4):
                raise ValueError('invalid_original_rgb_shape')
            camera = env.camera_manager.cameras[env_idx][i]
            if camera.get_lens_distortion_model() not in (None, '', 'pinhole'):
                raise ValueError('unsupported_lens_distortion')
            prim = get_current_stage().GetPrimAtPath(camera.prim_path)
            if not prim.IsValid() or not UsdGeom.Camera(prim):
                raise ValueError('invalid_camera_prim')
            actual = _array(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)).T
            raw = state['annotators'][i].get_data()
            value = decode_render_calibration(raw, (int(shape[1]), int(shape[0])),
                                              env.env_origins[env_idx], actual)
            row.update(status='available', value=value)
            row['sample']['render_time_s'] = float(dispatcher_time())
        except Exception as exc:
            row['reason_code'] = 'camera_calibration_unavailable'
            row['error_type'] = type(exc).__name__
        result.append(row)
    return result
