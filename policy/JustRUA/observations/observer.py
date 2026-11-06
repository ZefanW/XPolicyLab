"""Augment the existing RGB/EEF observation with state and predicted depths."""
import hashlib
import os

import numpy as np

from .depth import CAMERAS, GUIDE as DEPTH_GUIDE, DepthPro, predicted_image
from .readings import GRIPPER_GUIDE, GEOMETRY_GUIDE


SENSOR_GUIDE = '\n\n'.join((GRIPPER_GUIDE, GEOMETRY_GUIDE, DEPTH_GUIDE))


class Observer:
    """One instance per episode. ``launch`` attaches synchronized native sensors."""

    guide = SENSOR_GUIDE

    def __init__(self, task_env, *, checkpoint_dir=None, device=None, backend=None):
        self.backend = backend if backend is not None else DepthPro(
            checkpoint_dir or os.environ.get('JUSTRUA_DEPTHPRO_CHECKPOINT'),
            device or os.environ.get('JUSTRUA_DEPTHPRO_DEVICE', 'cuda:0'))
        self._latest_step = -1
        self._last_key = None
        self._parts = []
        self._closed = False

    def enrich(self, observation):
        """Return extra state text and labelled image parts, after the original three RGBs."""
        if self._closed:
            raise RuntimeError('The episode observer is closed')
        record = observation.state.get('justrua_sensors')
        if not isinstance(record, dict):
            raise RuntimeError('Missing native robot readings. Start RoboDojo with observations.launch before reset.')
        step = record['physics_step']
        if type(step) is not int or step < 0 or step < self._latest_step:
            raise ValueError('Invalid or stale native sensor observation')
        if set(observation.images) != set(CAMERAS):
            raise ValueError('Expected the original head, left-wrist and right-wrist RGB cameras')
        hashes = []
        for camera in CAMERAS:
            rgb = observation.images[camera]
            if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                    or rgb.shape[2] not in (3, 4) or min(rgb.shape[:2]) < 1):
                raise ValueError(f'Invalid original RGB observation for {camera}')
            size = record['calibrated_sizes'].get(camera)
            if size is not None and list(rgb.shape[:2][::-1]) != size:
                raise ValueError(f'{camera} calibration resolution differs from the RGB frame')
            hashes.append((rgb.shape[:2], hashlib.sha256(np.ascontiguousarray(rgb[:, :, :3])).digest()))
        key = (step, *hashes)
        if key != self._last_key:
            parts = []
            for camera in CAMERAS:
                uri, reason = predicted_image(self.backend, observation.images[camera])
                label = f"Predicted depth '{camera}'"
                if reason:
                    parts.append({'type': 'text', 'text': f'{label}: unknown ({reason})'})
                else:
                    parts.append({'type': 'text', 'text': label + ':'})
                    parts.append({'type': 'image_url', 'image_url': {'url': uri}})
            self._parts = parts
            self._last_key = key
        self._latest_step = step
        return record['text'], list(self._parts)

    def close(self):
        if not self._closed and hasattr(self.backend, 'close'):
            self.backend.close()
        self._parts = []
        self._closed = True
