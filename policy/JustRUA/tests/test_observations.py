"""Offline checks. No simulator, Depth Pro weights or model account required."""
import base64
import io
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from XPolicyLab.policy.JustRUA.observations import Observer
from XPolicyLab.policy.JustRUA.observations.depth import CAMERAS
from XPolicyLab.policy.JustRUA.observations.readings import render_fingers


class PixelModel:
    def __init__(self):
        self.calls = []

    def infer(self, rgb):
        assert rgb.shape == (2, 3, 3) and not rgb.flags.writeable
        self.calls.append(int(rgb[0, 0, 0]))
        if self.calls[-1] == 1:
            raise ValueError('Unavailable camera prediction')
        return np.array([[0.5, 1.0, 1.5], [2.0, 2.1, np.nan]], dtype=np.float32)


class TestObservations(unittest.TestCase):
    def setUp(self):
        self.images = {name: np.full((2, 3, 4), i, dtype=np.uint8) for i, name in enumerate(CAMERAS)}
        self.record = {'physics_step': 12, 'calibrated_sizes': {c: [3, 2] for c in CAMERAS},
                       'text': 'Measured robot readings'}
        # The predictor must receive only the named RGB pixels, even when other state is present.
        self.obs = SimpleNamespace(images=self.images, state={'justrua_sensors': self.record, 'depth': object()})
        self.model = PixelModel()
        self.observer = Observer(None, backend=self.model)
        self.addCleanup(self.observer.close)

    def test_cameras_color_scale_and_missing_view(self):
        text, parts = self.observer.enrich(self.obs)
        self.assertEqual(text, self.record['text'])
        self.assertEqual(self.model.calls, [0, 1, 2])
        self.assertEqual([p['type'] for p in parts], ['text', 'image_url', 'text', 'text', 'image_url'])
        self.assertIn('cam_head', parts[0]['text'])
        self.assertIn("'cam_left_wrist': unknown", parts[2]['text'])
        self.assertIn('cam_right_wrist', parts[3]['text'])
        prefix, payload = parts[1]['image_url']['url'].split(',', 1)
        self.assertEqual(prefix, 'data:image/png;base64')
        image = Image.open(io.BytesIO(base64.b64decode(payload, validate=True)))
        self.assertEqual(image.size, (3, 2))
        self.assertEqual(np.asarray(image).reshape(-1, 3).tolist(),
                         [[0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0],
                          [255, 255, 255], [255, 0, 255]])
        self.assertEqual(self.observer.enrich(self.obs), (text, parts))
        self.assertEqual(self.model.calls, [0, 1, 2])

    def test_stale_observations_and_mismatched_calibration_are_rejected(self):
        self.observer.enrich(self.obs)
        self.record['physics_step'] -= 1
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.observer.enrich(self.obs)
        self.record['physics_step'] += 1
        self.record['calibrated_sizes']['cam_head'] = [2, 3]
        with self.assertRaisesRegex(ValueError, 'resolution'):
            self.observer.enrich(self.obs)
        self.observer.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.observer.enrich(self.obs)

    def test_sensors_distinguish_command_from_measured_opening(self):
        reading = {'arm_name': 'left_arm', 'gripper_joint_names': ['joint7', 'joint8'],
                   'measured_joint_position': {'status': 'available', 'units': ['m', 'm'], 'value': [.011, .012]},
                   'previous_control': {'status': 'available', 'value': {'position': [0., 0.]}}}
        text = render_fingers({'schema': 'justrua-gripper-v1', 'robots': [reading]})
        self.assertIn('left_finger_position_mm=[11, 12]', text)
        self.assertIn('left_drive_target_mm=[0, 0]', text)
        self.assertIn('left_normal_N=[unknown, unknown]', text)
        self.assertIn('right_finger_position_mm=[unknown, unknown]', text)


if __name__ == '__main__':
    unittest.main()
