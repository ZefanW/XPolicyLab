"""Original-resolution metric depth from each RGB image and Depth Pro's learned FOV."""
import base64
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
MODEL = 'apple/DepthPro-hf'
REVISION = 'de816c8ce7168afcb231f96d501d72b869d0beda'
CHECKPOINT_FILES = {
    'config.json': '995993be477aefdf41468ca51e171afd26b59aeee4b0afc4ef16957f0b622222',
    'preprocessor_config.json': '88c3e95c10d3780840674bfd52581de03c547cf0a4976b9c4779dd8e2fc5af05',
    'model.safetensors': '9c6811e3165485b9a94a204329860cb333a79877e757eb795a179a4ea34bbcf7',
}
GUIDE = '''Each predicted depth image comes from its named RGB view and the predictor's learned field of view. Colors estimate camera-forward distance in metres. Blue = 0, cyan = 0.5, green = 1, yellow = 1.5 and red = 2. White = above 2 m, magenta = unknown. Scale, thin objects and occlusion boundaries can be wrong. Original RGB and measured camera calibration are available for checking geometry. An unavailable view is explicitly labelled and omitted.'''


def verify_checkpoint(directory):
    for name, expected in CHECKPOINT_FILES.items():
        digest = hashlib.sha256()
        with (directory / name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'{MODEL} {REVISION}: {name} does not match the pinned checkpoint')
    if json.loads((directory / 'config.json').read_text()).get('use_fov_model') is not True:
        raise ValueError('Depth Pro requires the learned field-of-view head')


class DepthPro:
    def __init__(self, directory, device):
        if not directory:
            raise ValueError(f'Set JUSTRUA_DEPTHPRO_CHECKPOINT to the local {MODEL} directory at revision {REVISION}')
        directory = Path(directory).expanduser().resolve()
        verify_checkpoint(directory)
        if importlib.metadata.version('transformers') != '4.57.6':
            raise ValueError('Install transformers==4.57.6 for the evaluated Depth Pro decoder')
        import torch
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor

        self.device = torch.device(device)
        self.dtype = torch.float32 if self.device.type == 'cpu' else torch.float16
        options = dict(local_files_only=True, trust_remote_code=False)
        self.processor = DepthProImageProcessor.from_pretrained(directory, **options)
        self.model = DepthProForDepthEstimation.from_pretrained(
            directory, use_safetensors=True, torch_dtype=self.dtype,
            attn_implementation='sdpa', **options).to(self.device).eval()

    def infer(self, rgb):
        import torch
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors='pt')
        if set(inputs) != {'pixel_values'}:
            raise ValueError('Depth Pro inputs must contain only RGB pixel values')
        with torch.inference_mode():
            output = self.model(pixel_values=inputs['pixel_values'].to(self.device, dtype=self.dtype))
            if output.field_of_view is None:
                raise ValueError('The model did not predict the field of view')
            output.field_of_view = output.field_of_view.float()
            output.predicted_depth = output.predicted_depth.float()
            decoded = self.processor.post_process_depth_estimation(output, target_sizes=[rgb.shape[:2]])[0]
            for key in ('field_of_view', 'focal_length'):
                value = float(decoded[key].cpu())
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f'Invalid predicted {key}')
            return decoded['predicted_depth'].to('cpu', dtype=torch.float32).numpy()

    def close(self):
        self.model = self.processor = None


def colorize(depth, valid):
    anchors = np.array([[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]], float)
    loc = np.clip(np.where(valid, depth, 0) * 2., 0., 4.)
    low = np.minimum(loc.astype(int), 3)
    fraction = (loc - low)[..., None]
    image = np.rint(anchors[low] * (1 - fraction) + anchors[low + 1] * fraction).astype(np.uint8)
    image[valid & (depth > 2.)] = (255, 255, 255)
    image[~valid] = (255, 0, 255)
    return image


def predicted_image(backend, rgb):
    """Return a PNG data URI or the reason this view is unavailable."""
    pixels = np.array(rgb[:, :, :3], copy=True, order='C')
    pixels.setflags(write=False)
    try:
        depth = backend.infer(pixels)
    except Exception as exc:
        return None, f'prediction failed ({type(exc).__name__})'
    if not isinstance(depth, np.ndarray) or depth.dtype != np.float32 or depth.shape != pixels.shape[:2]:
        return None, 'prediction shape or dtype mismatch'
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return None, 'no positive finite predictions'
    buffer = io.BytesIO()
    Image.fromarray(colorize(depth, valid)).save(buffer, format='PNG')
    uri = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    return uri, None
