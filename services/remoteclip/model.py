import hashlib
import importlib.metadata
import json
from pathlib import Path


class RemoteCLIP:
    def __init__(self, weights, expected_sha256, architecture='ViT-B-32', device='cpu'):
        if architecture not in ('RN50', 'ViT-B-32', 'ViT-L-14'):
            raise ValueError('Unsupported RemoteCLIP architecture')
        path = Path(weights)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if len(expected_sha256) != 64 or digest != expected_sha256:
            raise ValueError('Weight SHA256 does not match the configured trusted checkpoint')
        import torch
        import open_clip
        self.torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(architecture, pretrained=None)
        self.tokenizer = open_clip.get_tokenizer(architecture)
        state = torch.load(str(path), map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.info = {'model': 'RemoteCLIP', 'architecture': architecture, 'weights_sha256': digest,
            'upstream': 'https://github.com/ChenDelong1999/RemoteCLIP',
            'weight_repository': 'chendelong/RemoteCLIP', 'device': device,
            'open_clip_version': importlib.metadata.version('open-clip-torch'),
            'torch_version': torch.__version__, 'preprocess': repr(self.preprocess),
            'tokenizer': type(self.tokenizer).__name__}
        self.version = hashlib.sha256(json.dumps(self.info, sort_keys=True).encode()).hexdigest()
        self.info['embedding_version'] = self.version

    def image(self, image):
        with self.torch.inference_mode():
            result = self.model.encode_image(self.preprocess(image.convert('RGB')).unsqueeze(0).to(self.device))
        return result.float().cpu().numpy()[0]

    def text(self, query):
        with self.torch.inference_mode():
            result = self.model.encode_text(self.tokenizer([query]).to(self.device))
        return result.float().cpu().numpy()[0]
