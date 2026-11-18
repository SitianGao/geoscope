"""Atomic, hash-bound index. Gallery paths come only from operator configuration."""
import hashlib
import io
import json
import os
import re
import uuid
from pathlib import Path
from PIL import Image


class RetrievalError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def normalize(vector):
    import numpy as np
    a = np.asarray(vector, dtype=np.float32)
    if a.ndim != 1 or not np.isfinite(a).all() or float(np.linalg.norm(a)) <= 0:
        raise RetrievalError('invalid_embedding', 'Embedding must be finite and nonzero')
    return a/np.linalg.norm(a)


class Index:
    def __init__(self, model, directory, image_root):
        self.model, self.directory, self.image_root = model, Path(directory), Path(image_root).resolve()

    def image_path(self, image_id):
        if not isinstance(image_id, str) or not re.fullmatch('[a-f0-9]{12}', image_id):
            raise RetrievalError('invalid_arguments', 'Gallery image IDs must be platform session IDs')
        path = (self.image_root/image_id/'original.png').resolve()
        if not path.is_relative_to(self.image_root):
            raise RetrievalError('invalid_arguments', 'Image escapes configured root')
        return path

    def build(self, image_ids):
        import numpy as np
        if not image_ids or len(image_ids) != len(set(image_ids)):
            raise RetrievalError('empty_gallery', 'Gallery must be nonempty with unique IDs')
        records, vectors = [], []
        for image_id in image_ids:
            raw = self.image_path(image_id).read_bytes()
            with Image.open(io.BytesIO(raw)) as image:
                vectors.append(normalize(self.model.image(image)))
            records.append({'image_id': image_id, 'image_sha256': hashlib.sha256(raw).hexdigest()})
        vectors = np.stack(vectors)
        buffer = io.BytesIO(); np.save(buffer, vectors, allow_pickle=False)
        blob = buffer.getvalue()
        manifest = {'embedding_version': self.model.version, 'images': records,
                    'vectors_sha256': hashlib.sha256(blob).hexdigest(), 'model_info': self.model.info}
        manifest['index_version'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)
        vector_file = self.directory/(manifest['vectors_sha256']+'.npy')
        temp = self.directory/(uuid.uuid4().hex+'.tmp'); temp.write_bytes(blob); os.replace(temp, vector_file)
        temp = self.directory/(uuid.uuid4().hex+'.tmp')
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, self.directory/'manifest.json')
        return manifest

    def load(self):
        import numpy as np
        try:
            manifest = json.loads((self.directory/'manifest.json').read_text(encoding='utf-8'))
            digest = manifest['vectors_sha256']
            if not re.fullmatch('[a-f0-9]{64}', digest):
                raise ValueError('Bad vector digest')
            blob = (self.directory/(digest+'.npy')).read_bytes()
            if hashlib.sha256(blob).hexdigest() != digest or manifest['embedding_version'] != self.model.version:
                raise RetrievalError('stale_index', 'Index version or checksum mismatch; rebuild required')
            for record in manifest['images']:
                if hashlib.sha256(self.image_path(record['image_id']).read_bytes()).hexdigest() != record['image_sha256']:
                    raise RetrievalError('stale_index', 'Gallery image changed; rebuild index')
            vectors = np.load(io.BytesIO(blob), allow_pickle=False)
            if vectors.ndim != 2 or len(vectors) != len(manifest['images']) or len(vectors) == 0 or not np.isfinite(vectors).all():
                raise ValueError('Invalid index vectors')
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
                raise ValueError('Unnormalized index')
            return manifest, vectors
        except RetrievalError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RetrievalError('missing_index', 'Index missing or invalid; rebuild required') from exc

    def search(self, query, top_k=5):
        import numpy as np
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise RetrievalError('invalid_arguments', 'Query must be nonempty and at most 2000 characters')
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise RetrievalError('invalid_arguments', 'top_k must be an integer in [1, 100]')
        manifest, vectors = self.load()
        query_vector = normalize(self.model.text(query))
        if query_vector.shape[0] != vectors.shape[1]:
            raise RetrievalError('stale_index', 'Text/image dimensions differ')
        scores = vectors @ query_vector
        order = np.argsort(-scores, kind='stable')[:top_k]
        return {'query': query, 'model': 'RemoteCLIP', 'embedding_version': self.model.version,
                'index_version': manifest['index_version'], 'model_info': self.model.info,
                'results': [{**manifest['images'][int(i)], 'rank': rank, 'similarity': float(scores[i])} for rank, i in enumerate(order, 1)],
                'status': 'completed', 'real_execution': True,
                'notice': 'Normalized embedding cosine similarity is not probability or pixel localization'}
