"""Synthetic index tests are not model evidence; real inference is opt-in."""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from services.remoteclip.index import Index, RetrievalError


class SyntheticEncoder:
    version = 'SYNTHETIC-UNIT-TEST'
    info = {'model': 'SYNTHETIC-UNIT-TEST'}
    def image(self, image): return [image.getpixel((0, 0))[0]+1, 1]
    def text(self, text): return [1, 0]


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        for sid, value in [('aaaaaaaaaaaa', 0), ('bbbbbbbbbbbb', 255)]:
            (self.root/sid).mkdir(); Image.new('RGB', (2, 2), value).save(self.root/sid/'original.png')
        self.index = Index(SyntheticEncoder(), self.root/'index', self.root)

    def tearDown(self): self.temp.cleanup()

    def test_sorted_normalized_and_truncated(self):
        self.index.build(['aaaaaaaaaaaa', 'bbbbbbbbbbbb'])
        result = self.index.search('test', 10)
        self.assertEqual(len(result['results']), 2)
        self.assertEqual(result['results'][0]['image_id'], 'bbbbbbbbbbbb')
        self.assertLessEqual(result['results'][0]['similarity'], 1)

    def test_invalid_queries_gallery_and_missing_index(self):
        with self.assertRaises(RetrievalError): self.index.build([])
        for query, k in [('', 2), ('x', 0), ('x', True), ('x', 1.5)]:
            with self.subTest(query=query, k=k), self.assertRaises(RetrievalError): self.index.search(query, k)
        with self.assertRaises(RetrievalError): self.index.search('test')

    def test_image_and_embedding_changes_are_rejected(self):
        self.index.build(['aaaaaaaaaaaa'])
        self.index.model.version = 'changed'
        with self.assertRaises(RetrievalError): self.index.search('x')
        self.index.model.version = 'SYNTHETIC-UNIT-TEST'
        Image.new('RGB', (2, 2), 50).save(self.root/'aaaaaaaaaaaa/original.png')
        with self.assertRaises(RetrievalError): self.index.search('x')


@unittest.skipUnless(os.environ.get('REMOTECLIP_TEST_WEIGHTS'), 'Real RemoteCLIP requires explicit trusted weights')
class RealModelTests(unittest.TestCase):
    def test_real_image_text_embeddings(self):
        from services.remoteclip.model import RemoteCLIP
        weights = os.environ['REMOTECLIP_TEST_WEIGHTS']
        model = RemoteCLIP(weights, os.environ['REMOTECLIP_TEST_SHA256'], os.environ.get('REMOTECLIP_TEST_ARCH', 'ViT-B-32'))
        import numpy as np
        a = model.image(Image.open(Path(__file__).parents[1]/'web/assets/airport.jpg'))
        b = model.text('airport with airplanes')
        self.assertEqual(a.shape, b.shape); self.assertTrue(np.isfinite(a).all()); self.assertTrue(np.isfinite(b).all())
