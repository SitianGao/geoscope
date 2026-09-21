"""Checks for teammate A's dependency-free vLLM transport."""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

REMOTEAGENT_ROOT = Path(__file__).resolve().parents[1] / 'services/remoteagent/upstream'
sys.path.insert(0, str(REMOTEAGENT_ROOT))

from remoteagent.llm.vllm import VLLMChatClient


class FakeVLLM(BaseHTTPRequestHandler):
    authorization = None

    def log_message(self, *args): pass

    def reply(self, value):
        raw = json.dumps(value).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == '/v1/models':
            return self.reply({'data': [{'id': 'REMOTEAGENT-FAKE-TEST'}]})
        self.send_error(404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        type(self).authorization = self.headers.get('Authorization')
        if self.path == '/v1/chat/completions' and payload['model'] == 'REMOTEAGENT-FAKE-TEST':
            return self.reply({'choices': [{'message': {'content': '<answer>test</answer>'}}]})
        self.send_error(404)


class VLLMTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(('127.0.0.1', 0), FakeVLLM)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.httpd.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown(); cls.httpd.server_close(); cls.thread.join()

    def test_chat_uses_openai_compatible_path_and_key(self):
        with patch.dict(os.environ, {'GEO_SERVICE_PROXY_MODE': 'direct'}):
            client = VLLMChatClient(self.url, api_key='test-key-not-a-secret')
            result = client.chat([{'role': 'user', 'content': 'test'}],
                                 model='REMOTEAGENT-FAKE-TEST')
        self.assertEqual(result, '<answer>test</answer>')
        self.assertEqual(FakeVLLM.authorization, 'Bearer test-key-not-a-secret')

    def test_model_discovery(self):
        with patch.dict(os.environ, {'GEO_SERVICE_PROXY_MODE': 'direct'}):
            self.assertEqual(VLLMChatClient.list_first_model_id(self.url),
                             'REMOTEAGENT-FAKE-TEST')


if __name__ == '__main__': unittest.main()
