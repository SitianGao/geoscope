import os
import socket
import sys
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.manage_server import Runtime


@unittest.skipUnless(os.name == 'posix', 'Linux server process ownership checks')
class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = Runtime({'root': str(self.root), 'sam_python': sys.executable,
            'sam_sha256': 'test', 'sam_weights': 'test', 'sam_tokenizer': 'test',
            'agent_weights': 'test', 'clip_weights': 'test', 'clip_sha256': 'test'})

    def tearDown(self): self.temp.cleanup()

    def test_stop_never_signals_an_unrelated_pid(self):
        (self.root/'outputs/clip.pid').write_text(str(os.getpid()))
        with patch('scripts.manage_server.os.kill') as kill:
            self.assertEqual(self.runtime.stop('clip')['state'], 'not_running_or_not_owned')
            kill.assert_not_called()

    def test_start_preserves_occupied_ports(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
            with patch.object(self.runtime, 'spec', return_value=([sys.executable], self.root, port, 'marker')):
                with patch('scripts.manage_server.subprocess.Popen') as spawn:
                    with self.assertRaises(OSError): self.runtime.start('clip')
                    spawn.assert_not_called()

    def test_owned_child_is_idempotent_and_stops(self):
        spec = ([sys.executable, '-c', 'import time; time.sleep(60)'], self.root, 0, '-c')
        with patch.object(self.runtime, 'spec', return_value=spec):
            row = self.runtime.start('clip'); pid = row['pid']
            try:
                deadline = time.monotonic()+2
                while not self.runtime.owns('clip', pid) and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(self.runtime.owns('clip', pid))
                self.assertEqual(self.runtime.start('clip')['state'], 'already_running')
                self.assertEqual(self.runtime.stop('clip')['state'], 'stopped')
                self.assertFalse(self.runtime.owns('clip', pid))
            finally:
                if self.runtime.owns('clip', pid): self.runtime.stop('clip')
                if 'clip' in self.runtime.children:
                    child = self.runtime.children.pop('clip')
                    if child.poll() is None: child.terminate()
                    child.wait(timeout=2)

    def test_only_gpu_zero_is_selected(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}):
            self.assertEqual(self.runtime.environment()['CUDA_VISIBLE_DEVICES'], '0')

    def test_restart_after_a_completed_tcp_connection(self):
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0)); listener.listen()
        port = listener.getsockname()[1]
        client = socket.create_connection(('127.0.0.1', port))
        connection, _ = listener.accept()
        connection.close(); client.recv(1); client.close(); listener.close()
        spec = ([sys.executable, '-c', 'import time; time.sleep(60)'], self.root, port, '-c')
        with patch.object(self.runtime, 'spec', return_value=spec):
            row = self.runtime.start('clip')
            self.assertEqual(row['state'], 'starting')
            self.runtime.stop('clip')
