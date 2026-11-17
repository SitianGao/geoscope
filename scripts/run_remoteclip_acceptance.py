"""Start isolated real services, collect evidence, then stop both services."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import national_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/RemoteCLIP-RN50.pt')
    parser.add_argument('--sha256', default='11657d33a87fc197e19a7654a76a632415ebea42feb8a61025ef389f8990c0f9')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); clip_port = sock.getsockname()[1]
    os.environ['GEO_REMOTECLIP_URL'] = f'http://127.0.0.1:{clip_port}'
    server = ThreadingHTTPServer(('127.0.0.1', 0), national_server.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    log = root/'docs/evidence/remoteclip_startup.txt'; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, '-m', 'services.remoteclip.service', '--weights', args.weights,
                                    '--sha256', args.sha256, '--architecture', 'RN50', '--gallery', 'outputs/remoteclip-gallery.json',
                                    '--port', str(clip_port)], cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            deadline = time.monotonic()+180
            state = None
            while time.monotonic() < deadline:
                if process.poll() is not None: raise RuntimeError('Model service exited; see startup evidence')
                try:
                    with urllib.request.urlopen(os.environ['GEO_REMOTECLIP_URL']+'/ready', timeout=5) as response:
                        state = json.loads(response.read())
                    if state.get('ready') and state.get('index_ready'): break
                    if state.get('error') and state['error'] != 'Model loading': raise RuntimeError(state['error'])
                except OSError: pass
                time.sleep(.5)
            else: raise RuntimeError(f'Model readiness timeout: {state}')
            subprocess.run([sys.executable, 'scripts/remoteclip_evidence.py', '--platform', f'http://127.0.0.1:{server.server_port}'], cwd=root, check=True)
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__': main()
