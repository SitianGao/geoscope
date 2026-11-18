"""One local command starts the real CLIP sidecar and integrated workbench."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from http.server import ThreadingHTTPServer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from national_server import Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/RemoteCLIP-RN50.pt')
    parser.add_argument('--sha256', default='11657d33a87fc197e19a7654a76a632415ebea42feb8a61025ef389f8990c0f9')
    parser.add_argument('--architecture', default='RN50')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--port', type=int, default=4180)
    parser.add_argument('--clip-port', type=int, default=4190)
    parser.add_argument('--gallery', default='outputs/remoteclip-gallery.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if not Path(args.weights).is_file():
        parser.error('Missing trusted weights. Run python -m services.remoteclip.download_weights first.')
    gallery = Path(args.gallery)
    if not gallery.is_file():
        gallery.parent.mkdir(parents=True, exist_ok=True)
        gallery.write_text(json.dumps([server.new_session(sample)['id'] for sample in server.SAMPLES]), encoding='utf-8')
    os.environ['GEO_REMOTECLIP_URL'] = f'http://127.0.0.1:{args.clip_port}'
    http = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    Path('outputs').mkdir(exist_ok=True)
    with open('outputs/remoteclip-service.log', 'a', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-m', 'services.remoteclip.service', '--weights', args.weights,
                                    '--sha256', args.sha256, '--architecture', args.architecture, '--device', args.device,
                                    '--gallery', args.gallery, '--port', str(args.clip_port)],
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        print(f'Workbench: http://127.0.0.1:{args.port}/?workbench=national', flush=True)
        print('RemoteCLIP loading/indexing; inspect retrieval status or outputs/remoteclip-service.log. Ctrl+C stops both.', flush=True)
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            http.server_close(); process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()


if __name__ == '__main__': main()
