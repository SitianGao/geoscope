"""Run with python -m services.remoteclip.service --help."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .index import Index, RetrievalError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--architecture', default='ViT-B-32', choices=['RN50', 'ViT-B-32', 'ViT-L-14'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--image-root', default='experiments')
    parser.add_argument('--index', default='outputs/remoteclip-index')
    parser.add_argument('--gallery', help='JSON array of existing platform session IDs; build index at startup')
    parser.add_argument('--port', type=int, default=4190)
    args = parser.parse_args()
    state = {'ready': False, 'error': 'Model loading', 'model_info': None}
    lock = threading.Lock()
    index = None

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, status=200):
            raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_GET(self):
            if self.path not in ('/ready', '/model'):
                return self.reply({'error': 'Not found'}, 404)
            with lock:
                result = dict(state)
                if result['ready']:
                    try:
                        manifest, _ = index.load()
                        result.update(index_ready=True, index_version=manifest['index_version'], image_count=len(manifest['images']))
                    except RetrievalError as exc:
                        result.update(index_ready=False, index_error=exc.code)
                return self.reply(result)

        def do_POST(self):
            if self.path != '/search':
                return self.reply({'error': 'Not found'}, 404)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16000:
                    raise RetrievalError('invalid_arguments', 'Invalid request size')
                data = json.loads(self.rfile.read(length))
                if type(data) is not dict or set(data)-{'query', 'top_k'}:
                    raise RetrievalError('invalid_arguments', 'Unknown search arguments')
                with lock:
                    if not state['ready']:
                        raise RetrievalError('service_unavailable', state['error'])
                    result = index.search(data.get('query'), data.get('top_k', 5))
                return self.reply(result)
            except (ValueError, TypeError) as exc:
                code = getattr(exc, 'code', 'invalid_arguments')
                return self.reply({'status': 'failed' if code == 'invalid_arguments' else 'not_ready', 'real_execution': False,
                                   'error': {'code': code, 'message': str(exc)}}, 400 if code == 'invalid_arguments' else 503)
            except Exception as exc:
                code = 'gpu_out_of_memory' if 'out of memory' in str(exc).lower() else 'service_unavailable'
                return self.reply({'status': 'not_ready', 'real_execution': False, 'error': {'code': code, 'message': 'Inference failed; inspect service logs'}}, 503)

    def load():
        nonlocal index
        try:
            from .model import RemoteCLIP
            model = RemoteCLIP(args.weights, args.sha256, args.architecture, args.device)
            candidate = Index(model, args.index, args.image_root)
            if args.gallery:
                from pathlib import Path
                candidate.build(json.loads(Path(args.gallery).read_text(encoding='utf-8')))
            with lock:
                index = candidate
                state.update(ready=True, error=None, model_info=model.info)
            print(json.dumps({'event': 'model_loaded', **model.info}), flush=True)
        except Exception as exc:
            with lock:
                state.update(ready=False, error=f'{type(exc).__name__}: {exc}')
            print(json.dumps({'event': 'model_load_failed', 'error': state['error']}), flush=True)

    threading.Thread(target=load, daemon=True).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
