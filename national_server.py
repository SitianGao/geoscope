"""Opt-in national API; leaves server.py and the old launch command untouched."""
import csv
import json
import io
import zipfile
import os
import re
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import server
from geoscope_tools.backend import Backend
from geoscope_tools.registry import REGISTRY
from geoscope_tools.planner import PLAN_SCHEMA, execute_plan
from geoscope_tools.planner import validate_plan
from geoscope_tools.statistics import ToolError

PREFIX = '/api/national/'
ROUTES = {'tools/region-statistics': 'region_statistics', 'tools/connected-regions': 'connected_region_analysis',
          'tools/mask-compare': 'mask_compare', 'tools/spatial-relations': 'spatial_relation_analysis',
          'tools/report': 'report_generation', 'tools/adaptive-refinement': 'adaptive_refinement',
          'tools/geo-measurement': 'geo_measurement'}
EXPERIMENT_ROOT = 'evaluation/national_final/results/lab_runs'


def _mean(rows, key):
    values = []
    for row in rows:
        raw = row.get(key)
        if raw in (None, ''):
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return round(sum(values)/len(values), 6) if values else None


def experiment_runs():
    """Recompute per-image CSV means so the page is not limited to static archives."""
    root = server.ROOT/EXPERIMENT_ROOT
    runs = []
    if not root.is_dir():
        return runs
    for folder in sorted(root.iterdir()):
        results = folder/'results'
        if not results.is_dir():
            continue
        for csv_file in sorted(results.glob('*.csv')):
            try:
                with csv_file.open(encoding='utf-8', newline='') as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, ValueError, csv.Error):
                continue
            ok = [row for row in rows if row.get('status') == 'success']
            runs.append({'run': folder.name, 'file': csv_file.name,
                         'rows': len(rows), 'successes': len(ok),
                         'metrics': {'iou': _mean(ok, 'iou'), 'dice': _mean(ok, 'dice'),
                                     'precision': _mean(ok, 'precision'), 'recall': _mean(ok, 'recall'),
                                     'inference_ms': _mean(ok, 'service_inference_ms')}})
    return runs


def clip_request(path, payload=None):
    url = os.environ.get('GEO_REMOTECLIP_URL', '').rstrip('/')
    if not url:
        raise ToolError('service_unavailable', 'RemoteCLIP service is not configured')
    req = urllib.request.Request(url+path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=120 if payload else 5) as response:
            data = json.loads(response.read(4*1024*1024))
    except urllib.error.HTTPError as exc:
        try:
            data = json.loads(exc.read(16000))
        finally:
            exc.close()
        error = data.get('error', {})
        raise ToolError(error.get('code', 'service_unavailable'), error.get('message', 'RemoteCLIP request failed')) from None
    except (OSError, ValueError):
        raise ToolError('service_unavailable', 'RemoteCLIP unavailable or invalid response') from None
    if not isinstance(data, dict):
        raise ToolError('service_unavailable', 'Invalid RemoteCLIP response')
    return data


class Handler(server.Handler):
    def envelope(self, data=None, error=None, status='completed', args=None):
        args = args if isinstance(args, dict) else {}
        result = {'success': error is None and status == 'completed', 'request_id': uuid.uuid4().hex,
                  'image_id': args.get('image_id'), 'result_id': args.get('result_id'),
                  'status': status, 'data': data or {}, 'error': error,
                  'meta': {'source': 'geoscope_tool', 'tool_version': 'national-v1',
                           'real_execution': error is None and status == 'completed'}}
        if self.command == 'POST':
            folder = Backend(server).safe_path('national_events')
            folder.mkdir(exist_ok=True)
            server.save_json(folder/(result['request_id']+'.json'),
                             {'at': datetime.now(timezone.utc).isoformat(), 'route': self.path,
                              'arguments': args, 'response': result})
        http_status = 200 if result['success'] else 503 if status == 'not_ready' else 400
        return self.json(result, http_status)

    def fail(self, exc, args=None):
        code = exc.code if isinstance(exc, ToolError) else 'service_unavailable'
        message = str(exc) if isinstance(exc, ToolError) else 'Operation failed; inspect server logs'
        state = 'not_ready' if code in ('service_unavailable', 'missing_dependency', 'missing_index', 'stale_index', 'gpu_out_of_memory') else 'failed'
        return self.envelope(error={'code': code, 'message': message}, status=state, args=args)

    def do_GET(self):
        path = self.path.split('?')[0]
        export = re.fullmatch(r'/api/export/([a-f0-9]{12})/([a-f0-9]{12})', path)
        if export:
            sid, rid = export.groups()
            session = server.get_session(sid)
            if not session or not any(r['id'] == rid and r.get('mask_url') for r in session['runs']):
                return self.send_error(404)
            backend = Backend(server)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.write(backend.safe_path(sid, 'original.png'), 'original.png')
                for name in ('mask.png', 'full_mask.png', 'overlay.png', 'result.json', 'statistics.json',
                             'run_log.json', 'report.md', 'afr.json', 'probability_map.png', 'disagreement.png'):
                    file = backend.safe_path(sid, rid, name)
                    if file.is_file(): archive.write(file, name)
            raw = buffer.getvalue(); self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Disposition', f'attachment; filename="geoscope_{rid}.zip"')
            self.send_header('Content-Length', str(len(raw))); self.end_headers(); return self.wfile.write(raw)
        # Captain produces these artifacts, but the legacy handler only allows
        # original/mask/overlay. Serve a tightly bounded PNG allowlist here.
        artifact = re.fullmatch(r'/experiments/([a-f0-9]{12})/([a-f0-9]{12})/(disagreement|probability_map)\.png', path)
        if artifact:
            sid, rid, name = artifact.groups()
            session = server.get_session(sid)
            if not session or not any(r['id'] == rid for r in session['runs']):
                return self.send_error(404)
            file = Backend(server).safe_path(sid, rid, name+'.png')
            if not file.is_file():
                return self.send_error(404)
            raw = file.read_bytes(); self.send_response(200)
            self.send_header('Content-Type', 'image/png'); self.send_header('Content-Length', str(len(raw)))
            self.end_headers(); return self.wfile.write(raw)
        if not path.startswith(PREFIX):
            return super().do_GET()
        try:
            route = path[len(PREFIX):]
            if route == 'tools/registry':
                tools = {name: dict(entry) for name, entry in REGISTRY.items()}
                tools['adaptive_refinement'].update(available=callable(getattr(server, 'run_adaptive_pipeline', None)), reason='Uses captain AFR via existing Harness; live model configuration required')
                tools['semantic_image_retrieval'] = {'available': bool(os.environ.get('GEO_REMOTECLIP_URL')), 'reason': 'Inspect retrieval/status for actual model and index readiness'}
                return self.envelope({'tools': [{'name': key, **value} for key, value in tools.items()],
                                      'tools_by_name': tools, 'plan_schema': PLAN_SCHEMA})
            if route == 'retrieval/status':
                data = clip_request('/ready')
                if data.get('ready') is not True or data.get('index_ready') is not True:
                    return self.envelope(data, {'code': 'service_unavailable', 'message': data.get('error') or data.get('index_error') or 'Index not ready'}, 'not_ready')
                return self.envelope({**data, 'model': (data.get('model_info') or {}).get('model', 'RemoteCLIP'),
                                      'index_size': data.get('image_count'), 'max_top_k': 100})
            if route == 'experiments':
                runs = experiment_runs()
                return self.envelope({'runs': runs, 'root': EXPERIMENT_ROOT,
                                      'source': 'server_csv_recompute',
                                      'note': 'Means recomputed from the per-image CSVs on this deployment; not a live re-run.'})
            if re.fullmatch(r'artifacts/[a-f0-9]{64}\.png', route):
                file = Backend(server).safe_path('national_artifacts', route.split('/')[-1])
                if not file.is_file():
                    return self.send_error(404)
                raw = file.read_bytes(); self.send_response(200)
                self.send_header('Content-Type', 'image/png'); self.send_header('Content-Length', str(len(raw)))
                self.end_headers(); return self.wfile.write(raw)
            raise ToolError('unsupported_tool', 'Unknown national route')
        except Exception as exc:
            return self.fail(exc)

    def do_POST(self):
        if not self.path.startswith(PREFIX):
            return super().do_POST()
        args = None
        try:
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                raise ToolError('invalid_arguments', 'Invalid Content-Length') from None
            if not 0 < length <= 64000:
                raise ToolError('invalid_arguments', 'Invalid request size')
            try:
                args = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError):
                raise ToolError('invalid_arguments', 'Invalid JSON') from None
            if type(args) is not dict:
                raise ToolError('invalid_arguments', 'Expected JSON object')
            route = self.path[len(PREFIX):]
            backend = Backend(server)
            # B's published comparison payload names the target explicitly and
            # infers its target category from the bound saved version.
            if route == 'tools/mask-compare' and 'target_result_id' in args:
                if 'result_id' in args and args['result_id'] != args['target_result_id']:
                    raise ToolError('invalid_arguments', 'Conflicting target result IDs')
                args = dict(args)
                args['result_id'] = args.pop('target_result_id')
                session = server.get_session(args.get('image_id'))
                run = next((r for r in session['runs'] if r['id'] == args['result_id']), None) if session else None
                if not run:
                    raise ToolError('missing_dependency', 'Target version does not belong to this image')
                args.setdefault('target', run.get('task', {}).get('target'))
            if route == 'tools/adaptive-refinement':
                return self.envelope(backend.refine(args), args=args)
            if route in ROUTES:
                return self.envelope(backend.execute(ROUTES[route], args), args=args)
            if route == 'plans/execute':
                validate_plan(args)
                session = server.get_session(args['image_id'])
                if not session:
                    raise ToolError('missing_dependency', 'Image session does not exist')
                if not session['lock'].acquire(blocking=False):
                    raise ToolError('service_unavailable', 'Image session is busy')
                try:
                    result = execute_plan(args, backend)
                    backend.save_execution(args, result)
                finally:
                    session['lock'].release()
                return self.envelope(result, result['error'], result['status'], args)
            if route == 'retrieval/search':
                if set(args) - {'query', 'top_k'} or not isinstance(args.get('query'), str) or not args['query'].strip() or len(args['query']) > 2000:
                    raise ToolError('invalid_arguments', 'Search expects a nonempty query and optional top_k')
                if type(args.get('top_k', 5)) is not int or not 1 <= args.get('top_k', 5) <= 100:
                    raise ToolError('invalid_arguments', 'top_k must be integer [1,100]')
                result = clip_request('/search', args)
                if result.get('status') != 'completed' or result.get('real_execution') is not True:
                    raise ToolError('service_unavailable', 'RemoteCLIP did not confirm actual inference')
                # Selection uses the existing GET /api/session/{image_id} route.
                import hashlib
                for item in result.get('results', []):
                    sid = item.get('image_id')
                    session = server.get_session(sid)
                    if not session or hashlib.sha256(backend.safe_path(sid, 'original.png').read_bytes()).hexdigest() != item.get('image_sha256'):
                        raise ToolError('stale_index', 'Candidate is no longer bound to the current platform image')
                    item['session_url'] = '/api/session/'+sid
                    item['thumbnail_url'] = session['image_url']
                    item['name'] = session['name']
                    item['width'], item['height'] = session['width'], session['height']
                return self.envelope(result, args=args)
            raise ToolError('unsupported_tool', 'Unknown national route')
        except Exception as exc:
            return self.fail(exc, args)


if __name__ == '__main__':
    port = int(os.environ.get('GEO_PORT', '4180'))
    print(f'GeoScope national tools: http://127.0.0.1:{port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
