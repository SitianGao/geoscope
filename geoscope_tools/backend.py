"""Adapter for v0.7 session/run IDs. Read-only with respect to existing runs."""
import hashlib
import io
import json
import uuid
from PIL import Image, ImageOps
from .statistics import ToolError, binary, region_statistics, apply_scope
from .registry import validate_call
from .region_analysis import connected_region_analysis, spatial_relation_analysis
from .version_compare import mask_compare


class Backend:
    def __init__(self, server):
        self.server = server

    def safe_path(self, *parts):
        root = self.server.DATA.resolve()
        path = root.joinpath(*parts).resolve()
        if not path.is_relative_to(root):
            raise ToolError('invalid_arguments', 'Artifact escapes storage root')
        return path

    def load(self, image_id, result_id, target):
        s = self.server
        if not all(isinstance(v, str) and s.SID_RE.fullmatch(v) for v in (image_id, result_id)):
            raise ToolError('invalid_arguments', 'Invalid image/result ID; local paths are prohibited')
        session = s.get_session(image_id)
        if not session:
            raise ToolError('missing_dependency', 'Image session does not exist')
        # Result IDs are resolved only inside the bound image session.
        run = next((r for r in session['runs'] if r['id'] == result_id), None)
        if not run or run.get('session_id') != image_id or not run.get('mask_url'):
            raise ToolError('missing_dependency', 'Result does not belong to this image or has no mask')
        if run.get('task', {}).get('target') != target:
            raise ToolError('invalid_arguments', 'Target does not match the saved result')
        try:
            raw = self.safe_path(image_id, result_id, 'full_mask.png').read_bytes()
            if hashlib.sha256(raw).hexdigest() != run.get('full_mask_sha256'):
                raise ToolError('artifact_integrity_error', 'Full mask checksum verification failed')
            with Image.open(io.BytesIO(raw)) as im:
                mask = binary(im).copy()
            with Image.open(self.safe_path(image_id, 'original.png')) as original:
                if original.size != (session['width'], session['height']) or mask.size != original.size:
                    raise ToolError('invalid_arguments', 'Mask/image coordinate dimensions differ')
            # full_mask is saved before invert and range restriction in v0.7.
            if run.get('task', {}).get('invert') or run.get('execution_plan', {}).get('invert'):
                mask = ImageOps.invert(mask)
            expected = s.constrain(mask, run['task']['side'], run.get('task_options', {}).get('roi'))
            with Image.open(self.safe_path(image_id, result_id, 'mask.png')) as final:
                final = binary(final).copy()
            if final.size != mask.size or final.tobytes() != expected.tobytes():
                raise ToolError('artifact_integrity_error', 'Final mask does not match the verified full mask and saved scope')
            mask = final
        except (OSError, ValueError) as exc:
            if isinstance(exc, ToolError):
                raise
            raise ToolError('missing_dependency', 'Missing or unreadable artifact') from exc
        return {'mask': mask, 'provenance': {'image_id': image_id, 'result_id': result_id,
                'target': target, 'full_mask_sha256': run['full_mask_sha256'],
                'image_sha256': hashlib.sha256(self.safe_path(image_id, 'original.png').read_bytes()).hexdigest(),
                'model_execution': run.get('mode') == 'live', 'source_mode': run.get('mode'),
                'mask_basis': 'saved_final_mask', 'saved_scope': run['task']['side'],
                'saved_roi': run.get('task_options', {}).get('roi')}}

    def execute(self, name, args, loaded=None):
        validate_call(name, args)
        current = loaded or self.load(args['image_id'], args['result_id'], args['target'])
        if any(current['provenance'][key] != args[key] for key in ('image_id', 'result_id', 'target')):
            raise ToolError('invalid_arguments', 'Prior artifact binding mismatch')
        mask = current['mask']
        scope, roi = args.get('scope', 'all'), args.get('roi')
        if name in ('region_statistics', 'report_generation'):
            output = region_statistics(mask, scope, roi)
            if name == 'report_generation':
                output = {'statistics': output, 'report': json.dumps({**current['provenance'], **output}, ensure_ascii=False, indent=2),
                          'format': 'application/json', 'source': 'deterministic_computation'}
        elif name == 'connected_region_analysis':
            output = connected_region_analysis(mask, scope, roi, args.get('min_area_pixels', 1))
        else:
            base = self.load(args['image_id'], args['base_result_id'], args['target'])
            base_mask, target_mask = apply_scope(base['mask'], scope, roi), apply_scope(mask, scope, roi)
            if name == 'spatial_relation_analysis':
                output = spatial_relation_analysis(base_mask, target_mask, args.get('near_distance_pixels', 10))
            else:
                output, diff = mask_compare(base_mask, target_mask)
                # Content addressed, new artifacts only; never overwrite source results.
                buf = io.BytesIO(); diff.save(buf, format='PNG'); raw = buf.getvalue()
                digest = hashlib.sha256(raw).hexdigest()
                folder = self.safe_path('national_artifacts'); folder.mkdir(exist_ok=True)
                file = folder/(digest+'.png')
                temporary = folder/(uuid.uuid4().hex+'.tmp')
                temporary.write_bytes(raw); temporary.replace(file)
                output.update(diff_mask_ref='/api/national/artifacts/'+digest+'.png', diff_sha256=digest,
                              base_result_id=args['base_result_id'], target_result_id=args['result_id'],
                              base_provenance=base['provenance'], scope=scope, roi=roi)
        return {**output, **current['provenance']}

    def save_execution(self, args, result):
        s = self.server
        session = s.get_session(args['image_id'])
        run = next((r for r in session['runs'] if r['id'] == args['result_id']), None)
        if not run:
            return
        execution = {'image_id': args['image_id'], 'result_id': args['result_id'],
                     'status': result['status'], 'failed_step': result['failed_step'],
                     'steps': [{**event, 'inputs': {k: v for k, v in args.items() if k != 'steps'},
                                'outputs': {'result_id': args['result_id'], **event.get('output', {})}}
                               for event in result['events']]}
        run['tool_execution'] = execution
        s.save_json(self.safe_path(args['image_id'], args['result_id'], 'result.json'), {k: v for k, v in run.items() if k != 'trace'})
        s.save_json(self.safe_path(args['image_id'], 'session.json'), s.public_session(session))

    def refine(self, args):
        from .registry import validate, REGISTRY
        validate(args, REGISTRY['adaptive_refinement']['input_schema'])
        self.load(args['image_id'], args['result_id'], args['target'])
        s = self.server
        if not callable(getattr(s, 'run_adaptive_pipeline', None)):
            raise ToolError('missing_dependency', 'Captain AFR adapter is unavailable')
        import os
        if not os.environ.get('GEO_REMOTESAM_URL') or not os.environ.get('GEO_AGENT_BASE_URL'):
            raise ToolError('service_unavailable', 'Real Agent and RemoteSAM must be configured for AFR')
        session = s.get_session(args['image_id'])
        parent = next(r for r in session['runs'] if r['id'] == args['result_id'])
        aliases = {'all': 'all', 'left_half': 'left', 'right_half': 'right', 'top_half': 'top', 'bottom_half': 'bottom', 'roi': 'all'}
        scope = args.get('scope')
        roi = ({'xyxy': args['roi'], 'source': 'drawn', 'image_size': [session['width'], session['height']]}
               if scope == 'roi' and 'roi' in args else parent.get('task_options', {}).get('roi') if scope is None else None)
        if scope is not None:
            from .statistics import scope_mask
            scope_mask((session['width'], session['height']), scope, args.get('roi'))
        result = s.run_task(session, {'parent_run_id': args['result_id'], 'query': '提取'+s.TARGETS[args['target']],
                            'mode': 'live', 'quality_mode': 'adaptive', 'force_perception': True,
                            'scope': aliases[scope] if scope else parent['task']['side'], 'roi': roi})
        if result.get('afr', {}).get('status') != 'completed' or not result.get('mask_url'):
            raise ToolError('service_unavailable', 'AFR did not complete; any fallback result remains in session history: '+result['id'])
        return {'image_id': args['image_id'], 'result_id': result['id'], 'parent_result_id': args['result_id'], 'run': result}
