"""Validate the complete plan before touching artifacts. Execute sequentially."""
import time
from .registry import SCOPES, validate
from .statistics import ToolError

STEPS = ['get_segmentation', 'apply_scope', 'region_statistics', 'connected_region_analysis',
         'compare_versions', 'report_generation']
PLAN_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['task', 'image_id', 'result_id', 'target', 'steps'],
    'properties': {'task': {'enum': ['region_comparison', 'region_analysis']},
        'image_id': {'type': 'string'}, 'result_id': {'type': 'string'}, 'target': {'type': 'string'},
        'scope': {'enum': SCOPES}, 'roi': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 4, 'maxItems': 4},
        'reference_version': {'type': 'string'},
        'steps': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': {'type': 'string', 'enum': STEPS}}}}


def validate_plan(plan):
    validate(plan, PLAN_SCHEMA, 'plan')
    have_mask = scoped = statistics = False
    for step in plan['steps']:
        if step == 'get_segmentation':
            if have_mask:
                raise ToolError('invalid_arguments', 'get_segmentation must occur once')
            have_mask = True
        elif not have_mask:
            raise ToolError('missing_dependency', f'{step} requires get_segmentation first')
        elif step == 'apply_scope':
            if scoped or statistics:
                raise ToolError('invalid_arguments', 'apply_scope must occur once before statistics')
            scoped = True
        elif step in ('region_statistics', 'connected_region_analysis', 'compare_versions'):
            if not scoped:
                raise ToolError('missing_dependency', f'{step} requires apply_scope first')
            if step == 'compare_versions' and not plan.get('reference_version'):
                raise ToolError('missing_dependency', 'compare_versions requires reference_version')
            statistics = statistics or step == 'region_statistics'
        elif step == 'report_generation' and not statistics:
            raise ToolError('missing_dependency', 'report_generation requires region_statistics')


def execute_plan(plan, backend):
    validate_plan(plan)
    events, artifacts = [], {}
    current = None
    for number, step in enumerate(plan['steps'], 1):
        start = time.perf_counter()
        try:
            if step == 'get_segmentation':
                current = backend.load(plan['image_id'], plan['result_id'], plan['target'])
                output = current['provenance']
            elif step == 'apply_scope':
                from .statistics import apply_scope
                current = {**current, 'mask': apply_scope(current['mask'], plan.get('scope', 'all'), plan.get('roi'))}
                output = {'scope': plan.get('scope', 'all'), 'roi': plan.get('roi')}
            else:
                name = 'mask_compare' if step == 'compare_versions' else step
                arguments = {k: plan[k] for k in ('image_id', 'result_id', 'target', 'scope', 'roi') if k in plan}
                if step == 'compare_versions':
                    arguments['base_result_id'] = plan['reference_version']
                output = backend.execute(name, arguments, loaded=current)
            artifacts[step] = output
            events.append({'step': number, 'tool': step, 'status': 'completed', 'duration_ms': round((time.perf_counter()-start)*1000, 3), 'output': output})
        except Exception as exc:
            code = exc.code if isinstance(exc, ToolError) else 'service_unavailable'
            events.append({'step': number, 'tool': step, 'status': 'failed', 'error': {'code': code, 'message': str(exc) if isinstance(exc, ToolError) else 'Tool execution failed'}})
            return {'status': 'failed', 'failed_step': number, 'events': events, 'artifacts': artifacts,
                    'error': events[-1]['error']}
    return {'status': 'completed', 'failed_step': None, 'events': events, 'artifacts': artifacts, 'error': None}
