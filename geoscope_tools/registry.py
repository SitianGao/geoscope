"""Strict whitelist. No user supplied code, paths or dynamic imports."""
from .statistics import ToolError

SCOPES = ['all', 'left_half', 'right_half', 'top_half', 'bottom_half', 'roi']
COMMON = {'image_id': {'type': 'string'}, 'result_id': {'type': 'string'},
          'target': {'type': 'string'}, 'scope': {'enum': SCOPES},
          'roi': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 4, 'maxItems': 4}}


def schema(extra=None, required=None):
    return {'type': 'object', 'additionalProperties': False,
            'properties': {**COMMON, **(extra or {})},
            'required': required or ['image_id', 'result_id', 'target']}


REGISTRY = {
    'region_statistics': {'available': True, 'input_schema': schema()},
    'connected_region_analysis': {'available': True, 'input_schema': schema({'min_area_pixels': {'type': 'integer', 'minimum': 1}})},
    'mask_compare': {'available': True, 'input_schema': schema({'base_result_id': {'type': 'string'}} , ['image_id', 'result_id', 'base_result_id', 'target'])},
    'spatial_relation_analysis': {'available': True, 'input_schema': schema({'base_result_id': {'type': 'string'}, 'near_distance_pixels': {'type': 'number', 'minimum': 0, 'maximum': 100000}}, ['image_id', 'result_id', 'base_result_id', 'target'])},
    'report_generation': {'available': True, 'input_schema': schema()},
    'adaptive_refinement': {'available': False, 'reason': 'Captain AFR adapter not supplied', 'input_schema': schema()},
    'geo_measurement': {'available': False, 'reason': 'Georeferencing and unit validation not supplied', 'input_schema': schema()},
    'semantic_segmentation': {'available': False, 'legacy_endpoint': '/api/run', 'reason': 'Use existing validated Agent/Harness route; not directly executed by this registry', 'input_schema': schema()},
    'referring_expression_segmentation': {'available': False, 'legacy_endpoint': '/api/run', 'reason': 'Use existing validated Agent/Harness route; not directly executed by this registry', 'input_schema': schema()},
}


def validate(value, rule, name='arguments'):
    kind = rule.get('type')
    valid = {'object': lambda: type(value) is dict, 'string': lambda: type(value) is str and bool(value.strip()),
             'integer': lambda: type(value) is int, 'number': lambda: type(value) in (int, float),
             'array': lambda: type(value) is list}
    if kind and not valid[kind]():
        raise ToolError('invalid_arguments', f'{name}: expected {kind}')
    if 'enum' in rule and value not in rule['enum']:
        raise ToolError('invalid_arguments', f'{name}: unsupported value')
    if kind == 'object':
        properties = rule['properties']
        if set(rule.get('required', [])) - value.keys() or value.keys() - properties.keys():
            raise ToolError('invalid_arguments', f'{name}: missing required fields or unknown fields')
        for key, item in value.items():
            validate(item, properties[key], name+'.'+key)
    if kind == 'array':
        if not rule.get('minItems', 0) <= len(value) <= rule.get('maxItems', 100):
            raise ToolError('invalid_arguments', f'{name}: invalid length')
        for item in value:
            validate(item, rule['items'], name)
    if kind in ('integer', 'number'):
        import math
        if not math.isfinite(value) or not rule.get('minimum', -float('inf')) <= value <= rule.get('maximum', float('inf')):
            raise ToolError('invalid_arguments', f'{name}: out of range')


def validate_call(name, arguments):
    if name not in REGISTRY:
        raise ToolError('unsupported_tool', 'Tool is not in the whitelist')
    validate(arguments, REGISTRY[name]['input_schema'])
    if not REGISTRY[name]['available']:
        raise ToolError('missing_dependency', REGISTRY[name]['reason'])
