"""Use the delivered RemoteAgent planner; the workbench alone executes tools."""
from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import hashlib

from service_transport import request_json

UPSTREAM = Path(__file__).resolve().parent / 'services/remoteagent/upstream'
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
from remoteagent import RemoteAgent

ALLOWED_TOOLS = {'referring_expression_segmentation', 'semantic_segmentation'}
LABELS = {
    'building': ('building', 'buildings', '建筑', '房屋'),
    'aircraft': ('aircraft', 'airplane', 'airplanes', 'plane', 'planes', '飞机'),
    'road': ('road', 'roads', '道路'),
    'water': ('water', 'water body', 'water bodies', '水体'),
    'tree': ('tree', 'trees', 'vegetation', 'trees and vegetation', '植被', '树木'),
    'ship': ('ship', 'ships', '船舶'),
}
PROMPTS = {'building':'all buildings', 'aircraft':'all airplanes', 'road':'all roads',
           'water':'water bodies', 'tree':'trees and vegetation', 'ship':'all ships'}
CLASSES = {'building':'building', 'aircraft':'aircraft', 'road':'road',
           'water':'water', 'tree':'tree', 'ship':'ship'}
SIDES = {'all', 'left', 'right', 'top', 'bottom'}


class Clarification(ValueError):
    """A task outside the explicit workbench contract, not a service failure."""


def resolve_scope(query, context, selected=None):
    """Explicit, logged pixel-scope grammar; not an LLM reasoning capability."""
    q = query.lower()
    if re.search(r'靠近|附近|之间|东侧|西侧|南侧|北侧|平方米|公顷|米以内|最大|最小|除外|除了|不要.*[左右上下]|不含|左上|左下|右上|右下|\b(near|between|except|largest|smallest|north|south|east|west)\b', q):
        raise Clarification('当前只支持全图或单个像素半幅。复杂关系、排除条件和地理面积需要另行明确，尚未执行。')
    found = {side for side, pattern in {
        'left': r'左|\bleft\b', 'right': r'右|\bright\b',
        'top': r'上半|上方|顶部|\b(top|upper)\b',
        'bottom': r'下半|下方|底部|\b(bottom|lower)\b',
        'all': r'全图|全部|整幅|\bwhole image\b|\bentire image\b',
    }.items() if re.search(pattern, q)}
    if len(found) > 1:
        raise Clarification('一次实验请选择一个范围：全图、左半幅、右半幅、上半幅或下半幅。')
    if selected is not None:
        if selected not in SIDES:
            raise ValueError('不支持的像素范围')
        if found and selected not in found:
            raise Clarification('所选范围与文字指令不一致，请确认后重试。')
        return selected, 'explicit_scope_selection'
    if found:
        return found.pop(), 'harness_pixel_scope_rule'
    return context.get('side', 'all'), 'selected_version_context' if context.get('side') else 'whole_image_default'


class WorkbenchAgent(RemoteAgent):
    """Keep upstream planning/parsing/feedback; use our authenticated HTTP transport."""
    def __init__(self, task_route='dense'):
        self.base = os.environ.get('GEO_AGENT_BASE_URL', '').rstrip('/')
        model = os.environ.get('GEO_AGENT_MODEL', '')
        if not self.base or not model:
            raise ValueError('RemoteAgent 服务未配置，请填写实际服务地址和模型名称。')
        if task_route not in ('dense', 'internal'):
            raise ValueError('未知的 RemoteAgent 任务路由')
        self.task_route = task_route
        allowed_tools = ALLOWED_TOOLS if task_route == 'dense' else set()
        super().__init__(vllm_url=self.base, model_name=model, allowed_tools=allowed_tools,
                         api_urls={}, max_tokens=1024)

    def _runtime_system_prompt(self):
        base = super()._runtime_system_prompt()
        if self.task_route == 'internal':
            return base + (
                '\nGeoScope turn route: INTERNAL VISUAL UNDERSTANDING ONLY. No external tool is available '
                'or needed for this turn. Inspect the supplied image yourself and answer the user directly '
                'inside exactly one <answer>...</answer> block. Scene description, image captioning, visual '
                'question answering, classification, counting, comparison, and asking what the image shows '
                'are internal tasks even when the image contains several object categories. Do not output '
                'T_call, do not request segmentation, and do not ask the user to choose one target category. '
                'Answer in concise Chinese unless the user requested another language.'
            )
        return base + (
            '\nGeoScope workbench contract: preserve <answer> and T_call syntax. '
            'For extraction, request exactly one supported category: building, aircraft, road, water, tree, ship. '
            'Use a whole-image category prompt (e.g. all buildings), or semantic_segmentation with one class. '
            'The Harness applies the explicit pixel scope in context AFTER whole-image segmentation. '
            'Mandatory extraction output rule: emit exactly one allowed T_call and no second tool call. '
            'Never call contour_extraction, region_contour_extraction, or any other spatial tool, even when '
            'the user asks for left, right, top, bottom, area, or ratio; those operations belong to the Harness. '
            'If context.invert is true because the user asks for non-X or everything except X, call segmentation '
            'for the positive base category X only; the Harness deterministically inverts the returned mask. '
            'Do not use an XML <T_call> wrapper. A plain T_call(...) or <answer>T_call(...)</answer> is accepted. '
            'Never invent new spatial conditions. For unsupported targets or ambiguous requests, '
            'ask for clarification inside <answer>. The selected version supplies previous target and scope. '
            'Area and ratio must come from the Harness execution result. Never claim a mask was generated before execution.'
        )

    def _run_llm(self, messages):
        response = request_json(self.base + '/chat/completions', {
            'model': self.model_name, 'messages': messages, 'max_tokens': self.max_tokens,
            'temperature': 0,
        }, headers={'Authorization': 'Bearer ' + os.environ.get('GEO_AGENT_API_KEY', 'EMPTY')})
        try:
            message = response['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            raise ValueError('认知服务未返回有效的文本响应') from None
        if not isinstance(message, str) or not message.strip():
            raise ValueError('认知服务返回空响应')
        return message.strip()


def decision_record(decision):
    raw = decision.raw_response or ''
    visible = re.sub(r'<think>[\s\S]*?</think>', '<think>[推理过程不写入实验记录]</think>', raw,
                     flags=re.IGNORECASE).strip()
    history = []
    for item in decision.history:
        record = dict(item)
        if isinstance(record.get('response'), str):
            record['response'] = re.sub(r'<think>[\s\S]*?</think>',
                                        '<think>[推理过程不写入实验记录]</think>',
                                        record['response'], flags=re.IGNORECASE).strip()
        history.append(record)
    return {'status': decision.status, 'raw_response': visible,
            'raw_response_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest() if raw else None,
            'text': decision.text, 'history': history,
            'tool_call': None if decision.tool_call is None else {
                'name': decision.tool_call.name, 'arguments': decision.tool_call.arguments}}


def target_from_label(label, side):
    if not isinstance(label, str):
        raise Clarification('模型没有提供有效的目标类别。')
    label = label.strip().lower()
    # Only this exact simple suffix may be normalized; arbitrary relations are rejected.
    match = re.fullmatch(r'(.+?) (?:on|in) the (left|right|top|bottom)(?: (?:half|side))?', label)
    if match:
        if match.group(2) != side:
            raise Clarification('模型返回的空间范围与本次任务范围不一致，请确认后重试。')
        label = match.group(1)
    label = re.sub(r'^(?:all (?:the )?|the )', '', label)
    for target, aliases in LABELS.items():
        if label in aliases:
            return target
    raise Clarification('本版仅支持六类目标的全图分割及像素半幅筛选；模型给出了其他类别或额外条件，请简化指令。')


def segmentation_plan(decision, image_path, side, scope_source, invert=False):
    call = decision.tool_call
    if decision.status != 'tool_call' or call is None or call.name not in ALLOWED_TOOLS:
        raise ValueError('认知核心没有返回允许执行的分割工具')
    args = call.arguments
    if Path(args.get('image_path', '')).resolve() != Path(image_path).resolve():
        raise ValueError('工具影像与当前实验不一致')
    if call.name == 'referring_expression_segmentation':
        if set(args) != {'image_path', 'prompt'}:
            raise ValueError('指代分割参数不符合约定')
        target = target_from_label(args['prompt'], side)
        request = {'task': 'referring_seg', 'text': PROMPTS[target]}
        mask_field = None
    else:
        classes = args.get('classes')
        if set(args) != {'image_path', 'classes'} or not isinstance(classes, list) or len(classes) != 1:
            raise Clarification('每次实验只支持一个目标类别，请分别提取并保存版本。')
        target = target_from_label(classes[0], side)
        request = {'task': 'semantic_seg', 'classes': [CLASSES[target]]}
        mask_field = CLASSES[target]
    return {'action': 'segment', 'target': target, 'side': side, 'invert': bool(invert), 'scope_source': scope_source,
            'tool_name': call.name, 'service_request': request, 'mask_field': mask_field,
            'planner_protocol': 'remoteagent', 'scope_rule': '整图目标分割后按像素半幅裁切；由 Harness 执行'}


class AgentTurn:
    def __init__(self, query, image_path, context, side, scope_source, task_route='dense', invert=False):
        self.agent = WorkbenchAgent(task_route=task_route)
        self.query, self.image_path = query, image_path
        self.context = {**context, 'side': side, 'scope_source': scope_source,
                        'scope_rule': 'whole-image segmentation then deterministic pixel-half clipping',
                        'task_route': task_route, 'invert': bool(invert)}
        self.decision = self.agent.plan(query, image_path, context=self.context)

    def feedback(self, result):
        return self.agent.continue_with_tool_result(self.query, self.image_path, self.decision,
                                                   result, context=self.context)
