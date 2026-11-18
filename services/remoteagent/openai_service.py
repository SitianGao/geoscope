"""Optional OpenAI-compatible serving of the existing official Agent checkpoint.

Uses Transformers when the previous vLLM environment is unavailable. This is
an inference runtime adapter, not a replacement planner or a new model.
"""
import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    state = {'ready': False, 'error': 'loading', 'runtime': 'transformers', 'model': 'RemoteAgent'}
    lock = threading.Lock()
    processor = model = None

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, status=200):
            raw = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_GET(self):
            if self.path == '/ready': return self.reply(state)
            if self.path == '/v1/models':
                return self.reply({'data': [{'id': 'RemoteAgent', 'object': 'model'}] if state['ready'] else [], **state})
            return self.reply({'error': 'not found'}, 404)

        def do_POST(self):
            if self.path != '/v1/chat/completions': return self.reply({'error': 'not found'}, 404)
            try:
                if not state['ready']: return self.reply({'error': state['error']}, 503)
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 24*1024*1024: raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                if payload.get('model') != 'RemoteAgent' or payload.get('stream'):
                    raise ValueError('Only RemoteAgent non-streaming requests are supported')
                limit = payload.get('max_tokens', 512)
                if type(limit) is not int or not 1 <= limit <= 2048: raise ValueError('Invalid max_tokens')
                messages, images = [], []
                for message in payload['messages']:
                    if message['role'] not in ('system', 'user', 'assistant'): raise ValueError('Invalid role')
                    content = message['content']
                    if isinstance(content, str): content = [{'type': 'text', 'text': content}]
                    parts = []
                    for part in content:
                        if part['type'] == 'text': parts.append({'type': 'text', 'text': part['text']})
                        elif part['type'] == 'image_url':
                            url = part['image_url']['url']
                            if not url.startswith('data:image/') or ';base64,' not in url: raise ValueError('Only inline images are supported')
                            image = Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1], validate=True))).convert('RGB')
                            if image.width*image.height > 16_000_000: raise ValueError('Image too large')
                            images.append(image); parts.append({'type': 'image', 'image': image})
                        else: raise ValueError('Unsupported content')
                    messages.append({'role': message['role'], 'content': parts})
                with lock, torch.inference_mode():
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=images or None, padding=True, return_tensors='pt').to('cuda:0')
                    tokens = inputs['input_ids'].shape[1]
                    if tokens+limit > 16384: raise ValueError(f'Context limit exceeded ({tokens}+{limit}>16384)')
                    output = model.generate(**inputs, max_new_tokens=limit, do_sample=False)
                    answer = processor.batch_decode(output[:, tokens:], skip_special_tokens=True)[0]
                return self.reply({'id': 'remoteagent-'+str(time.time_ns()), 'model': 'RemoteAgent',
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': answer}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': tokens, 'completion_tokens': output.shape[1]-tokens}})
            except Exception as exc:
                print(json.dumps({'event': 'inference_failed', 'type': type(exc).__name__, 'message': str(exc)}), flush=True)
                return self.reply({'error': {'message': str(exc), 'type': type(exc).__name__}}, 400 if isinstance(exc, ValueError) else 503)

    def load():
        nonlocal processor, model
        try:
            processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, use_fast=False, max_pixels=1024*28*28)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda:0').eval()
            state.update(ready=True, error=None)
            print(json.dumps({'event': 'model_loaded', **state}), flush=True)
        except Exception as exc:
            state.update(error=type(exc).__name__+': '+str(exc))
            print(json.dumps(state), flush=True)
    threading.Thread(target=load, daemon=True).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__': main()
