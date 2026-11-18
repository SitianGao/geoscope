"""Manage only this deployment's Linux processes; never terminate by port/name."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request


class Runtime:
    def __init__(self, config):
        self.config = config
        self.root = Path(config['root']).resolve()
        self.output = self.root/'outputs'
        self.output.mkdir(exist_ok=True)
        self.python = str(self.root/'.venv/bin/python')
        self.agent_python = str(self.root/'.venv-agent/bin/python')
        self.children = {}

    def spec(self, name):
        c = self.config
        if name == 'sam':
            return ([c['sam_python'], '-u', 'remotesam.py'], self.root/'services/remotesam/upstream', 6657, 'remotesam.py')
        if name == 'agent':
            return ([self.agent_python, '-u', '-m', 'services.remoteagent.openai_service', '--model-path', c['agent_weights']], self.root, 8000, 'services.remoteagent.openai_service')
        if name == 'clip':
            return ([self.python, '-u', '-m', 'services.remoteclip.service', '--weights', c['clip_weights'],
                     '--sha256', c['clip_sha256'], '--architecture', c.get('clip_architecture', 'ViT-B-32'),
                     '--device', 'cuda:0', '--gallery', c.get('gallery', 'outputs/remoteclip-gallery.json')], self.root, 4190, 'services.remoteclip.service')
        return ([self.python, '-u', 'national_server.py'], self.root, 4180, 'national_server.py')

    def pid(self, name):
        try:
            pid = int((self.output/(name+'.pid')).read_text())
            if pid <= 1: raise ValueError('Invalid PID')
            return pid
        except FileNotFoundError:
            return None

    def owns(self, name, pid):
        if not pid: return False
        proc = Path('/proc')/str(pid)
        try:
            if proc.joinpath('stat').read_text().split(') ', 1)[1].startswith('Z '): return False
            argv = proc.joinpath('cmdline').read_bytes().decode().split('\0')
            command, cwd, _, marker = self.spec(name)
            return proc.joinpath('cwd').resolve() == cwd.resolve() and marker in argv and Path(argv[0]).resolve() == Path(command[0]).resolve()
        except (OSError, ValueError, IndexError):
            return False

    def environment(self):
        c = self.config
        return dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            GEO_AGENT_BASE_URL='http://127.0.0.1:8000/v1', GEO_AGENT_MODEL='RemoteAgent',
            GEO_REMOTESAM_URL='http://127.0.0.1:6657/predict', GEO_REMOTESAM_REVISION=c['sam_sha256'],
            GEO_REMOTECLIP_URL='http://127.0.0.1:4190', GEO_SERVICE_PROXY_MODE='direct',
            GEO_REMOTESAM_DEVICE='cuda:0', GEO_REMOTESAM_FP16='1',
            GEO_REMOTESAM_CHECKPOINT=c['sam_weights'], GEO_REMOTESAM_TOKENIZER=c['sam_tokenizer'])

    def start(self, name):
        pid = self.pid(name)
        if self.owns(name, pid): return {'component': name, 'pid': pid, 'state': 'already_running'}
        command, cwd, port, _ = self.spec(name)
        # An occupied port may belong to a teammate; never take it over.
        with socket.socket() as check:
            check.bind(('127.0.0.1', port))
        with (self.output/(name+'.log')).open('ab') as log:
            process = subprocess.Popen(command, cwd=cwd, env=self.environment(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        (self.output/(name+'.pid')).write_text(str(process.pid))
        self.children[name] = process
        return {'component': name, 'pid': process.pid, 'state': 'starting'}

    def stop(self, name):
        pid = self.pid(name)
        if not self.owns(name, pid): return {'component': name, 'state': 'not_running_or_not_owned'}
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic()+10
        while self.owns(name, pid) and time.monotonic() < deadline: time.sleep(.1)
        if self.owns(name, pid): os.kill(pid, signal.SIGKILL)
        if name in self.children: self.children.pop(name).wait(timeout=2)
        return {'component': name, 'state': 'stopped'}

    def status(self, name):
        pid = self.pid(name)
        result = {'component': name, 'pid': pid, 'owned_process': self.owns(name, pid), 'ready': False}
        path = '/api/national/retrieval/status' if name == 'workbench' else '/ready'
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open('http://127.0.0.1:'+str(self.spec(name)[2])+path, timeout=5) as response:
                data = json.loads(response.read())
            result['ready'] = data.get('ready') is True or data.get('status') in ('ready', 'completed')
            if name == 'clip': result['ready'] = result['ready'] and data.get('index_ready') is True
            result['error'] = data.get('error')
        except Exception as exc: result['error'] = type(exc).__name__
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'stop', 'status'])
    parser.add_argument('--config', default='outputs/server_runtime.json')
    args = parser.parse_args()
    if os.name != 'posix': parser.error('Run this manager on the Linux deployment server')
    runtime = Runtime(json.loads(Path(args.config).read_text()))
    names = ['sam', 'agent', 'clip', 'workbench']
    if args.action == 'stop': names.reverse()
    rows = [getattr(runtime, args.action)(name) for name in names]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if args.action == 'status' and not all(row['owned_process'] and row['ready'] for row in rows): raise SystemExit(1)


if __name__ == '__main__': main()
