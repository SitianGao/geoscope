"""Read-only Linux/Windows resource check. No model load, installations or global edits."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request


def probe(url, direct):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
    try:
        with opener.open(url, timeout=10) as response:
            # Only metadata/configuration endpoints are probed, never model weights.
            return {'status': response.status, 'reachable': True}
    except urllib.error.HTTPError as exc:
        return {'status': exc.code, 'reachable': True, 'file_access_confirmed': False}
    except Exception as exc:
        return {'reachable': False, 'error_type': type(exc).__name__, 'message': '连接失败；检查代理、网络和证书。'}


def collect(model_dir=None, network=False):
    report = {'host': socket.gethostname(), 'python': sys.version.split()[0], 'python_path': sys.executable,
              'note': '只读检查；GPU空闲不等于已获准使用；文件存在不等于模型推理已验证。'}
    try:
        report['home_free_gib'] = round(shutil.disk_usage(Path.home()).free/(1024**3), 1)
    except OSError:
        report['home_free_gib'] = None
        report['disk_note'] = '当前权限无法读取主目录剩余空间，请在服务器终端核对。'
    report['proxies'] = {}
    for kind, value in urllib.request.getproxies().items():
        if kind not in ('http','https','all'): continue
        p = urllib.parse.urlsplit(value if '://' in value else 'http://'+value)
        report['proxies'][kind] = {'host': p.hostname, 'port': p.port}
    try:
        gpu = subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        report['gpu'] = gpu.stdout.strip() if gpu.returncode == 0 else 'nvidia-smi执行失败'
    except (OSError,subprocess.TimeoutExpired): report['gpu'] = '未找到或无法读取nvidia-smi'
    if model_dir:
        folder = Path(model_dir).expanduser()
        result = {'directory': str(folder.resolve()), 'exists': folder.is_dir(), 'config_exists': (folder/'config.json').is_file()}
        index = folder/'model.safetensors.index.json'
        if index.is_file():
            try:
                shards = set(json.loads(index.read_text())['weight_map'].values())
                for shard in shards:
                    if not isinstance(shard, str):
                        raise ValueError('Invalid shard path')
                    # relative_to also works with the Python 3.8 shipped on Ubuntu 20.04.
                    (folder / shard).resolve().relative_to(folder.resolve())
                result['missing_or_empty_shards'] = sorted(p for p in shards if not (folder/p).is_file() or (folder/p).stat().st_size == 0)
                result['shard_count'] = len(shards)
                result['note'] = '仅检查存在与非空，未校验完整哈希。'
            except Exception: result['index_error'] = '无法解析权重索引'
        else: result['index_exists'] = False
        report['model_files'] = result
    if network:
        urls = {'modelscope':'https://modelscope.cn', 'model_config':'https://modelscope.cn/api/v1/models/AIMGroup/RemoteAgent/repo?Revision=master&FilePath=config.json'}
        report['network'] = {name:{'environment':probe(url,False),'direct':probe(url,True)} for name,url in urls.items()}
        report['network_note'] = '404本身不能确定是权限、模型文件状态还是地址问题。'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir'); parser.add_argument('--network',action='store_true')
    args = parser.parse_args()
    print(json.dumps(collect(args.model_dir,args.network),ensure_ascii=False,indent=2))
