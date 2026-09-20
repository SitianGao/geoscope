"""Check the imported Agent contract without loading weights or using a GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(ROOT/'services/remoteagent/upstream'))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source).resolve()))
    report = {'inference_verified': False}
    try:
        from remoteagent import RemoteAgent
        report['import_ok'] = True
        report['missing_methods'] = [name for name in ('plan','continue_with_tool_result') if not callable(getattr(RemoteAgent,name,None))]
        report['contract_ready'] = not report['missing_methods']
        report['message'] = '接口形状检查通过，真实模型与工作台接入仍待验证' if report['contract_ready'] else '当前源码未提供全部 Harness 接口，请 A 补齐并提交'
    except Exception as exc:
        report.update(import_ok=False, contract_ready=False, error_type=type(exc).__name__, message=str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['contract_ready'] else 2)
