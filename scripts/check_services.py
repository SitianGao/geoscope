"""Run on the WORKBENCH machine: python scripts/check_services.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_config import load_settings
from service_transport import inspect_services

if __name__ == '__main__':
    load_settings()
    report = inspect_services()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['services_ready'] else 2)
