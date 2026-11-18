"""Bind built-in sample images to platform sessions, then emit gallery IDs."""
import argparse
import json
from pathlib import Path
import server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/remoteclip-gallery.json')
    args = parser.parse_args()
    sessions = [server.new_session(sample) for sample in server.SAMPLES]
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([s['id'] for s in sessions], indent=2), encoding='utf-8')
    print(json.dumps({'gallery': str(path), 'images': [{'image_id': s['id'], 'name': s['name']} for s in sessions]}, ensure_ascii=False))


if __name__ == '__main__': main()
