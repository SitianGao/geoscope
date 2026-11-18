"""Actual HTTP chain: CLIP -> Agent/SAM -> AFR -> tools -> export. No mocks."""
import io
import argparse
import json
import urllib.request
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-id', help='Explicit selection among actual retrieval candidates')
    parser.add_argument('--output-name', default='full_chain.json')
    args = parser.parse_args()
    base = 'http://127.0.0.1:4180'
    output = Path('docs/evidence/server'); output.mkdir(parents=True, exist_ok=True)
    evidence = {'execution_source': 'laboratory server; real checkpoints; no mock responses'}
    def request(path, data=None):
        req = urllib.request.Request(base+path, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=300) as response: return json.loads(response.read())
    def save(): (output/args.output_name).write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        evidence['retrieval'] = request('/api/national/retrieval/search', {'query': 'airport with airplanes', 'top_k': 5})
        candidates = evidence['retrieval']['data']['results']
        sid = args.image_id or candidates[0]['image_id']
        assert any(item['image_id'] == sid for item in candidates), 'Selected image is not in actual retrieval results'
        evidence['baseline'] = request('/api/run', {'session_id': sid, 'query': '提取全图飞机并计算面积占比', 'mode': 'live', 'quality_mode': 'fast', 'force_perception': True})
        save()
        run = evidence['baseline']
        assert run['mode'] == 'live' and run['status'] in ('completed', 'needs_review') and run.get('mask_url'), run.get('message')
        evidence['adaptive'] = request('/api/national/tools/adaptive-refinement', {'image_id': sid, 'result_id': run['id'], 'target': 'aircraft', 'scope': 'all'})
        save()
        refined = evidence['adaptive']['data']['run']
        assert refined['afr']['status'] == 'completed'
        evidence['plan'] = request('/api/national/plans/execute', {'task': 'region_comparison', 'image_id': sid,
            'result_id': refined['id'], 'target': 'aircraft', 'scope': 'left_half', 'reference_version': run['id'],
            'steps': ['get_segmentation', 'apply_scope', 'region_statistics', 'compare_versions', 'report_generation']})
        assert evidence['plan']['success']
        evidence['comparison'] = request('/api/national/tools/mask-compare', {'image_id': sid, 'base_result_id': run['id'], 'target_result_id': refined['id']})
        with urllib.request.urlopen(base+refined['export_url']) as response:
            archive = response.read()
        with zipfile.ZipFile(io.BytesIO(archive)) as package:
            evidence['export_members'] = package.namelist()
        evidence['afr_artifacts'] = {}
        for key in ('disagreement_ref', 'probability_map_ref'):
            if refined['afr'].get(key):
                with urllib.request.urlopen(base+refined['afr'][key]) as response:
                    evidence['afr_artifacts'][key] = {'http_status': response.status, 'bytes': len(response.read())}
        evidence['status'] = 'completed'; save()
        print(json.dumps({'status': 'completed', 'image_id': sid, 'base': run['id'], 'afr': refined['id'], 'evidence': str(output/args.output_name)}, ensure_ascii=False))
    except Exception as exc:
        evidence['status'] = 'failed'; evidence['failure'] = str(exc); save(); raise


if __name__ == '__main__': main()
