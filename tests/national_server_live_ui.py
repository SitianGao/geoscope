"""Read genuine persisted model results in B's UI; no inference/test doubles."""
import io
import json
import urllib.request
import zipfile
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    evidence = Path('docs/evidence/server')
    data = json.loads((evidence/'full_chain_p1529.json').read_text())
    assert data['status'] == 'completed'
    base, refined = data['baseline'], data['adaptive']['data']['run']
    sid = base['session_id']
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
        try:
            page = browser.new_page(viewport={'width': 1512, 'height': 1040})
            errors = []; page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto('http://127.0.0.1:4180/?workbench=national')
            page.wait_for_function("typeof state !== 'undefined' && state.session && state.original && !document.querySelector('#modal').open")
            page.evaluate('async id => {await openExperiment(id)}', sid)
            page.locator('[data-run="'+refined['id']+'"]').click()
            page.locator('.execution-panel').evaluate('el => el.open=true')
            page.wait_for_function("document.querySelectorAll('#nationalTrace .national-state').length === 5")
            page.screenshot(path=str(evidence/'real_afr_steps.png'))
            page.locator('#nationalCompare').click()
            page.locator('#nationalBase').select_option(base['id'])
            page.locator('#nationalTarget').select_option(refined['id'])
            page.locator('#nationalVerifyDiff').click()
            page.wait_for_function("document.querySelector('#nationalVerifyStatus').textContent.includes('后端统计与浏览器逐像素结果一致')")
            page.screenshot(path=str(evidence/'real_version_compare.png'))
            page.locator('#closeModal').click()
            page.locator('#nationalDiagnosis').click()
            page.locator('#diagnosticBody').wait_for(state='visible')
            page.screenshot(path=str(evidence/'real_disagreement.png'))
            assert not errors, errors
            with urllib.request.urlopen('http://127.0.0.1:4180'+refined['export_url']) as response:
                with zipfile.ZipFile(io.BytesIO(response.read())) as archive: members = archive.namelist()
            assert {'afr.json', 'probability_map.png', 'disagreement.png'} <= set(members)
            (evidence/'real_ui.json').write_text(json.dumps({'image_id': sid, 'base_result_id': base['id'],
                'target_result_id': refined['id'], 'actual_model_results': True, 'browser_compare_matches_backend': True,
                'export_members': members, 'page_errors': errors}, indent=2))
            print('Real AFR version/steps/difference/diagnostics/export UI: passed')
        finally: browser.close()


if __name__ == '__main__': main()
