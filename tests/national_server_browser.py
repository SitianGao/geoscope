"""Server-side real CLIP UI acceptance. No response interception or fake scores."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    evidence = Path('docs/evidence/server'); evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
        try:
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            errors = []; page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto('http://127.0.0.1:4180/?workbench=national')
            page.wait_for_function("typeof state !== 'undefined' && state.session && state.original && !document.querySelector('#modal').open")
            page.locator('#nationalRetrieval').click()
            page.wait_for_function("document.querySelector('#nationalSearchStatus').textContent.includes('检索就绪')")
            page.locator('#nationalSearchQuery').fill('airport with airplanes')
            page.screenshot(path=str(evidence/'before_search.png'))
            print('search state', page.evaluate("({modal:document.querySelector('#modal').open,button:document.querySelector('#nationalSearchSubmit').getBoundingClientRect().toJSON()})"), flush=True)
            with page.expect_response(lambda r: r.url.endswith('/retrieval/search')) as response:
                page.locator('#nationalSearchQuery').press('Enter')
            result = response.value.json()
            assert result['success'] and result['data']['real_execution'] is True
            page.locator('#nationalSearchHits article').first.wait_for()
            page.screenshot(path=str(evidence/'clip_gpu_workbench.png'))
            page.get_by_role('button', name='分析该影像').first.click()
            selected = result['data']['results'][0]['image_id']
            page.wait_for_function('id => state.session?.id === id && state.original', arg=selected)
            assert not errors, errors
            (evidence/'browser.json').write_text(json.dumps({'location': 'laboratory server', 'real_clip': result,
                'selected_image_id': selected, 'page_errors': errors,
                'segmentation_executed': False}, ensure_ascii=False, indent=2), encoding='utf-8')
            print('SERVER real GPU CLIP -> B workbench candidate -> existing image selection: passed')
        finally:
            browser.close()


if __name__ == '__main__': main()
