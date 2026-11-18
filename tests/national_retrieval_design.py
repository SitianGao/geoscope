"""Real server UI: compact results, traceability, responsive layout and selection."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    output = Path('docs/evidence/server'); records = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
        try:
            for width, height, name in [(1733, 897, 'desktop'), (390, 844, 'mobile')]:
                page = browser.new_page(viewport={'width': width, 'height': height})
                errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto('http://127.0.0.1:4180/?workbench=national')
                page.wait_for_function("typeof state !== 'undefined' && state.session && state.original && !document.querySelector('#modal').open")
                page.locator('#nationalRetrieval').click()
                page.wait_for_function("document.querySelector('#nationalSearchStatus').textContent.includes('检索就绪')")
                options = page.locator('#nationalSearchTopK option').evaluate_all('els => els.map(el => Number(el.value))')
                assert options and max(options) <= 3, options
                page.locator('#nationalSearchQuery').fill('airport with airplanes')
                with page.expect_response(lambda response: response.url.endswith('/retrieval/search')) as response:
                    page.locator('#nationalSearchSubmit').click()
                result = response.value.json()['data']
                page.locator('#nationalSearchHits article').first.wait_for()
                page.wait_for_function("""() => [...document.querySelectorAll('#nationalSearchHits img')]
                    .filter(img => {const r=img.getBoundingClientRect(); return r.top < innerHeight && r.bottom > 0})
                    .every(img => img.complete && img.naturalWidth > 0)""")
                scores = page.locator('.retrieval-score strong').all_text_contents()
                assert scores == [f"{hit['similarity']:.3f}" for hit in result['results']]
                assert page.locator('#nationalRetrievalDetails').evaluate('el => el.open') is False
                assert 'dfe52' not in page.locator('#nationalSearchStatus').inner_text()
                layout = page.evaluate("""() => ({pageOverflow:document.documentElement.scrollWidth-innerWidth,
                    dialogOverflow:document.querySelector('#modal').scrollWidth-document.querySelector('#modal').clientWidth,
                    contentOverflow:document.querySelector('#modalBody').scrollHeight-document.querySelector('#modalBody').clientHeight,
                    buttons:[...document.querySelectorAll('#nationalSearchHits article button')].map(el=>el.getBoundingClientRect().bottom)})""")
                assert layout['pageOverflow'] <= 1 and layout['dialogOverflow'] <= 1, layout
                if name == 'desktop':
                    assert max(layout['buttons'])-min(layout['buttons']) <= 1, layout
                    assert layout['contentOverflow'] <= 2, layout
                page.screenshot(path=str(output/('retrieval_redesign_'+name+'.png')))
                page.locator('#nationalRetrievalDetails summary').click()
                assert result['index_version'] in page.locator('#nationalRetrievalProvenance').inner_text()
                assert str(result['results'][0]['similarity']) in page.locator('#nationalRetrievalProvenance').inner_text()
                page.locator('#nationalRetrievalDetails summary').click()
                page.get_by_role('button', name='分析该影像').first.click()
                selected = result['results'][0]['image_id']
                page.wait_for_function('id => state.session?.id === id', arg=selected)
                assert not errors, errors
                records.append({'viewport': name, 'options': options, 'scores': scores, 'layout': layout,
                                'selected_image_id': selected, 'real_execution': True, 'errors': errors})
                page.close()
            page = browser.new_page(viewport={'width': 1733, 'height': 1000})
            page.goto('http://127.0.0.1:4180/?workbench=national')
            page.wait_for_function("typeof state !== 'undefined' && state.session && state.original && !document.querySelector('#modal').open")
            page.evaluate("async () => { await openExperiment('7c8d57152d4b') }")
            page.locator('[data-run="628d33918a0e"]').click()
            page.wait_for_function("document.querySelector('#resultStatus').textContent.includes('旧版结果需重跑')")
            assert page.locator('#reviewApprove').is_disabled()
            page.locator('[data-run="e8e1eb4a6be7"]').click()
            page.wait_for_function("document.querySelector('#resultTitle').textContent.includes('最下方的船舶')")
            assert '全图' not in page.locator('#resultTitle').inner_text()
            assert page.locator('#scopeValue').inner_text() == '单目标指代'
            page.screenshot(path=str(output/'ship_reference_workbench.png'))
        finally: browser.close()
    (output/'retrieval_redesign_checks.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Real retrieval layout/precision/traceability/selection + ship reference labels: passed')


if __name__ == '__main__': main()
