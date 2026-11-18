const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');

const base = process.env.GEO_TEST_BASE_URL || 'http://127.0.0.1:4180';
const evidence = path.resolve(process.env.GEO_NATIONAL_EVIDENCE || 'test-results/national');
const recording = process.env.GEO_NATIONAL_RECORD === '1';
const pauseMs = Number(process.env.GEO_NATIONAL_PACE_MS || 5000);
const screenshot = (page, name) => page.screenshot({ path: path.join(evidence, name) });
const matches = (page, selector, text) => page.waitForFunction(
    ([selector, text]) => document.querySelector(selector)?.textContent.includes(text),
    [selector, text], { timeout: 12000 }
);
const bodyJson = (route, data) => route.fulfill({ json: data });

async function task(page, query) {
    await page.locator('#query').fill(query);
    const response = page.waitForResponse(r => r.url().endsWith('/api/run'));
    await page.locator('#sendBtn').click();
    const result = await (await response).json();
    await page.locator('#sendBtn:not([disabled])').waitFor();
    return result;
}

async function openSample(page, sample) {
    await page.locator('#galleryBtn').click();
    await page.locator('[data-sample="' + sample + '"]').click();
    await page.locator('#modal').waitFor({ state: 'hidden' });
}

async function drawRoi(page, bounds = [.18, .51, .49, .81]) {
    await page.locator('#drawRoiBtn').click();
    const rect = await page.locator('#imageCanvas').boundingBox();
    const [x1, y1, x2, y2] = bounds;
    await page.mouse.move(rect.x + rect.width * x1, rect.y + rect.height * y1);
    await page.mouse.down();
    await page.mouse.move(rect.x + rect.width * x2, rect.y + rect.height * y2, { steps: 12 });
    await page.mouse.up();
    await matches(page, '#roiLabel', '研究区：');
}

async function verifyWorkflow(browser) {
    const context = await browser.newContext({
        viewport: { width: 1512, height: 1040 },
        ...(recording ? { recordVideo: { dir: evidence, size: { width: 1512, height: 1040 } } } : {})
    });
    const page = await context.newPage();
    const errors = [];
    const pace = () => recording ? page.waitForTimeout(pauseMs) : Promise.resolve();
    const runRecords = [];
    page.on('pageerror', error => errors.push(error.message));
    try {
        await page.goto(base + '/?workbench=national');
        await matches(page, '#qualityMode', '自动（按建筑选择）');
        await openSample(page, 'airport');
        await matches(page, '#imageName', 'JFK');
        await pace();

        const first = await task(page, '提取全图飞机并计算面积占比');
        assert.equal(first.status, 'completed');
        assert.equal(first.mode, 'demo');
        await pace();
        await drawRoi(page);
        const second = await task(page, '提取框选区域内的飞机并统计面积');
        assert.equal(second.status, 'completed');
        const [x1, y1, x2, y2] = second.task.roi.xyxy;
        const scope = (x2 - x1) * (y2 - y1);
        const expected = new Intl.NumberFormat('zh-CN', {
            style: 'percent', maximumFractionDigits: 2
        }).format(second.metrics.pixel_area / scope);
        assert.equal(await page.locator('#nationalStats').getAttribute('data-result-id'), second.id);
        assert.equal(await page.locator('#scopeRatio').textContent(), expected);
        assert.equal(await page.locator('#ratioValue').textContent(), (second.metrics.area_ratio * 100).toFixed(2));
        assert.ok((await page.locator('#scopePixels').textContent()).includes(scope.toLocaleString('en-US')));
        runRecords.push(first, second);
        await page.locator('body').press('Home');
        await screenshot(page, '01-workbench.png');
        await pace();

        await page.locator('#nationalRegions').click();
        await matches(page, '#modalTitle', '候选连通区域');
        assert.ok(await page.locator('#nationalRegionRows tr').count() > 0);
        await screenshot(page, '02-regions.png');
        await pace();
        await page.locator('#closeModal').click();
        await page.locator('#nationalDiagnosis').click();
        await matches(page, '#modalBody', '尚无质量诊断结果');
        await page.locator('#closeModal').click();

        await page.locator('#nationalCompare').click();
        await page.locator('#nationalCompareBody').waitFor();
        assert.equal(await page.locator('#nationalBase').inputValue(), first.id);
        assert.equal(await page.locator('#nationalTarget').inputValue(), second.id);
        assert.ok((await page.locator('#nationalCompareMeta').textContent()).includes(second.id));
        await pace();
        await page.locator('#nationalView').selectOption('difference');
        const diffValues = await page.locator('#nationalCompareMetrics dd').allTextContents();
        assert.deepEqual(diffValues.slice(0, 4), [
            '0', (first.metrics.pixel_area - second.metrics.pixel_area).toLocaleString('en-US'),
            second.metrics.pixel_area.toLocaleString('en-US'), first.metrics.pixel_area.toLocaleString('en-US')
        ]);
        assert.equal(diffValues[4], (second.metrics.pixel_area / first.metrics.pixel_area).toFixed(4));
        assert.equal(await page.locator('#nationalCompareCanvas').getAttribute('width'), '800');
        await page.locator('#nationalCompareViewport').scrollIntoViewIfNeeded();
        await screenshot(page, '03-difference.png');
        await pace();

        await page.locator('#nationalBase').selectOption(second.id);
        await matches(page, '#nationalCompareStatus', '请选择两个不同的结果');
        assert.ok(await page.locator('#nationalCompareBody').isHidden());
        await page.locator('#closeModal').click();
        await page.locator('[data-run="' + first.id + '"]').click();
        await matches(page, '#scopeValue', '全图');
        assert.equal(await page.locator('#nationalStats').getAttribute('data-result-id'), first.id);
        await page.locator('[data-run="' + second.id + '"]').click();
        await matches(page, '#scopeValue', '矩形研究区');
        await page.locator('.execution-panel summary').click();
        await matches(page, '#nationalTrace', second.id);
        assert.deepEqual(await page.locator('.national-steps li strong').allTextContents(), second.trace.map(event => event.name));
        await page.locator('#nationalTrace').scrollIntoViewIfNeeded();
        await screenshot(page, '04-execution.png');
        await pace();

        await page.locator('#connectionBtn').click();
        await page.locator('#failureDemo').click();
        await matches(page, '#nationalTrace', '失败');
        assert.equal(await page.locator('#nationalStats').getAttribute('data-result-id'), second.id);
        await page.locator('#clearRoiBtn').click();
        await matches(page, '#roiLabel', '未设置矩形研究区');
        const invalid = await task(page, '提取框选区域内的飞机并统计面积');
        assert.equal(invalid.status, 'failed');
        await matches(page, '#nationalTrace', '没有有效的矩形 ROI');
        const ambiguous = await task(page, '把那个区域提出来');
        assert.equal(ambiguous.status, 'needs_clarification');
        await matches(page, '#nationalTrace', '待澄清');
        await drawRoi(page, [.02, .02, .08, .08]);
        const empty = await task(page, '提取框选区域内的飞机并统计面积');
        assert.equal(empty.status, 'needs_review');
        assert.equal(await page.locator('#scopeRatio').textContent(), '0%');
        await page.locator('[data-run="' + second.id + '"]').click();
        await page.waitForFunction(id => document.getElementById('nationalStats').dataset.resultId === id, second.id);

        const downloadEvent = page.waitForEvent('download');
        await page.locator('#exportBtn').click();
        const zip = await downloadEvent;
        await zip.saveAs(path.join(evidence, 'roi-experiment.zip'));
        assert.ok(zip.suggestedFilename().includes(second.id));
        const report = await page.request.get(new URL(second.report_url, base).href);
        assert.equal(report.status(), 200);
        const reportText = await report.text();
        assert.ok(reportText.includes(second.metrics.pixel_area.toLocaleString('en-US')));
        await fs.writeFile(path.join(evidence, 'roi-report.md'), reportText);
        await page.locator('#nationalExperiments').click();
        await matches(page, '#nationalExperimentTable', '0.3056');
        assert.equal(await page.locator('#nationalExperimentTable tbody tr').count(), 9);
        assert.equal(await page.locator('#nationalExperimentTable tbody a').count(), 9);
        const a4 = page.locator('[data-experiment="a4_b025_l07"]');
        assert.match(await a4.textContent(), /历史诊断集.*10 图.*25%.*λ 0\.7.*0\.3191.*0\.4251/);
        assert.match(await a4.locator('a').getAttribute('href'), /c1ef225.*\/a4_b025_l07\.csv$/);
        assert.match(await page.locator('[data-experiment="a4_b025_l03"]').textContent(), /0\.3100（9 图）/);
        await screenshot(page, '05-experiments.png');
        await pace();
        await page.locator('#closeModal').click();
        await page.locator('#nationalRetrieval').click();
        await matches(page, '#nationalSearchStatus', '未就绪');
        assert.ok(await page.locator('#nationalSearchSubmit').isDisabled());
        assert.equal(await page.locator('#nationalSearchHits article').count(), 0);
        await screenshot(page, '06-retrieval-not-ready.png');
        await pace();
        await page.locator('#closeModal').click();

        await page.reload();
        await matches(page, '#scopeValue', '矩形研究区');
        assert.equal(await page.locator('#nationalStats').getAttribute('data-result-id'), second.id);
        assert.deepEqual(errors, []);
        const exported = runRecords.map(run => ({
            id: run.id, session_id: run.session_id, task: run.task, mode: run.mode,
            status: run.status, provenance: run.provenance, metrics: run.metrics, trace: run.trace
        }));
        await fs.writeFile(path.join(evidence, 'saved-versions.json'), JSON.stringify(exported, null, 2));
        console.log(JSON.stringify({ comparedIds: runRecords.map(run => run.id), scope, actualRatio: expected, diffValues }));
    } finally {
        const video = page.video();
        await context.close();
        if (video) {
            await video.saveAs(path.join(evidence, 'workbench-rehearsal.webm'));
            await video.delete();
        }
    }
}

async function verifyContracts(browser) {
    // Test doubles for unpublished teammate interfaces. No evidence screenshots or recordings use these routes.
    const page = await browser.newPage({ viewport: { width: 390, height: 850 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    try {
        await page.goto(base + '/?workbench=national');
        await matches(page, '#qualityMode', '自动（按建筑选择）');
        const original = await task(page, '提取全图建筑');
        await page.route('**/api/run', async route => {
            const response = await route.fetch();
            const run = await response.json();
            run.metrics = { ...run.metrics, pixel_area: 10, area_ratio: .1, total_pixels: 100 };
            run.task = { ...run.task, side: 'all', roi: { xyxy: [0, 0, 5, 5] } };
            run.tool_execution = {
                image_id: run.session_id, result_id: run.id,
                steps: [
                    { tool: 'get_segmentation', status: 'completed' },
                    { tool: 'apply_scope', status: 'completed' },
                    { tool: 'region_statistics', status: 'failed', error: { message: '测试：步骤 3 校验失败' } },
                    { tool: 'compare_versions', status: 'not_executed' }
                ]
            };
            await bodyJson(route, run);
        }, { times: 1 });
        const synthetic = await task(page, '提取右侧建筑');
        assert.equal(await page.locator('#nationalStats').getAttribute('data-result-id'), synthetic.id);
        assert.equal(await page.locator('#scopeRatio').textContent(), '40%');
        assert.equal(await page.locator('#ratioValue').textContent(), '10.00');
        assert.deepEqual(await page.locator('.national-steps .national-state').allTextContents(), ['完成', '完成', '失败', '未执行']);
        await page.locator('[data-run="' + original.id + '"]').click();
        await matches(page, '#scopeValue', '全图');

        await page.route('**/api/national/retrieval/status', route => bodyJson(route, {
            ready: true, model: 'Test RemoteCLIP fixture', index_version: 'test-index', max_top_k: 2
        }));
        await page.route('**/api/national/retrieval/search', route => {
            assert.deepEqual(route.request().postDataJSON(), { query: 'airport with airplanes', top_k: 2 });
            return bodyJson(route, {
                status: 'completed', real_execution: true, model: 'Test RemoteCLIP fixture',
                results: [{ image_id: 'airport', rank: 1, similarity: .31 }]
            });
        });
        await page.locator('#nationalRetrieval').click();
        await matches(page, '#nationalSearchStatus', '检索就绪');
        await page.locator('#nationalSearchQuery').fill('airport with airplanes');
        await page.locator('#nationalSearchSubmit').click();
        await matches(page, '#nationalSearchHits', '相似度 0.31');
        await page.getByRole('button', { name: '分析该影像' }).click();
        await matches(page, '#imageName', 'JFK');
        assert.equal(await page.locator('#nationalStats').isHidden(), true);
        await page.locator('#query:focus').waitFor();
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
        assert.ok(overflow <= 1, 'mobile horizontal overflow: ' + overflow);
        assert.deepEqual(errors, []);
        console.log('Frontend test doubles: ROI 10% / 40%, failure at step 3, RemoteCLIP hit mapping and mobile layout passed');
    } finally {
        await page.context().close();
    }
}

async function verifyArchivedAfr(browser, archive, dataRoot) {
    const imageId = 'ISAID_VAL_P1529';
    const resultDir = path.join(archive, 'artifacts', 'a5_b025_l07', imageId);
    const input = path.join(dataRoot, 'iSAID', 'selected_20260927', 'images', imageId + '.png');
    const [metadata, csv, original] = await Promise.all([
        fs.readFile(path.join(resultDir, 'afr.json'), 'utf8').then(JSON.parse),
        fs.readFile(path.join(archive, 'results', 'a5_b025_l07.csv'), 'utf8'),
        fs.readFile(input)
    ]);
    const [header, ...lines] = csv.trim().split(/\r?\n/).map(line => line.split(','));
    const fields = lines.find(line => line[header.indexOf('image_id')] === imageId);
    assert.equal(fields?.length, header.length);
    const record = Object.fromEntries(header.map((key, i) => [key, fields[i]]));
    assert.equal(createHash('sha256').update(original).digest('hex'), record.image_sha256);
    assert.equal(metadata.config_sha256, record.config_sha256);
    const [width, height] = metadata.image_size;
    const pixels = Number(record.prediction_foreground_pixels);
    const imageIdInApp = 'af1529000000';
    const runId = 'af1529000001';
    const imageDir = '/experiments/' + imageIdInApp;
    const runDir = imageDir + '/' + runId;
    const run = {
        id: runId, session_id: imageIdInApp, version: 1, mode: 'live', status: 'completed',
        query: '回放队长归档的 A5 飞机结果', message: '历史归档回放 · ' + imageId,
        task: {target: 'aircraft', side: 'all', effective_quality_mode: 'adaptive'},
        trace: [], duration_ms: Number(record.client_wall_ms),
        mask_url: runDir + '/mask.png', service_metadata: {...metadata.p0_service, service_version: record.service_version},
        metrics: {pixel_area: pixels, area_ratio: pixels / (width * height), total_pixels: width * height, width, height},
        afr: {
            status: 'completed', algorithm: metadata.algorithm_name, algorithm_version: metadata.algorithm_version,
            image_size: metadata.image_size, disagreement_ref: runDir + '/disagreement.png',
            disagreement: {
                available: metadata.disagreement_available,
                review_regions: metadata.review.review_regions, notice: metadata.disagreement_notice
            }
        }
    };
    const session = {
        id: imageIdInApp, width, height, name: 'A5 历史归档回放 · ' + imageId,
        image_url: imageDir + '/original.png', runs: [run], context: {}
    };

    // Supply the frozen files to the browser using the production AFR response shape.
    // This path exercises rendering and version binding without calling the model.
    const page = await browser.newPage({viewport: {width: 1512, height: 1040}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    try {
        let unavailable = false;
        await page.addInitScript(id => localStorage.setItem('geoscope-session', id), imageIdInApp);
        await page.route('**/api/session/' + imageIdInApp, route => bodyJson(route, session));
        await page.route('**' + session.image_url, route => route.fulfill({body: original, contentType: 'image/png'}));
        await page.route('**' + run.mask_url, route => route.fulfill({path: path.join(resultDir, 'mask.png')}));
        await page.route('**' + run.afr.disagreement_ref, route =>
            unavailable ? route.fulfill({status: 404}) : route.fulfill({
                path: path.join(resultDir, 'disagreement.png'), headers: {'Cache-Control': 'no-store'}
            })
        );
        await page.route('**/api/review', route => {
            assert.deepEqual(route.request().postDataJSON(), {
                session_id: imageIdInApp, run_id: runId, decision: 'accepted'
            });
            return bodyJson(route, {
                session_id: imageIdInApp, run_id: runId,
                human_review: {decision: 'accepted', label: '符合预期'}
            });
        });
        await page.goto(base + '/?workbench=national');
        await matches(page, '#imageName', '历史归档回放');
        await matches(page, '#qualityValue', 'GeoScope-AFR 自适应推理');
        await page.locator('#nationalDiagnosis').click();
        await page.locator('#diagnosticBody').waitFor();
        await matches(page, '#diagnosticStatus', '待复核');
        assert.equal(await page.locator('#diagnosticRegions button').count(), Number(record.review_region_count));
        const target = page.locator('#diagnosticCanvas');
        assert.equal(await target.getAttribute('width'), String(width));
        const samplePixels = await page.evaluate(async url => {
            const mask = await createImageBitmap(await (await fetch(url)).blob());
            const canvas = document.createElement('canvas');
            canvas.width = mask.width;
            canvas.height = mask.height;
            const ctx = canvas.getContext('2d', {willReadFrequently: true});
            ctx.drawImage(mask, 0, 0);
            mask.close();
            let foreground;
            let background;
            for (let y = 0; y < canvas.height && (!foreground || !background); y += 64) {
                const data = ctx.getImageData(0, y, canvas.width, Math.min(64, canvas.height - y)).data;
                for (let i = 0; i < data.length && (!foreground || !background); i += 4) {
                    const offset = i / 4;
                    const point = [offset % canvas.width, y + Math.floor(offset / canvas.width)];
                    if (data[i] === 255 && !foreground) foreground = point;
                    if (data[i] === 0 && !background) background = point;
                }
            }
            return [foreground, background];
        }, run.afr.disagreement_ref);
        async function renderPixels() {
            return target.evaluate((canvas, points) => {
                const ctx = canvas.getContext('2d');
                return points.map(([x, y]) => [...ctx.getImageData(x, y, 1, 1).data]);
            }, samplePixels);
        }
        const tinted = await renderPixels();
        await page.locator('#diagnosticToggle').uncheck();
        const plain = await renderPixels();
        assert.notDeepEqual(tinted[0], plain[0]);
        assert.deepEqual(tinted[1], plain[1]);
        await page.locator('#diagnosticToggle').check();
        const initialSize = await target.evaluate(canvas => canvas.clientWidth);
        await page.locator('#diagnosticRegions button').first().click();
        assert.ok(await target.evaluate(canvas => canvas.clientWidth) > initialSize);
        await page.locator('#diagnosticFit').click();
        await screenshot(page, '07-afr-archive-review.png');
        await page.locator('#diagnosticAccept').click();
        await matches(page, '#diagnosticStatus', '符合预期');
        await page.locator('#closeModal').click();

        unavailable = true;
        await page.reload();
        await page.locator('#nationalDiagnosis').click();
        await matches(page, '#diagnosticStatus', '分歧图不可用');
        assert.equal(await page.locator('#diagnosticBody').isHidden(), true);
        run.afr = {status: 'fallback'};
        await page.reload();
        await matches(page, '#qualityValue', 'AFR 回退');
        await page.locator('#nationalDiagnosis').click();
        await matches(page, '#modalBody', '已回退整图模式');
        assert.deepEqual(errors, []);
        console.log('Archived A5 P1529: verified PNG overlay, review regions, zoom, missing artifact and fallback states.');
    } finally {
        await page.context().close();
    }
}

(async () => {
    await fs.mkdir(evidence, { recursive: true });
    const browser = await chromium.launch({ headless: true, channel: process.env.GEO_BROWSER_CHANNEL || 'msedge' });
    try {
        await verifyWorkflow(browser);
        await verifyContracts(browser);
        if (process.env.GEO_AFR_ARCHIVE && process.env.GEO_AFR_DATA_ROOT) {
            await verifyArchivedAfr(browser, process.env.GEO_AFR_ARCHIVE, process.env.GEO_AFR_DATA_ROOT);
        }
    } finally {
        await browser.close();
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
