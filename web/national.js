(() => {
    'use strict';

    if (new URLSearchParams(location.search).get('workbench') !== 'national') return;

    const missing = '未记录';
    const apiRoot = '/api/national';
    const tools = [
        ['semantic_segmentation', '语义分割', '模型模式 · Harness 白名单'],
        ['referring_expression_segmentation', '指代分割', '模型模式 · Harness 白名单'],
        ['region_statistics', '研究区统计', '当前版本 · 像素'],
        ['connected_region_analysis', '连通域分析', '当前版本 · 8 邻域候选'],
        ['mask_compare', '蒙版对照', '当前影像 · 保存的二值蒙版'],
        ['spatial_relation_analysis', '空间关系分析', '尚未开放'],
        ['adaptive_refinement', 'GeoScope-AFR 自适应推理', '尚未接入 · 可继续使用现有模式'],
        ['semantic_image_retrieval', 'RemoteCLIP 语义检索', '打开检索页面检查就绪状态'],
        ['geo_measurement', '地理面积测算', '尚未开放'],
        ['report_generation', '实验报告', '当前版本 · 保留原有报告和实验包']
    ];
    const statuses = {
        completed: '完成', answered: '完成', running: '运行中', pending: '待执行',
        skipped: '未执行', not_executed: '未执行', failed: '失败', warning: '告警',
        rejected: '拒绝', needs_clarification: '待澄清', waiting: '待澄清',
        not_ready: '未就绪', empty: '无结果'
    };

    const finite = value => typeof value === 'number' && Number.isFinite(value);
    const count = value => finite(value) ? number(value) : missing;
    const percentage = value => finite(value)
        ? new Intl.NumberFormat('zh-CN', {style: 'percent', maximumFractionDigits: 2}).format(value) : missing;
    const ms = value => finite(value) ? count(value) + ' ms' : missing;
    const measurement = (label, value) => '<div><dt>' + escape(label) + '</dt><dd>' + escape(value) + '</dd></div>';
    const active = element => element.isConnected && $('modal').open;
    const outputSource = run => run.mode === 'demo' ? '预置演示标注' : '模型运行记录';
    const hasResult = run => Boolean(run?.mask_url && run?.metrics);
    const versionLabel = run => 'V' + run.version + ' · ' + targetLabel(run);

    function sameOriginUrl(value) {
        if (typeof value !== 'string' || !value) return null;
        try {
            const url = new URL(value, location.href);
            return url.origin === location.origin && /^https?:$/.test(url.protocol) && !url.username && !url.password
                ? url.href : null;
        } catch {
            return null;
        }
    }

    function rectangle(value, width, height) {
        if (!Array.isArray(value) || value.length !== 4 || !value.every(Number.isInteger)) return null;
        const [x1, y1, x2, y2] = value;
        return x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 && x2 <= width && y2 <= height ? value : null;
    }

    // The existing Harness uses half-open pixel bounds and floor splits for odd dimensions.
    function scopeFor(run, width, height) {
        const halfX = Math.floor(width / 2);
        const halfY = Math.floor(height / 2);
        const boxes = {
            all: [0, 0, width, height], left: [0, 0, halfX, height],
            right: [halfX, 0, width, height], top: [0, 0, width, halfY],
            bottom: [0, halfY, width, height]
        };
        const side = run.task?.side ?? run.metrics.scope;
        const bounds = boxes[side];
        if (!bounds) return null;
        const roi = run.task?.roi ?? run.metrics.roi;
        if (!roi) return {bounds, label: sideNames[side]};
        const box = rectangle(roi.xyxy, width, height);
        if (!box) return null;
        return {
            bounds: [Math.max(bounds[0], box[0]), Math.max(bounds[1], box[1]),
                Math.min(bounds[2], box[2]), Math.min(bounds[3], box[3])],
            label: side === 'all' ? '矩形研究区' : sideNames[side] + ' ∩ 矩形研究区'
        };
    }

    function metadata(run) {
        const algorithm = run.algorithm_info;
        const service = run.service_metadata;
        const describe = (name, version) => [name, version].filter(Boolean).join(' · ') || missing;
        return [
            ['结果 ID', run.id],
            ['来源', outputSource(run)],
            ['模型', run.mode === 'demo' ? '未调用' : describe(
                run.model_info?.name ?? service?.model ?? run.provenance?.perception,
                run.model_info?.version ?? service?.service_version)],
            ['算法 / 版本', describe(algorithm?.name ?? algorithm?.algorithm_id, algorithm?.version)],
            ['实际模式', run.mode === 'demo' ? '预置演示' :
                qualityNames[run.task?.effective_quality_mode] ?? service?.quality_mode ?? missing],
            ['执行耗时', ms(run.duration_ms)]
        ];
    }

    function renderStatistics(run) {
        const panel = $('nationalStats');
        panel.hidden = !hasResult(run);
        if (panel.hidden) {
            panel.replaceChildren();
            delete panel.dataset.resultId;
            return;
        }

        const metrics = run.metrics;
        const {width, height} = state.session;
        const scope = scopeFor(run, width, height);
        const scopePixels = scope ? Math.max(0, scope.bounds[2] - scope.bounds[0]) *
            Math.max(0, scope.bounds[3] - scope.bounds[1]) : null;
        const scopeRatio = scopePixels > 0 && finite(metrics.pixel_area) && metrics.pixel_area <= scopePixels
            ? metrics.pixel_area / scopePixels : null;
        panel.dataset.resultId = run.id;
        panel.innerHTML = '<div class="national-coverage"><div><span>研究区内占比</span><strong id="scopeRatio">' +
            escape(percentage(scopeRatio)) + '</strong></div><small id="scopePixels">' +
            (scopePixels === null ? missing : count(scopePixels) + ' px 研究区') + '</small></div>' +
            '<p class="national-muted">' + escape(scope?.label ?? '范围' + missing) + ' · ' +
                width + ' × ' + height + ' px</p>' +
            '<div class="national-result-actions"><button type="button" class="secondary" id="nationalRegions">区域分布</button>' +
            '<button type="button" class="secondary" id="nationalDiagnosis">质量诊断</button></div>' +
            '<details class="national-meta"><summary>数据来源与版本</summary><dl>' +
            metadata(run).map(([label, value]) => measurement(label, value)).join('') + '</dl></details>';
        $('nationalRegions').onclick = () => showRegions(run);
        $('nationalDiagnosis').onclick = () => showDiagnostics(run);
    }

    function showRegions(run) {
        const stats = run.metrics.candidate_stats;
        if (!stats) {
            modal('候选区域分布', '<p role="status">当前版本尚未保存连通域统计。</p>', versionLabel(run));
            return;
        }
        const rows = Array.isArray(stats.candidates) ? stats.candidates : [];
        const bins = [
            ['1–15', 1, 15], ['16–63', 16, 63], ['64–255', 64, 255],
            ['256–1023', 256, 1023], ['≥1024', 1024, Infinity]
        ];
        const binCounts = bins.map(([, low, high]) =>
            rows.filter(item => item.area_pixels >= low && item.area_pixels <= high).length);
        const max = Math.max(1, ...binCounts);
        const headers = ['编号', '面积 / px', '中心 (x, y)', '外接框 (x, y, w, h)'];
        modal('候选连通区域分布', '<div class="national-dialog"><p>基于裁切后的蒙版，边界可能截断候选区域。' +
            '候选连通区域不等于准确实例计数。</p><dl class="national-metrics">' + [
                ['候选区域', count(stats.candidate_count)], ['过滤前', count(stats.raw_candidate_count)],
                ['已过滤', count(stats.filtered_out_count)],
                ['最小保留面积 / px', count(stats.min_area_pixels)]
            ].map(([label, value]) => measurement(label, value)).join('') + '</dl>' +
            (rows.length ? '<h3>面积分布</h3><div class="national-histogram">' +
            bins.map(([label], i) => '<div><span>' + label + ' px</span><meter min="0" max="' + max +
                '" value="' + binCounts[i] + '"></meter><b>' + count(binCounts[i]) + '</b></div>').join('') +
            '</div><p>最小 ' + count(stats.min_area) + ' px · 平均 ' + count(stats.mean_area) + ' px · 最大 ' +
            count(stats.max_area) + ' px</p><div class="national-scroll national-region-scroll"><table class="national-table"><thead><tr>' +
            headers.map(text => '<th scope="col">' + text + '</th>').join('') +
            '</tr></thead><tbody id="nationalRegionRows"></tbody></table></div>' +
            '<div class="national-controls"><button id="regionPrev" type="button">上一页</button>' +
            '<span id="regionPage"></span><button id="regionNext" type="button">下一页</button></div>' :
            '<p role="status">该版本没有通过面积阈值的候选区域。</p>') + '</div>', versionLabel(run) + ' · ' + run.id);
        if (!rows.length) return;
        let page = 0;
        const pageSize = 20;
        function drawPage() {
            $('nationalRegionRows').innerHTML = rows.slice(page * pageSize, (page + 1) * pageSize).map(item => {
                const box = item.bbox;
                const center = item.center;
                const values = [item.candidate_id, count(item.area_pixels),
                    center ? [center.x, center.y].join(', ') : missing,
                    box ? [box.x, box.y, box.width, box.height].join(', ') : missing];
                return '<tr>' + values.map(value => '<td>' + escape(value) + '</td>').join('') + '</tr>';
            }).join('');
            $('regionPage').textContent = (page + 1) + ' / ' + Math.ceil(rows.length / pageSize);
            $('regionPrev').disabled = page === 0;
            $('regionNext').disabled = (page + 1) * pageSize >= rows.length;
        }
        $('regionPrev').onclick = () => { page--; drawPage(); };
        $('regionNext').onclick = () => { page++; drawPage(); };
        drawPage();
    }

    function renderExecution(run, error = null, pending = false) {
        const panel = $('nationalTrace');
        if (!run) {
            panel.innerHTML = '<p role="status">' + escape(pending ? '正在等待服务返回实际执行事件…' :
                error ? '请求失败 · ' + error : '执行任务后查看实际步骤。') + '</p>';
            return;
        }
        const execution = run.tool_execution;
        const typed = execution?.image_id === run.session_id && execution?.result_id === run.id &&
            Array.isArray(execution.steps);
        const steps = typed ? execution.steps : (run.trace ?? []);
        panel.innerHTML = '<div class="national-run-header"><strong>' + escape(run.query) + '</strong><span>' +
            escape(run.id) + ' · ' + escape(outputSource(run)) + '</span></div><ol class="national-steps">' +
            steps.map(step => {
                const status = step.status ?? step.state;
                const duration = finite(step.duration_ms) && (typed || step.duration_ms > 0) ? ms(step.duration_ms) : '';
                const detail = step.error?.message ?? step.error ?? step.detail ?? '';
                const inputs = step.inputs ? '<details><summary>实际输入</summary><pre>' +
                    escape(JSON.stringify(step.inputs, null, 2)) + '</pre></details>' : '';
                const output = step.outputs?.result_id;
                return '<li><div><strong>' + escape(step.tool ?? step.name ?? step.step_id ?? missing) +
                    '</strong><span class="national-state ' + escape(status) + '">' + escape(statuses[status] ?? missing) +
                    (duration ? ' · ' + escape(duration) : '') + '</span></div><p>' + escape(detail) + '</p>' +
                    inputs + (output ? '<small>产物版本 · ' + escape(output) + '</small>' : '') + '</li>';
            }).join('') + '</ol>' + (run.report_url ? '<a href="' +
                escape(sameOriginUrl(run.report_url) ?? '#') + '" download>下载此运行的报告</a>' : '');
    }

    function showToolRegistry() {
        modal('科研工具', '<div class="national-dialog"><p id="nationalToolStatus" role="status">正在检查工具注册表…</p>' +
            '<dl class="national-tool-list">' + tools.map(([key, label, fallback]) =>
                '<div><dt>' + label + '<code>' + key + '</code></dt><dd data-national-tool="' + key + '">' +
                fallback + '</dd></div>').join('') + '</dl></div>', '执行能力');
        const label = $('nationalToolStatus');
        requestNational('/tools/registry').then(response => {
            if (!active(label)) return;
            const registry = response.tools;
            if (!Array.isArray(registry)) throw new Error('工具注册表格式未确认');
            label.textContent = '后端工具状态';
            for (const entry of registry) {
                const row = [...document.querySelectorAll('[data-national-tool]')]
                    .find(item => item.dataset.nationalTool === entry.name);
                const text = entry.available === true ? '后端可用' : entry.reason ?? statuses[entry.status] ?? '不可用';
                if (row) row.textContent = text;
            }
        }).catch(() => {
            if (active(label)) label.textContent = '国赛工具注册表未就绪。下方为当前工作台能力。';
        });
    }

    async function requestNational(path, options = {}) {
        let response;
        try {
            response = await fetch(apiRoot + path, {signal: AbortSignal.timeout(30000), ...options});
        } catch {
            throw new Error('连接失败或请求超时，请检查服务后重试。');
        }
        if (response.status === 404 || response.status === 501) {
            const error = new Error('国赛接口尚未接入此工作台。');
            error.status = response.status;
            throw error;
        }
        const raw = await response.json().catch(() => null);
        const error = raw?.error;
        if (!raw || !response.ok || raw.success === false || ['not_ready', 'failed', 'rejected'].includes(raw.status)) {
            throw new Error(error?.message ?? (typeof error === 'string' ? error : null) ??
                raw?.message ?? '服务暂不可用，请检查模型和索引状态。');
        }
        if (!raw.data) return raw;
        return {
            ...raw.data, status: raw.status ?? raw.data.status,
            real_execution: raw.meta?.real_execution ?? raw.data.real_execution
        };
    }

    function contextFor(canvas, session) {
        canvas.width = session.width;
        canvas.height = session.height;
        return canvas.getContext('2d', {willReadFrequently: true});
    }

    async function loadAligned(url, session) {
        const href = sameOriginUrl(url);
        if (!href) throw new Error('产物地址无效或不属于当前服务。');
        const image = await loadImage(href);
        if (image.naturalWidth !== session.width || image.naturalHeight !== session.height) {
            throw new Error('蒙版或诊断图尺寸与原图不一致，已停止比较。');
        }
        return image;
    }

    async function readMask(run, session) {
        const image = await loadAligned(run.mask_url, session);
        const canvas = document.createElement('canvas');
        const ctx = contextFor(canvas, session);
        const {width, height} = canvas;
        ctx.drawImage(image, 0, 0);
        const mask = new Uint8Array(width * height);
        const stripHeight = Math.max(1, Math.floor(262144 / width));
        for (let y = 0; y < height; y += stripHeight) {
            const data = ctx.getImageData(0, y, width, Math.min(stripHeight, height - y)).data;
            for (let i = 0, offset = y * width; i < data.length; i += 4, offset++) {
                const value = data[i];
                if ((value !== 0 && value !== 255) || data[i + 1] !== value || data[i + 2] !== value || data[i + 3] !== 255) {
                    throw new Error('比较要求 0/255 不透明二值蒙版，当前有效像素编码不符。');
                }
                mask[offset] = value / 255;
            }
        }
        canvas.width = 0;
        return {mask, image};
    }

    function compareMasks(base, target, session) {
        const canvas = document.createElement('canvas');
        const ctx = contextFor(canvas, session);
        const {width, height} = canvas;
        const diff = ctx.createImageData(width, height);
        const colors = {added: [0, 223, 198], removed: [255, 163, 91]};
        let intersection = 0;
        let added = 0;
        let removed = 0;
        for (let i = 0; i < base.mask.length; i++) {
            if (base.mask[i] === target.mask[i]) {
                intersection += base.mask[i];
                continue;
            }
            if (target.mask[i]) added++;
            else removed++;
            diff.data.set(colors[target.mask[i] ? 'added' : 'removed'], i * 4);
            diff.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(diff, 0, 0);
        const union = intersection + added + removed;
        return {canvas, added, removed, intersection, union, agreement: union ? intersection / union : 1};
    }

    function viewer(viewport, canvas, session, draw) {
        let zoom = 1;
        const fitWidth = () => {
            const maximumHeight = parseFloat(getComputedStyle(viewport).maxHeight);
            return Math.max(1, Math.min(viewport.clientWidth, maximumHeight * session.width / session.height));
        };
        function redraw() {
            canvas.style.width = (fitWidth() * zoom) + 'px';
            draw();
        }
        const observer = new ResizeObserver(() => {
            if (active(viewport)) redraw();
        });
        observer.observe(viewport);
        $('modal').addEventListener('close', () => observer.disconnect(), {once: true});
        return {
            redraw,
            zoom(value) {
                zoom = value;
                redraw();
                if (value === 1) viewport.scrollTo(0, 0);
            },
            locate(box) {
                const [x1, y1, x2, y2] = box;
                const fit = fitWidth();
                zoom = Math.min(8, Math.max(2, Math.min(
                    0.8 * viewport.clientWidth * session.width / (fit * (x2 - x1)),
                    0.8 * viewport.clientHeight * session.width / (fit * (y2 - y1)))));
                redraw();
                viewport.scrollTo(
                    (x1 + x2) / 2 * canvas.clientWidth / session.width - viewport.clientWidth / 2,
                    (y1 + y2) / 2 * canvas.clientHeight / session.height - viewport.clientHeight / 2
                );
                return zoom;
            }
        };
    }

    function showComparison() {
        const session = state.session;
        const runs = (session?.runs ?? []).filter(hasResult);
        if (runs.length < 2) {
            modal('结果版本对照', '<p role="status">请在同一影像上保存至少两个带蒙版的结果。</p>');
            return;
        }
        const options = runs.map(run => '<option value="' + escape(run.id) + '">' + escape(versionLabel(run)) + '</option>').join('');
        modal('同图版本对照', '<div class="national-dialog national-wide"><p class="national-context">' +
            escape(session.name) + ' · ' + session.width + ' × ' + session.height + ' px</p>' +
            '<div class="national-selectors"><label>基准版本<select id="nationalBase">' + options + '</select></label>' +
            '<label>对照版本<select id="nationalTarget">' + options + '</select></label></div>' +
            '<p id="nationalCompareStatus" class="national-notice" role="status"></p>' +
            '<section id="nationalCompareBody" hidden><div id="nationalCompareMeta" class="national-version-pair"></div>' +
            '<div class="national-controls"><label>显示<select id="nationalView">' +
            '<option value="split">基准 / 对照滑动</option><option value="difference">差异叠加</option>' +
            '<option value="original">原图</option><option value="base">基准二值蒙版</option>' +
            '<option value="target">对照二值蒙版</option></select></label>' +
            '<label id="nationalSplitControl">分割线<input aria-label="版本分割线" id="nationalSplit" type="range" min="0" max="100" value="50"></label>' +
            '<label>叠加透明度<input id="nationalCompareOpacity" type="range" min="0" max="100" value="68"></label>' +
            '<label>缩放<select id="nationalCompareZoom"><option value="1">适应</option><option value="1.5">150%</option><option value="2">200%</option><option value="3">300%</option></select></label></div>' +
            '<div class="national-compare-grid"><div><div class="national-viewport" id="nationalCompareViewport" tabindex="0" aria-label="对照影像，可滚动定位">' +
            '<canvas id="nationalCompareCanvas" aria-label="同原图像素坐标的基准、对照或差异蒙版"></canvas></div>' +
            '<div id="nationalCompareLegend" class="national-legend"></div></div>' +
            '<dl id="nationalCompareMetrics" class="national-metrics"></dl></div>' +
            '<div class="national-controls"><button id="nationalVerifyDiff" type="button">核对后端差异</button>' +
            '<output id="nationalVerifyStatus" aria-live="polite"></output></div></section></div>', '结果与复核');
        const baseSelect = $('nationalBase');
        const targetSelect = $('nationalTarget');
        const status = $('nationalCompareStatus');
        const body = $('nationalCompareBody');
        const target = runs.find(item => item.id === state.selected?.id) ?? runs.at(-1);
        const base = runs.find(item => item.id === target.parent_run_id) ?? runs.find(item => item.id !== target.id);
        baseSelect.value = base.id;
        targetSelect.value = target.id;
        let revision = 0;
        let comparison;
        const canvas = $('nationalCompareCanvas');
        const ctx = contextFor(canvas, session);
        const zoom = viewer($('nationalCompareViewport'), canvas, session, drawComparison);

        function drawComparison() {
            if (!comparison) return;
            const mode = $('nationalView').value;
            const {baseMask, targetMask, diff} = comparison;
            const opacity = Number($('nationalCompareOpacity').value) / 100;
            const {width, height} = session;
            $('nationalSplitControl').hidden = mode !== 'split';
            $('nationalCompareOpacity').disabled = !['split', 'difference'].includes(mode);
            ctx.globalAlpha = 1;
            ctx.drawImage(state.original, 0, 0);
            if (mode === 'base' || mode === 'target') {
                ctx.drawImage(mode === 'base' ? baseMask.image : targetMask.image, 0, 0);
            } else if (mode === 'difference') {
                ctx.globalAlpha = opacity;
                ctx.drawImage(diff.canvas, 0, 0);
                ctx.globalAlpha = 1;
            } else if (mode === 'split') {
                const split = width * $('nationalSplit').value / 100;
                ctx.globalAlpha = opacity;
                ctx.drawImage(baseMask.overlay, 0, 0);
                ctx.save();
                ctx.beginPath();
                ctx.rect(split, 0, width - split, height);
                ctx.clip();
                ctx.globalAlpha = 1;
                ctx.drawImage(state.original, 0, 0);
                ctx.globalAlpha = opacity;
                ctx.drawImage(targetMask.overlay, 0, 0);
                ctx.restore();
                ctx.globalAlpha = 1;
                ctx.fillStyle = '#fff';
                ctx.fillRect(split - 1, 0, 2, height);
            }
            $('nationalCompareLegend').innerHTML = {
                split: '<span><i class="base"></i>左侧 · 基准蒙版</span><span><i class="target"></i>右侧 · 对照蒙版</span>',
                difference: '<span><i class="added"></i>对照新增像素</span><span><i class="removed"></i>对照减少像素</span>',
                base: '<span>白色 · 基准前景</span><span>黑色 · 背景</span>',
                target: '<span>白色 · 对照前景</span><span>黑色 · 背景</span>'
            }[mode] ?? '';
        }

        function overlayFor(mask, color) {
            const layer = document.createElement('canvas');
            const layerContext = contextFor(layer, session);
            const data = layerContext.createImageData(layer.width, layer.height);
            for (let i = 0; i < mask.length; i++) {
                if (!mask[i]) continue;
                data.data.set(color, i * 4);
            }
            layerContext.putImageData(data, 0, 0);
            return layer;
        }

        async function loadComparison() {
            const current = ++revision;
            body.hidden = true;
            comparison = null;
            const baseRun = runs.find(item => item.id === baseSelect.value);
            const targetRun = runs.find(item => item.id === targetSelect.value);
            if (!baseRun || !targetRun || baseRun.id === targetRun.id) {
                status.textContent = '请选择两个不同的结果版本。';
                return;
            }
            if ([baseRun, targetRun].some(run => run.session_id !== session.id) ||
                (baseRun.provenance?.image_sha256 && targetRun.provenance?.image_sha256 &&
                    baseRun.provenance.image_sha256 !== targetRun.provenance.image_sha256) ||
                baseRun.task?.target !== targetRun.task?.target ||
                Boolean(baseRun.task?.invert) !== Boolean(targetRun.task?.invert)) {
                status.textContent = '请选择属于同一原图、目标类别和取反方式的版本。';
                return;
            }
            status.textContent = '正在读取两版二值蒙版…';
            try {
                const [baseMask, targetMask] = await Promise.all([readMask(baseRun, session), readMask(targetRun, session)]);
                if (!active(status) || current !== revision || state.session !== session) return;
                baseMask.overlay = overlayFor(baseMask.mask, [133, 176, 255, 255]);
                targetMask.overlay = overlayFor(targetMask.mask, [42, 224, 164, 255]);
                const diff = compareMasks(baseMask, targetMask, session);
                comparison = {baseMask, targetMask, diff, baseRun, targetRun};
                $('nationalCompareMeta').innerHTML = [baseRun, targetRun].map((run, i) =>
                    '<section><h3>' + (i ? '对照' : '基准') + ' · ' + escape(versionLabel(run)) + '</h3><dl>' +
                    metadata(run).map(([key, value]) => measurement(key, value)).join('') + '</dl></section>').join('');
                $('nationalCompareMetrics').innerHTML = [
                    ['对照新增 / px', count(diff.added)], ['对照减少 / px', count(diff.removed)],
                    ['交集 / px', count(diff.intersection)], ['并集 / px', count(diff.union)],
                    ['蒙版一致性 IoU', diff.agreement.toFixed(4)]
                ].map(([label, value]) => measurement(label, value)).join('');
                const sameScope = JSON.stringify(scopeFor(baseRun, session.width, session.height)?.bounds) ===
                    JSON.stringify(scopeFor(targetRun, session.width, session.height)?.bounds);
                status.textContent = '按原图坐标复算已保存蒙版。' +
                    (sameScope ? '' : ' 两版范围不同，差异包含裁切影响。') +
                    (baseRun.mode === 'demo' || targetRun.mode === 'demo' ? ' 含预置演示标注。' : '') +
                    ' IoU 衡量两张蒙版的一致性。' + (diff.union ? '' : ' 两份空蒙版按 1.0000 计。');
                $('nationalVerifyStatus').textContent = '';
                $('nationalVerifyDiff').disabled = false;
                body.hidden = false;
                zoom.redraw();
            } catch (error) {
                if (active(status) && current === revision) status.textContent = error.message;
            }
        }

        $('nationalVerifyDiff').onclick = async () => {
            const requested = comparison;
            const output = $('nationalVerifyStatus');
            const button = $('nationalVerifyDiff');
            if (!requested) return;
            button.disabled = true;
            output.textContent = '正在请求后端复算…';
            try {
                const {baseRun, targetRun, diff} = requested;
                const payload = {image_id: session.id, base_result_id: baseRun.id, target_result_id: targetRun.id};
                const reply = await requestNational('/tools/mask-compare', jsonPost(payload));
                if (!active(output) || requested !== comparison) return;
                const fields = [
                    ['width', session.width], ['height', session.height], ['added_pixels', diff.added],
                    ['removed_pixels', diff.removed], ['intersection_pixels', diff.intersection], ['union_pixels', diff.union]
                ];
                const match = [...Object.entries(payload), ...fields].every(([key, value]) => reply[key] === value) &&
                    finite(reply.prediction_agreement_iou) && Math.abs(reply.prediction_agreement_iou - diff.agreement) < 1e-6;
                output.textContent = match ? '后端统计与浏览器逐像素结果一致。' : '后端版本绑定或差异统计与当前结果不一致。';
            } catch (error) {
                if (active(output) && requested === comparison) output.textContent = error.message;
            } finally {
                if (active(output) && requested === comparison) button.disabled = false;
            }
        };
        baseSelect.onchange = loadComparison;
        targetSelect.onchange = loadComparison;
        $('nationalView').onchange = drawComparison;
        $('nationalSplit').oninput = drawComparison;
        $('nationalCompareOpacity').oninput = drawComparison;
        $('nationalCompareZoom').onchange = event => zoom.zoom(Number(event.target.value));
        loadComparison();
    }

    async function showDiagnostics(run) {
        const diagnostic = run.diagnostics;
        const session = state.session;
        const aligned = diagnostic?.image_id === session.id && diagnostic?.result_id === run.id;
        if (!aligned || diagnostic.status !== 'completed' || !diagnostic.overlay_url) {
            modal('预测分歧与人工复核', '<p role="status">尚无质量诊断结果。可在版本对照中检查已保存的蒙版。</p>',
                versionLabel(run));
            return;
        }
        modal('预测分歧与人工复核', '<div class="national-dialog national-wide"><p id="diagnosticStatus" role="status">正在读取质量诊断产物…</p>' +
            '<section id="diagnosticBody" hidden><div class="national-controls"><label><input id="diagnosticToggle" type="checkbox" checked> 显示分歧</label>' +
            '<label>分歧透明度 <input id="diagnosticOpacity" type="range" min="0" max="100" value="60"></label>' +
            '<button id="diagnosticFit" type="button">适应画布</button></div>' +
            '<div id="diagnosticViewport" class="national-viewport" tabindex="0"><canvas id="diagnosticCanvas" aria-label="原图与分歧叠加"></canvas></div>' +
            '<ol id="diagnosticRegions" class="national-diagnostic-list"></ol><div class="national-controls">' +
            '<button type="button" id="diagnosticAccept">确认该版本</button>' +
            '<button type="button" id="diagnosticAdjust">标记需要调整</button></div></section></div>', versionLabel(run) + ' · ' + run.id);
        const status = $('diagnosticStatus');
        try {
            const layer = await loadAligned(diagnostic.overlay_url, session);
            if (!active(status) || state.session !== session || state.selected?.id !== run.id) return;
            const canvas = $('diagnosticCanvas');
            const ctx = contextFor(canvas, session);
            const bounds = (Array.isArray(diagnostic.regions) ? diagnostic.regions : []).flatMap(region => {
                const box = rectangle(region.xyxy, session.width, session.height);
                return box ? [{...region, box}] : [];
            });
            let selected = null;
            function drawDiagnostic() {
                ctx.globalAlpha = 1;
                ctx.drawImage(state.original, 0, 0);
                if ($('diagnosticToggle').checked) {
                    ctx.globalAlpha = Number($('diagnosticOpacity').value) / 100;
                    ctx.drawImage(layer, 0, 0);
                    ctx.globalAlpha = 1;
                }
                if (selected) {
                    const [x1, y1, x2, y2] = selected;
                    ctx.strokeStyle = '#fff1ac';
                    ctx.lineWidth = Math.max(2, session.width / 700);
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                }
            }
            function showReviewStatus() {
                status.textContent = (run.human_review?.label ?? '待复核') + ' · ' +
                    (diagnostic.legend ?? '分歧层来自该版本的质量诊断产物') +
                    ' · ' + outputSource(run);
                $('diagnosticAccept').disabled = run.mode === 'demo' || run.human_review?.decision === 'accepted';
                $('diagnosticAdjust').disabled = run.mode === 'demo' || run.human_review?.decision === 'needs_adjustment';
            }
            const display = viewer($('diagnosticViewport'), canvas, session, drawDiagnostic);
            $('diagnosticRegions').innerHTML = bounds.map((region, i) =>
                '<li><button type="button" data-region-index="' + i + '">' +
                escape(region.label ?? '待复核区域 ' + (i + 1)) + ' · ' + region.box.join(', ') + ' px</button></li>').join('');
            $('diagnosticRegions').onclick = event => {
                const button = event.target.closest('[data-region-index]');
                if (!button) return;
                selected = bounds[Number(button.dataset.regionIndex)].box;
                display.locate(selected);
            };
            $('diagnosticToggle').onchange = drawDiagnostic;
            $('diagnosticOpacity').oninput = drawDiagnostic;
            $('diagnosticFit').onclick = () => { selected = null; display.zoom(1); };
            function review(decision) {
                return async () => {
                    if (state.selected?.id !== run.id || state.busy) return;
                    $('diagnosticAccept').disabled = $('diagnosticAdjust').disabled = true;
                    await saveReview(decision);
                    if (active(status)) showReviewStatus();
                };
            }
            $('diagnosticAccept').onclick = review('accepted');
            $('diagnosticAdjust').onclick = review('needs_adjustment');
            $('diagnosticBody').hidden = false;
            showReviewStatus();
            display.redraw();
        } catch (error) {
            if (active(status)) status.textContent = error.message;
        }
    }

    const jsonPost = payload => ({
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
    });

    function showRetrieval() {
        modal('RemoteCLIP 图文语义检索', '<div class="national-dialog"><p>按文字查询相似的影像。相似度用于排序，像素分割在工作台中单独执行。</p>' +
            '<form id="nationalSearchForm"><label>检索文本<input id="nationalSearchQuery" type="search" placeholder="airport with airplanes" maxlength="240" required></label>' +
            '<label>候选上限<select id="nationalSearchTopK"><option>3</option><option selected>5</option><option>10</option></select></label>' +
            '<button type="submit" class="primary" id="nationalSearchSubmit" disabled>检索影像</button></form>' +
            '<p id="nationalSearchStatus" class="national-notice" role="status">正在检查模型和图库索引…</p>' +
            '<div class="national-controls"><button id="nationalSearchRetry" type="button">重新检查</button>' +
            '<button id="nationalOpenGallery" type="button">使用现有影像库</button></div>' +
            '<div id="nationalSearchHits" class="national-search-hits"></div></div>', '影像库');
        const form = $('nationalSearchForm');
        const submit = $('nationalSearchSubmit');
        const retry = $('nationalSearchRetry');
        const status = $('nationalSearchStatus');
        const hits = $('nationalSearchHits');
        const topK = $('nationalSearchTopK');
        let ready = false;

        async function check() {
            ready = false;
            submit.disabled = retry.disabled = true;
            status.textContent = '正在检查模型和图库索引…';
            hits.replaceChildren();
            try {
                const result = await requestNational('/retrieval/status');
                if (!active(status)) return;
                ready = result.ready === true || result.status === 'ready';
                if (result.index_size === 0) {
                    ready = false;
                    status.textContent = '图库为空，请先构建索引。';
                    return;
                }
                // Only offer positive choices allowed by the service.
                const limit = Number.isInteger(result.max_top_k) ? result.max_top_k : 10;
                if (limit < 1) {
                    ready = false;
                    status.textContent = '后端没有提供可用的检索候选数量。';
                    return;
                }
                const choices = [...new Set([Math.min(3, limit), Math.min(5, limit), Math.min(10, limit)])];
                topK.replaceChildren(...choices.map(item => new Option(item, item, false, item === Math.min(5, limit))));
                status.textContent = ready ? '检索就绪 · ' + (result.model ?? 'RemoteCLIP') +
                    (result.index_version ? ' · 索引 ' + result.index_version : '') :
                    result.message ?? '模型或图库索引未就绪。';
            } catch (error) {
                if (active(status)) status.textContent = 'RemoteCLIP 未就绪 · ' + error.message;
            } finally {
                if (active(status)) {
                    submit.disabled = !ready;
                    retry.disabled = false;
                }
            }
        }

        form.onsubmit = async event => {
            event.preventDefault();
            if (!ready || submit.disabled) return;
            const query = $('nationalSearchQuery').value.trim();
            if (!query) {
                status.textContent = '请输入检索文本。';
                return;
            }
            submit.disabled = retry.disabled = true;
            status.textContent = '正在等待 RemoteCLIP 返回排序…';
            hits.replaceChildren();
            try {
                const result = await requestNational('/retrieval/search', jsonPost({query, top_k: Number(topK.value)}));
                if (!active(status)) return;
                if (result.status !== 'completed' || result.real_execution !== true || !Array.isArray(result.results)) {
                    throw new Error('响应尚无经过标记的真实检索结果。');
                }
                const results = result.results;
                if (!results.length) {
                    status.textContent = '本次检索没有候选影像。';
                    return;
                }
                if (results.some(hit => typeof hit?.image_id !== 'string' || !Number.isInteger(hit.rank) ||
                    hit.rank < 1 || !finite(hit.similarity))) {
                    throw new Error('检索结果缺少影像 ID、排名或相似度。');
                }
                status.textContent = (result.model ?? 'RemoteCLIP') + ' · ' + results.length + ' 条候选' +
                    (result.embedding_version ? ' · 向量版本 ' + result.embedding_version : '') +
                    (result.index_version ? ' · 索引 ' + result.index_version : '');
                for (const hit of results.slice().sort((a, b) => a.rank - b.rank)) {
                    const sample = Object.hasOwn(state.status?.samples ?? {}, hit.image_id)
                        ? state.status.samples[hit.image_id] : null;
                    const thumb = sameOriginUrl(hit.thumbnail_url) ?? (sample ? '/assets/' + encodeURIComponent(sample.file) : null);
                    const card = document.createElement('article');
                    card.innerHTML = (thumb ? '<img src="' + escape(thumb) + '" alt="" loading="lazy">' : '') +
                        '<div><small>第 ' + hit.rank + ' 位 · ' + escape(hit.image_id) + '</small><h3>' +
                        escape(hit.name ?? sample?.name ?? hit.image_id) + '</h3><span>相似度 ' + escape(hit.similarity) +
                        '</span><button type="button" class="secondary">分析该影像</button></div>';
                    card.querySelector('button').onclick = async () => {
                        if (sample) await newExperiment(hit.image_id);
                        else if (/^[a-f0-9]{12}$/.test(hit.image_id)) await openExperiment(hit.image_id);
                        else {
                            status.textContent = '影像 ' + hit.image_id + ' 尚未映射到工作台样例或现有实验。';
                            return;
                        }
                        if (!$('modal').open) $('query').focus();
                    };
                    hits.append(card);
                }
            } catch (error) {
                if (active(status)) status.textContent = error.message;
            } finally {
                if (active(status)) submit.disabled = retry.disabled = false;
            }
        };
        retry.onclick = check;
        $('nationalOpenGallery').onclick = gallery;
        check();
    }

    async function showExperiments() {
        modal('算法实验', '<div class="national-dialog national-wide"><p id="nationalExperimentStatus" role="status">正在读取可复算记录…</p>' +
            '<div id="nationalExperimentTable" class="national-scroll"></div></div>', '实验记录');
        const status = $('nationalExperimentStatus');
        const historical = '/data/national-history.json';
        const current = '/data/national-experiments.json';
        const read = async path => {
            const reply = await fetch(path, {signal: AbortSignal.timeout(10000)});
            if (reply.status === 404 && path === current) return [];
            if (!reply.ok) throw new Error('实验记录读取失败。');
            const data = await reply.json();
            if (!Array.isArray(data)) throw new Error('实验记录格式不符。');
            return data;
        };
        const [historyReply, currentReply] = await Promise.allSettled([read(historical), read(current)]);
        if (!active(status)) return;
        const source = [...(currentReply.value ?? []), ...(historyReply.value ?? [])];
        const labels = ['原始整图', '提示规范化', '固定重叠切片', '自适应区域选择', '跨尺度融合', '一致性复核'];
        const columns = ['分组 / 方法', '来源 / 样本数', '模型与参数', 'IoU', 'Dice', 'Precision', 'Recall', '模型耗时 / 图', '产物'];
        function artifact(value) {
            const local = sameOriginUrl(value);
            if (local) return local;
            if (typeof value === 'string' && value.startsWith('https://github.com/SitianGao/geoscope/')) return value;
            return null;
        }
        const rows = labels.map((label, index) => {
            const group = 'A' + index;
            const record = source.find(item => item?.group === group && item.status === 'completed' &&
                item.real_execution === true && Number.isInteger(item.samples) && item.samples > 0 &&
                item.dataset_role && item.config && item.metrics && artifact(item.artifact_url));
            const cell = value => '<td>' + escape(value) + '</td>';
            const metric = key => finite(record?.metrics[key]) ? record.metrics[key].toFixed(4) : missing;
            const header = '<tr><th scope="row">' + group + ' · ' + label + '</th>';
            if (!record) return header + '<td colspan="8" class="national-muted">未完成或尚未登记可核查结果</td></tr>';
            return header + cell(record.dataset_role + ' · ' + record.samples + ' 图') + cell(record.config) +
                ['iou', 'dice', 'precision', 'recall'].map(key => cell(metric(key))).join('') +
                cell(ms(record.metrics.inference_ms)) + '<td><a href="' + escape(artifact(record.artifact_url)) +
                '" target="_blank" rel="noopener">逐图证据</a></td></tr>';
        });
        status.textContent = 'A0–A2 历史记录来自固定飞机诊断集，参数在同一批影像上选择。' +
            ' 各组显示已有真实模型评测记录；独立测试以登记的数据角色为准。' +
            (historyReply.status === 'rejected' || currentReply.status === 'rejected' ? ' 部分记录未能加载。' : '');
        $('nationalExperimentTable').innerHTML = '<table class="national-table national-experiment-table"><thead><tr>' +
            columns.map(label => '<th scope="col">' + label + '</th>').join('') +
            '</tr></thead><tbody>' + rows.join('') + '</tbody></table>';
    }

    document.body.classList.add('national-workbench');
    document.title = '观域 GeoScope · 遥感科研工作台';
    document.querySelector('.page-heading h1').textContent = '遥感科研工作台';
    document.querySelector('.hero-copy > p').textContent = '任务分析、同图版本对照与可复算实验记录';
    document.querySelector('.hero-copy > .eyebrow').hidden = true;
    $('workbenchSwitch').textContent = '返回初赛版';
    $('workbenchSwitch').href = '/';
    $('nationalRetrieval').hidden = false;
    $('nationalExperiments').hidden = false;
    $('nationalCompare').hidden = false;
    $('nationalTrace').hidden = false;
    $('nationalRetrieval').onclick = showRetrieval;
    $('nationalExperiments').onclick = showExperiments;
    $('nationalCompare').onclick = showComparison;
    $('metricsPanel').before($('nationalStats'));
    document.querySelector('.execution-panel .expand-hint').textContent = '步骤 · 状态 · 运行记录';
    const toolButton = document.createElement('button');
    toolButton.type = 'button';
    toolButton.className = 'text-button';
    toolButton.textContent = '工具能力';
    toolButton.onclick = showToolRegistry;
    $('qualityMode').after(toolButton);
    document.addEventListener('geoscope:result', event => {
        renderStatistics($('metricsPanel').hidden ? null : state.selected);
        renderExecution(event.detail.run, event.detail.error);
    });
    document.addEventListener('geoscope:running', () => renderExecution(null, null, true));
    renderStatistics(state.selected);
    renderExecution(state.selected ?? state.last);
})();
