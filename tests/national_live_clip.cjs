// Actual RemoteCLIP + actual B UI + demo segmentation, with no request mocks.
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
(async () => {
 const base = process.env.GEO_TEST_BASE_URL;
 const browser = await chromium.launch({headless:true, channel:'msedge'});
 try {
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'/?workbench=national');
  await page.locator('#nationalRetrieval').click();
  await page.waitForFunction(()=>document.querySelector('#nationalSearchStatus').textContent.includes('检索就绪'));
  await page.locator('#nationalSearchQuery').fill('airport with airplanes');
  const search = page.waitForResponse(r=>r.url().endsWith('/retrieval/search'));
  await page.locator('#nationalSearchSubmit').click();
  const result=await (await search).json();assert.equal(result.data.real_execution,true);
  await page.locator('#nationalSearchHits article').first().waitFor();
  await fs.mkdir('docs/evidence',{recursive:true});
  await page.screenshot({path:'docs/evidence/clip_live_workbench.png'});
  await page.getByRole('button',{name:'分析该影像'}).first().click();
  await page.waitForFunction(()=>document.querySelector('#imageName').textContent.includes('JFK'));
  const sid=result.data.results[0].image_id;
  const response=await page.request.post(base+'/api/run',{data:{session_id:sid,mode:'demo',query:'提取全部飞机'}});
  const run=await response.json();assert.equal(run.mode,'demo');assert.equal(run.status,'completed');
  const plan=await page.request.post(base+'/api/national/plans/execute',{data:{task:'region_analysis',image_id:sid,result_id:run.id,target:'aircraft',scope:'all',steps:['get_segmentation','apply_scope','region_statistics','report_generation']}});
  const execution=await plan.json();assert.equal(execution.success,true);
  await page.reload();await page.locator('#galleryBtn').click();
  // Open the existing selected image through the actual platform API and UI.
  await page.evaluate(async id=>{ await openExperiment(id); },sid);
  await page.locator('[data-run="'+run.id+'"]').click();
  await page.locator('.execution-panel').evaluate(el=>el.open=true);
  await page.waitForFunction(()=>document.querySelectorAll('#nationalTrace .national-state').length===4);
  await page.screenshot({path:'docs/evidence/verifiedplan_live_workbench.png'});
  assert.deepEqual(errors,[]);
  await fs.writeFile('docs/evidence/browser_real_clip.json',JSON.stringify({actual_clip:true,segmentation_mode:'demo',image_id:sid,result_id:run.id,plan:execution.data,errors},null,2));
  console.log('Real RemoteCLIP → B image selection → explicit demo segmentation → VerifiedPlan persisted steps: passed');
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
