const $ = id => document.getElementById(id);
const state = { session:null, selected:null, last:null, original:null, mask:null, mode:'demo', view:'original', zoom:1, busy:false, status:null, loadToken:0 };
const sideNames = {all:'全图',left:'左半幅',right:'右半幅',top:'上半幅',bottom:'下半幅'};
const targetNames = {building:'建筑',aircraft:'飞机',road:'道路',water:'水体',tree:'植被',ship:'船舶'};
const statusNames = {completed:'待语义复核',answered:'已回答',needs_clarification:'待确认',failed:'未完成',needs_review:'待复核',export_ready:'可导出'};
const number = n => Number(n).toLocaleString('en-US');
const escape = s => String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const icon = name => `<svg><use href="#i-${name}"/></svg>`;
const intro = $('chat').innerHTML;
let toastTimer;
function toast(text){$('toast').textContent=text;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,3800);}
async function api(path,payload){const response=await fetch(path,payload?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}:{});const data=await response.json();if(!response.ok)throw new Error(data.error||'请求失败');return data;}
function loadImage(url){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('影像加载失败'));image.src=url;});}
function modal(title,html,eyebrow='GEOSCOPE LAB'){$('modalTitle').textContent=title;$('modalEyebrow').textContent=eyebrow;$('modalBody').innerHTML=html;if(!$('modal').open)$('modal').showModal();}
function closeModal(){$('modal').close();}
$('closeModal').onclick=closeModal;
$('modal').addEventListener('click',e=>{if(e.target===$('modal')){const r=$('modal').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)closeModal();}});

async function newExperiment(sample='urban',image=null,name=null){
 if(state.busy)return toast('请等待当前任务完成');
 try{
  const session=await api('/api/session',{sample,image,name});
  const original=await loadImage(session.image_url);
  state.loadToken++;Object.assign(state,{session,original,selected:null,last:null,mask:null,view:'original',zoom:1});
  clearTimeout(toastTimer);$('toast').hidden=true;
  $('chat').innerHTML=intro;
  if(session.sample==='airport'){
   const btn=$('starterList').querySelectorAll('button')[1];btn.dataset.prompt='提取右侧飞机并计算面积占比';btn.querySelector('span').innerHTML='提取右侧飞机<small>目标提取 + 面积占比</small>';
  }
  $('imageName').textContent=session.name;$('imageMeta').textContent=`RGB · ${session.width} × ${session.height} px · 像素坐标`;
  $('fileThumb').src=session.image_url;$('coordinateLabel').textContent=`${session.width} × ${session.height} px`;
  $('stageLabel').textContent=session.sample?'研究样区 / '+(session.sample==='urban'?'01':'02'):'用户上传影像';
  $('imageWatermark').textContent=session.sample?'SAMPLE IMAGE':'USER IMAGE';
  $('contextChip').hidden=true;$('versions').innerHTML='<span class="small-muted">完成提取后，在这里保留每个结果版本</span>';
  $('trace').innerHTML='<div class="trace-placeholder">'+icon('target')+'<span>每一步操作，都有可追溯的依据。</span><small>任务理解 → 专业感知 → 几何统计 → 结果检查</small></div>';
  $('runLabel').textContent='等待实验开始';$('query').value='';updateMode();resetMetrics();draw();syncView();closeModal();
  localStorage.setItem('geoscope-session',session.id);
 }catch(e){toast(e.message);}
}

function updateMode(){
 $('connectionLabel').textContent=state.mode==='demo'?'演示模式':'模型模式';
 $('provenanceLabel').textContent=state.mode==='demo'?(state.session?.sample?'内置样例 · 预置标注演示':'上传影像 · 需连接模型'):'模型模式 · 实际服务调用';
 $('composerNote').textContent=state.mode==='demo'?'当前为流程演示，未调用模型':'上传影像将发送至你配置的模型服务';
}
function resetMetrics(){
 $('ratioValue').textContent='—';$('areaValue').innerHTML='— <small>px</small>';$('scopeValue').textContent='—';$('timeValue').textContent='—';
 $('totalValue').textContent=number(state.session?.width*state.session?.height||0);$('resultTitle').textContent='等待目标提取';$('resultSubtitle').textContent='运行一条任务，结果将在这里呈现';
 $('resultStatus').textContent='准备就绪';$('resultStatus').className='result-badge';$('coverageBar').style.width='0%';$('ringValue').setAttribute('stroke-dasharray','0 164');
 $('checksList').innerHTML=['结果与原始影像对齐','空间条件完成检查','统计与产物已生成'].map(s=>`<p class="pending"><span>○</span>${s}</p>`).join('');$('checkCount').textContent='0 / 3';$('exportBtn').disabled=true;
}

function draw(){
 if(!state.original)return;
 const canvas=$('imageCanvas'),ctx=canvas.getContext('2d'),w=state.original.naturalWidth,h=state.original.naturalHeight;
 canvas.width=w;canvas.height=h;
 ctx.drawImage(state.original,0,0);
 if(state.mask){
  const maskCanvas=document.createElement('canvas');maskCanvas.width=w;maskCanvas.height=h;const m=maskCanvas.getContext('2d',{willReadFrequently:true});m.drawImage(state.mask,0,0,w,h);
  if(state.view==='mask'){ctx.drawImage(state.mask,0,0,w,h);}
  else if(state.view==='overlay'||state.view==='compare'){
   const data=m.getImageData(0,0,w,h),opacity=Number($('opacity').value)/100;
   for(let i=0;i<data.data.length;i+=4){const yes=data.data[i]>127;data.data[i]=76;data.data[i+1]=224;data.data[i+2]=153;data.data[i+3]=yes?Math.round(opacity*255):0;}
   m.putImageData(data,0,0);ctx.save();
   const split=w*Number($('compareSlider').value)/100;
   if(state.view==='compare'){ctx.beginPath();ctx.rect(split,0,w-split,h);ctx.clip();}
   ctx.drawImage(maskCanvas,0,0);ctx.restore();
   if(state.view==='compare'){ctx.strokeStyle='#f4faef';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(split,0);ctx.lineTo(split,h);ctx.stroke();}
  }
  if(state.view!=='original'&&state.view!=='mask'){
   const side=state.selected?.task?.side;
   if(side&&side!=='all'){ctx.save();ctx.setLineDash([7,7]);ctx.strokeStyle='#effbd8bb';ctx.lineWidth=1.5;ctx.beginPath();if(side==='left'||side==='right'){ctx.moveTo(w/2,0);ctx.lineTo(w/2,h);}else{ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);}ctx.stroke();ctx.restore();}
  }
 }
 canvas.style.transform=`scale(${state.zoom})`; $('zoomLabel').textContent=Math.round(state.zoom*100)+'%';
 $('legendScope').textContent=state.selected?.metrics?sideNames[state.selected.task.side]:'暂无提取结果';
 $('imageWatermark').textContent=state.selected?.mode==='demo'&&state.view!=='original'?'DEMO ANNOTATION · NOT MODEL OUTPUT':state.session?.sample?'SAMPLE IMAGE':'USER IMAGE';
}
function syncView(){document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('selected',b.dataset.view===state.view));$('compareControl').hidden=state.view!=='compare';}
function view(mode){if(mode!=='original'&&!state.mask)return toast('请先运行提取任务，再查看蒙版或对比');state.view=mode;draw();syncView();}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>view(b.dataset.view));
$('opacity').oninput=()=>{$('opacityValue').textContent=$('opacity').value+'%';draw();};$('compareSlider').oninput=draw;
$('zoomIn').onclick=()=>{state.zoom=Math.min(3,state.zoom+.25);draw();};$('zoomOut').onclick=()=>{state.zoom=Math.max(.5,state.zoom-.25);draw();};$('fitBtn').onclick=()=>{state.zoom=1;draw();};
$('imageCanvas').addEventListener('mousemove',e=>{const r=e.target.getBoundingClientRect();const x=Math.floor((e.clientX-r.left)/r.width*e.target.width),y=Math.floor((e.clientY-r.top)/r.height*e.target.height);$('coordinateLabel').textContent=`X ${x} · Y ${y} px`;});
$('imageCanvas').addEventListener('mouseleave',()=>{$('coordinateLabel').textContent=`${state.session.width} × ${state.session.height} px`;});

function addMessage(role,text,kind='',run=null){
 const d=document.createElement('div');d.className=`message ${role} ${kind}`;
 d.innerHTML=`<div class="message-label"><span>${role==='user'?'YOU':'GEOSCOPE'}</span><span>${new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'})}</span></div><div class="bubble"></div>`;
 d.querySelector('.bubble').textContent=text;
 if(run?.mask_url){
  const target=targetNames[run.task.target],next=run.task.side==='left'?'右':'左';
  const chips=document.createElement('div');chips.className='followup-chips';
  for(const q of [`改成${next}侧的${target}`,'提取全图'+target,'导出刚才的结果']){const btn=document.createElement('button');btn.textContent=q;btn.dataset.prompt=q;chips.append(btn);}d.append(chips);
  const meta=document.createElement('div');meta.className='message-meta';meta.textContent=`V${run.version} · ${run.mode==='demo'?'预置标注演示':'真实服务返回'} · ${run.duration_ms} ms`;d.append(meta);
 }
 if(run?.status==='needs_clarification'){
  const chips=document.createElement('div');chips.className='followup-chips';const target=state.selected?.task?.target||state.status.samples[state.session.sample]?.target||'building';
  for(const side of ['左侧','右侧']){const b=document.createElement('button');b.textContent=side+targetNames[target];b.dataset.prompt='提取'+side+targetNames[target];chips.append(b);}d.append(chips);
 }
 $('chat').append(d);$('chat').scrollTop=$('chat').scrollHeight;
}
function renderTrace(run){
 $('runLabel').textContent=`RUN ${run.id.slice(0,6).toUpperCase()} · ${run.duration_ms} ms`;
 $('trace').innerHTML=run.trace.map(t=>`<div class="trace-step ${escape(t.state)}" title="${escape(t.detail)}"><strong><b>${String(t.step).padStart(2,'0')}</b>${escape(t.name)}</strong><p>${escape(t.detail)}</p></div>`).join('');
 $('executionDot').style.background=run.status==='failed'?'#c89969':'#6d967b';
}
function renderVersions(){
 const versions=state.session.runs.filter(r=>r.mask_url);
 $('versions').innerHTML=versions.map(r=>`<button class="version-pill ${r.id===state.selected?.id?'selected':''}" data-run="${r.id}"><span>V${r.version}</span>${escape(sideNames[r.task.side])}${escape(targetNames[r.task.target])}</button>`).join('')||'<span class="small-muted">完成提取后，在这里保留每个结果版本</span>';
}
async function selectRun(run){
 const token=++state.loadToken;
 try{const mask=await loadImage(run.mask_url);if(token!==state.loadToken)return;
  state.selected=run;state.mask=mask;state.view='overlay';const m=run.metrics,ratio=m.area_ratio*100;
  $('ratioValue').textContent=ratio.toFixed(2);$('areaValue').innerHTML=number(m.pixel_area)+' <small>px</small>';$('scopeValue').textContent=sideNames[run.task.side];$('timeValue').textContent=run.duration_ms+' ms';
  $('resultTitle').textContent=sideNames[run.task.side]+targetNames[run.task.target];$('resultSubtitle').textContent=`V${run.version} · ${run.mode==='demo'?'预置演示标注，非模型输出':'模型输出，需人工复核'}`;
  $('resultStatus').textContent=statusNames[run.status];$('resultStatus').className='result-badge '+(run.status==='completed'?'success':'warning');
  $('coverageBar').style.width=ratio+'%';$('ringValue').setAttribute('stroke-dasharray',`${ratio*1.634} 164`);
  $('checksList').innerHTML=['蒙版尺寸与原始影像一致',`像素范围符合${sideNames[run.task.side]}条件`,'统计及结果文件已生成'].map(s=>`<p><span>✓</span>${escape(s)}</p>`).join('');$('checkCount').textContent='3 / 3';
  $('contextChip').hidden=false;$('contextChip').lastElementChild.textContent=`继承 V${run.version} · ${targetNames[run.task.target]} · ${sideNames[run.task.side]}`;
  $('exportBtn').disabled=false;renderVersions();renderTrace(run);draw();syncView();
 }catch(e){toast(e.message);}
}
async function run(query,simulateFailure=false){
 if(state.busy||!state.session)return;
 query=query.trim();if(!query)return;
 state.busy=true;$('sendBtn').disabled=true;$('newBtn').disabled=true;$('busyOverlay').hidden=false;$('query').value='';
 if($('starterList'))$('starterList').remove();
 const welcome=$('chat').querySelector('.assistant-intro');if(welcome)welcome.remove();
 addMessage('user',query);const parent=state.selected?.id||null;
 try{
  const result=await api('/api/run',{session_id:state.session.id,query,mode:state.mode,parent_run_id:parent,simulate_failure:simulateFailure});
  state.session.runs.push(result);state.last=result;
  addMessage('assistant',result.message,['failed','needs_clarification','needs_review'].includes(result.status)?'warning':'',result);
  if(result.mask_url)await selectRun(result);else renderTrace(result);
  if(result.status==='export_ready')download(result.export_url);
  if(result.status==='failed'&&state.selected)toast('本次未完成，画布保留上一次有效结果');
 }catch(e){addMessage('assistant',e.message,'warning');toast('请求未完成，未生成结果');}
 finally{state.busy=false;$('sendBtn').disabled=false;$('newBtn').disabled=false;$('busyOverlay').hidden=true;}
}
$('taskForm').onsubmit=e=>{e.preventDefault();run($('query').value);};
$('query').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();run($('query').value);}};
document.addEventListener('click',e=>{const prompt=e.target.closest('[data-prompt]');if(prompt)run(prompt.dataset.prompt);const v=e.target.closest('[data-run]');if(v&&!state.busy){const r=state.session.runs.find(x=>x.id===v.dataset.run);if(r?.mask_url){selectRun(r);closeModal();}}});
function download(url){const a=document.createElement('a');a.href=url;a.download='';document.body.append(a);a.click();a.remove();}
$('exportBtn').onclick=()=>{if(state.selected){download(state.selected.export_url);toast('实验包已开始下载，包含6个真实生成的文件');}};
function gallery(){
 if(state.busy)return toast('请等待当前任务完成');
 modal('选择一幅研究影像',`<p>从预置样例体验完整实验，或上传自己的影像连接模型分析。</p><div class="sample-grid">${Object.entries(state.status?.samples||{}).map(([key,s])=>`<button class="sample-card" data-sample="${key}"><img src="assets/${escape(s.file)}" alt="${escape(s.name)}"><strong>${escape(s.name)}</strong><small>${s.size.join(' × ')} px · 预置${targetNames[s.target]}近似标注</small></button>`).join('')}</div><p style="margin-top:16px">样例蒙版用于演示交互与几何计算，不是RemoteSAM预测，也不是精度评测真值。</p><button class="secondary" id="modalUpload">${icon('image')}上传自己的影像</button>`,'IMAGE LIBRARY');
 $('modalBody').querySelectorAll('[data-sample]').forEach(b=>b.onclick=()=>newExperiment(b.dataset.sample));$('modalUpload').onclick=()=>{closeModal();$('fileInput').click();};
}
$('galleryBtn').onclick=gallery;$('newBtn').onclick=gallery;$('uploadBtn').onclick=gallery;$('workspaceBtn').onclick=()=>window.scrollTo({top:0,behavior:'smooth'});
$('fileInput').onchange=async e=>{const file=e.target.files[0];if(!file)return;if(file.size>12*1024*1024){toast('请上传12MB以内的影像');e.target.value='';return;}const reader=new FileReader();reader.onload=()=>newExperiment(null,reader.result,file.name);reader.onerror=()=>toast('无法读取影像');reader.readAsDataURL(file);e.target.value='';};
function history(){
 const runs=state.session?.runs||[];
 modal('实验记录',runs.length?`<p>同一影像的结果分别保存。点击有效结果恢复查看，并以该版本为后续任务上下文。</p>${[...runs].reverse().map(r=>`<button class="history-row" ${r.mask_url?`data-run="${r.id}"`:''} ${!r.mask_url?'disabled':''}><span>V${r.version}</span><div><strong>${escape(r.query)}</strong><small>${statusNames[r.status]} · ${r.mode==='demo'?'演示':'模型'}</small></div><time>${r.duration_ms} ms</time></button>`).join('')}`:'<p>尚未运行实验。从提取右侧建筑开始，你的任务和结果会记录在这里。</p>','EXPERIMENT HISTORY');
}
$('historyBtn').onclick=history;
function details(log=false){
 const r=log?state.last||state.selected:state.selected||state.last;
 if(!r)return modal('运行详情','<p>尚未执行任务。完成实验后，这里会展示真实的工具事件、数据来源和运行参数。</p>');
 modal(log?'完整执行日志':'结果与运行详情',`<p><strong>RUN ${r.id}</strong> · ${r.mode==='demo'?'预置标注演示':'模型服务模式'} · ${statusNames[r.status]}</p><div class="modal-notice">${escape(r.provenance.perception)}</div>${r.mask_url?`<div class="download-list"><a href="${r.mask_url}" download="mask.png">mask.png ↓</a><a href="${r.overlay_url}" download="overlay.png">overlay.png ↓</a><a href="${r.export_url}" download>完整实验包 ↓</a></div>`:''}<h3>数据来源与方法</h3><p>${escape(r.provenance.source)}</p><p>规划器：${escape(r.provenance.planner)}。面积按实际二值蒙版计算，未推断平方米。</p><pre>${escape(JSON.stringify(log?r.trace:r,null,2))}</pre>`,'RUN PROVENANCE');
}
$('detailBtn').onclick=()=>details(false);$('logBtn').onclick=()=>details(true);
$('sourceBtn').onclick=()=>modal('影像与数据来源',`<p>${escape(state.session?.source||'未选择影像')}</p><p>内置建筑和飞机区域为人工绘制的近似演示多边形，只用于检验交互、上下文和几何计算。不能据此报告模型准确率。</p><p>坐标以图像左上角为原点，X向右、Y向下。上下左右均为影像方向。面积占比使用整图像素为分母。</p>`,'DATA NOTES');
async function connections(){
 try{state.status=await api('/api/status');}catch(e){return toast('本地服务不可用');}
 const canLive=state.status.agent_configured&&state.status.sam_configured;
 modal('实验运行方式',`<div class="mode-options"><button id="demoMode" class="${state.mode==='demo'?'selected':''}">样例演示<small>预置标注 · 真实几何计算</small></button><button id="liveMode" class="${state.mode==='live'?'selected':''}" ${!canLive?'disabled':''}>模型服务<small>${canLive?'已配置，执行时验证连通':'等待配置认知与感知服务'}</small></button></div><div class="connection-list"><div><span>认知核心</span><b>${state.status.agent_configured?'已配置（待调用验证）':'未连接'}</b></div><div><span>RemoteSAM</span><b>${state.status.sam_configured?'已配置（待调用验证）':'未连接'}</b></div><div><span>几何计算与实验存储</span><b style="color:#5d8850">本地服务可用</b></div></div><p>真实模型模式将影像发送至本地服务端配置的模型地址。API密钥不进入网页。</p><h3>连接方法</h3><p>按项目README配置GEO_AGENT_BASE_URL、GEO_AGENT_MODEL和GEO_REMOTESAM_URL，然后重启服务。配置状态不等于已通过模型测试。</p><button class="secondary" id="failureDemo">体验工具失败反馈</button>`,'RUNTIME SETTINGS');
 $('demoMode').onclick=()=>{state.mode='demo';updateMode();closeModal();toast('已切换到样例演示');};$('liveMode').onclick=()=>{state.mode='live';updateMode();closeModal();toast('已切换到模型服务');};
 $('failureDemo').onclick=()=>{closeModal();run('检查工具失败时能否保留已有实验结果',true);};
}
$('connectionBtn').onclick=connections;
$('aboutBtn').onclick=()=>modal('让一个问题，成为可复查的实验',`<p>观域 GeoScope 是按你的v3技术方案制作的第一版工作台。</p><h3>建议体验顺序</h3><p>了解场景 → 提取右侧建筑 → 改成左侧 → 对比原图与结果 → 导出实验包。</p><h3>目前已经能做什么</h3><p>真实影像上传、演示任务解析、上下文继承、版本切换、像素范围检查、面积统计、文件导出、异常反馈；已提供真实模型服务接口。</p><h3>当前边界</h3><p>内置样例来自本地科研仓库，蒙版为近似人工演示标注。当前未运行RemoteAgent或RemoteSAM权重，不提供模型精度结论。任意新影像需要接通模型服务。支持全图及上下左右半幅；复杂关系和地理面积暂未开放。</p><p>实验文件保存在本机experiments目录。当前界面会话在服务重启后需新建，旧实验文件仍保留。</p>`,'ABOUT THE PROTOTYPE');

async function init(){
 try{state.status=await api('/api/status');await newExperiment('urban');}
 catch(e){modal('本地服务尚未启动','<p>请双击项目目录中的“启动Demo.bat”，再打开 http://127.0.0.1:4180。</p>');}
}
init();
