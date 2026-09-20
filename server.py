"""GeoScope prototype: traceable sessions, real geometry, explicit demo/live modes."""
from __future__ import annotations
import base64
import hashlib
import io
import json
import os
import re
import threading
import time
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
WEB = ROOT / 'web'
DATA = ROOT / 'experiments'
DATA.mkdir(exist_ok=True)
SESSIONS = {}
LOCK = threading.RLock()
SID_RE = re.compile(r'^[a-f0-9]{12}$')
SIDES = {'all': '全图', 'left': '左半幅', 'right': '右半幅', 'top': '上半幅', 'bottom': '下半幅'}
TARGETS = {'building': '建筑', 'aircraft': '飞机', 'road': '道路', 'water': '水体', 'tree': '植被', 'ship': '船舶'}
SAMPLES = {
 'urban': {'name': '街区建筑研究样区', 'file': 'urban.png', 'target': 'building', 'size': [800,600],
  'source': '本地 RemoteAgent / SM3Det 仓库 tools/cam/5428.tif；原始影像来源与发布许可需在正式参赛前进一步核对。',
  'scene': '这是一幅包含建筑、道路和集中绿地的城市街区航拍样例。可以提取建筑区域，比较影像两侧的覆盖占比。此说明为样例预置文本，并非模型看图结果。'},
 'airport': {'name': '机场目标研究样区', 'file': 'airport.jpg', 'target': 'aircraft', 'size': [800,800],
  'source': '本地 RemoteSAM 官方仓库 assets/demo.jpg；仅作为原型样例，正式发布前核对影像许可。',
  'scene': '这幅预置样例展示停机坪上的两架飞机及周边车辆。可以提取左侧或右侧飞机并计算像素覆盖。此说明为样例预置文本，并非模型看图结果。'},
}

def uid(): return uuid.uuid4().hex[:12]
def now(): return datetime.now(timezone.utc).isoformat()
def save_json(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
def image_b64(image):
    b=io.BytesIO(); image.save(b,format='PNG'); return base64.b64encode(b.getvalue()).decode()

def demo_mask(sample):
    """Human-authored approximate polygons, not ground truth or model predictions."""
    mask=Image.new('L',tuple(SAMPLES[sample]['size']),0); d=ImageDraw.Draw(mask)
    if sample=='urban':
        polys=[[(0,0),(99,0),(85,31),(3,12)],[(117,0),(164,0),(191,9),(157,90),(111,71),(96,60)],
          [(209,17),(329,53),(314,79),(195,41)],[(315,77),(400,101),(388,124),(310,101)],
          [(415,94),(551,136),(538,165),(408,124)],[(498,158),(531,169),(488,237),(444,225)],
          [(426,243),(466,258),(445,289),(408,276)],[(379,294),(412,310),(369,385),(334,369)],
          [(327,371),(362,385),(350,410),(314,398)],[(249,324),(334,351),(314,391),(230,360)],
          [(117,278),(204,307),(188,334),(100,305)],[(5,235),(89,264),(72,292),(0,268)],
          [(24,377),(96,400),(80,432),(11,410)],[(9,413),(87,439),(75,471),(0,447)],
          [(659,205),(689,214),(627,313),(582,300)],[(570,353),(608,365),(553,437),(514,424)],
          [(467,440),(510,453),(487,480),(450,466)],[(552,469),(593,480),(574,512),(529,494)],
          [(628,443),(674,458),(639,518),(598,500)],[(681,381),(716,394),(690,439),(655,426)],
          [(736,435),(800,455),(800,481),(719,460)],[(706,484),(794,513),(782,537),(693,510)],
          [(431,532),(530,557),(512,600),(384,582)],[(659,535),(758,567),(746,592),(642,562)]]
    else:
        polys=[[(226,429),(235,442),(245,511),(382,521),(382,542),(249,555),(241,620),(285,641),(284,653),(235,649),(227,658),(217,649),(169,653),(169,639),(213,620),(211,557),(72,541),(72,524),(211,514),(214,450)],
               [(615,428),(624,443),(635,513),(762,524),(760,544),(632,557),(621,620),(665,639),(662,653),(614,650),(607,659),(598,650),(550,653),(548,637),(591,620),(590,556),(454,542),(454,521),(595,512),(602,447)]]
    for p in polys: d.polygon(p,fill=255)
    return mask

def constrain(mask, side):
    w,h=mask.size; region=Image.new('L',(w,h),0)
    boxes={'all':(0,0,w,h),'left':(0,0,w//2,h),'right':(w//2,0,w,h),'top':(0,0,w,h//2),'bottom':(0,h//2,w,h)}
    box=boxes[side]; region.paste(mask.crop(box),box[:2]); return region

def statistics(mask,side):
    w,h=mask.size; area=sum(mask.histogram()[1:]); bbox=mask.getbbox()
    return {'pixel_area':area,'total_pixels':w*h,'area_ratio':area/(w*h),'bbox_xyxy':list(bbox) if bbox else None,
            'width':w,'height':h,'scope':side,'scope_label':SIDES[side], 'unit':'pixel',
            'spatial_rule':'像素范围裁切；面积比例的分母始终为整幅影像',
            'spatial_check':mask.tobytes()==constrain(mask,side).tobytes(),'ground_area_available':False}

def parse_task(query,context):
    q=query.strip().lower()
    if re.search(r'靠近|附近|之间|东侧|西侧|平方米|公顷|米以内',q):
        return {'action':'clarify','question':'当前只支持影像坐标中的左右、上下范围和像素占比。请改用这类条件；真实距离或地理面积需要额外的空间参考数据。'}
    if re.search(r'那个|那片|那块|这块',q) and not re.search(r'建筑|飞机|左|右|上|下',q):
        return {'action':'clarify','question':'请确认要处理的目标和范围。你是指当前目标，还是影像中的其他区域？'}
    if re.search(r'场景|什么地方|介绍.*影像|什么类型',q): return {'action':'scene'}
    if re.search(r'导出|下载',q): return {'action':'export'}
    target=None
    for key,pattern in [('building',r'建筑|房屋|楼|building'),('aircraft',r'飞机|aircraft|airplane'),('road',r'道路|公路|road'),('water',r'水体|河流|water'),('tree',r'植被|树木|tree'),('ship',r'船|ship')]:
        if re.search(pattern,q): target=key; break
    side=None
    for key,pattern in [('left',r'左|left'),('right',r'右|right'),('top',r'上半|上方|top'),('bottom',r'下半|下方|bottom'),('all',r'全图|全部|整幅.*提取|all')]:
        if re.search(pattern,q): side=key; break
    inherited=target is None
    target=target or context.get('target')
    if not target or not re.search(r'提取|分割|建筑|飞机|改|换|占比|面积|统计|左|右|上半|下半|全部|全图|segment|building|aircraft',q):
        return {'action':'clarify','question':'请说明想提取什么目标，以及要分析全图还是某一侧。例如：提取右侧建筑并计算面积占比。'}
    return {'action':'segment','target':target,'side':side or context.get('side','all'), 'inherited_target':inherited,
            'reuse':bool(re.search(r'统计|占比|面积',q) and not re.search(r'提取|分割|改|换',q) and side is None and context.get('target')==target)}

def post_json(url,payload,timeout=120,headers=None):
    req=urllib.request.Request(url,json.dumps(payload).encode(),headers={'Content-Type':'application/json',**(headers or {})})
    with urllib.request.urlopen(req,timeout=timeout) as r: return json.load(r)

def live_plan(query,context,image):
    base=os.environ.get('GEO_AGENT_BASE_URL','').rstrip('/')
    model=os.environ.get('GEO_AGENT_MODEL','')
    if not base or not model: raise ValueError('RemoteAgent 服务未配置。请配置认知核心后使用模型模式。')
    schema='仅输出JSON。action为scene/segment/clarify/export。segment含target(building/aircraft/road/water/tree/ship)、side(all/left/right/top/bottom)。scene含answer；clarify含question。不得估算面积。需求含糊或超出基础半幅空间范围时clarify。'
    response=post_json(base+'/chat/completions',{'model':model,'temperature':0,'max_tokens':650,'messages':[
      {'role':'system','content':'你是遥感实验任务规划器。'+schema},
      {'role':'user','content':[{'type':'text','text':json.dumps({'query':query,'context':context},ensure_ascii=False)},
                               {'type':'image_url','image_url':{'url':'data:image/png;base64,'+image_b64(image)}}]}]},headers={'Authorization':'Bearer '+os.environ.get('GEO_AGENT_API_KEY','EMPTY')})
    raw=response['choices'][0]['message']['content']; match=re.search(r'\{.*\}',raw,re.S)
    plan=json.loads(match.group() if match else raw)
    if plan.get('action') not in ('scene','segment','clarify','export'): raise ValueError('认知核心返回未知动作')
    if plan['action']=='segment' and (plan.get('target') not in TARGETS or plan.get('side') not in SIDES): raise ValueError('认知核心返回不支持的目标或范围')
    if plan['action'] in ('scene','clarify') and not isinstance(plan.get('answer' if plan['action']=='scene' else 'question'),str): raise ValueError('认知核心未返回有效说明')
    return plan

def public_session(s):
    return {k:v for k,v in s.items() if k not in ('lock',)}

def new_session(sample='urban',uploaded=None,name=None):
    sid=uid(); folder=DATA/sid; folder.mkdir()
    if uploaded is None:
        if sample not in SAMPLES: raise ValueError('未知样例')
        image=Image.open(WEB/'assets'/SAMPLES[sample]['file']).convert('RGB')
    else:
        raw=base64.b64decode(uploaded.split(',')[-1],validate=True)
        if len(raw)>12*1024*1024: raise ValueError('影像不能超过12MB')
        image=Image.open(io.BytesIO(raw))
        if image.width*image.height>16_000_000: raise ValueError('首版支持不超过1600万像素的影像')
        image=ImageOps.exif_transpose(image).convert('RGB'); sample=None
    image.save(folder/'original.png')
    s={'id':sid,'sample':sample,'name':(name or SAMPLES[sample]['name'])[:120], 'width':image.width,'height':image.height,
       'image_url':f'/experiments/{sid}/original.png','created_at':now(),'runs':[], 'context':{},
       'source':SAMPLES[sample]['source'] if sample else '用户上传影像', 'lock':threading.Lock()}
    with LOCK: SESSIONS[sid]=s
    save_json(folder/'session.json',public_session(s)); return public_session(s)

def run_task(s,p):
    if not s['lock'].acquire(blocking=False): raise ValueError('当前实验仍在执行，请稍后再试')
    try: return _run_task(s,p)
    finally: s['lock'].release()

def _run_task(s,p):
    query=str(p.get('query','')).strip()
    if not query or len(query)>1800: raise ValueError('请填写不超过1800字的任务')
    mode=p.get('mode','demo')
    if mode not in ('demo','live'): raise ValueError('未知模式')
    previous=next((r for r in s['runs'] if r['id']==p.get('parent_run_id')),None)
    if p.get('parent_run_id') and previous is None: raise ValueError('结果版本不属于当前影像')
    context=previous.get('task',{}) if previous else s['context']
    start=time.perf_counter(); trace=[]; runid=uid(); folder=DATA/s['id']/runid; folder.mkdir()
    def event(name,detail,state='completed',elapsed=0): trace.append({'step':len(trace)+1,'name':name,'detail':detail,'state':state,'duration_ms':round(elapsed,2),'at':now()})
    result={'id':runid,'session_id':s['id'],'parent_run_id':previous['id'] if previous else None,
            'version':len(s['runs'])+1,'query':query,'mode':mode,'created_at':now(),'status':'failed','trace':trace,
            'provenance':{'planner':'演示规则解析器' if mode=='demo' else os.environ.get('GEO_AGENT_MODEL','未配置'),
                          'perception':'手工绘制的近似样例蒙版，非模型预测、非评测真值' if mode=='demo' else 'RemoteSAM服务',
                          'source':s['source'],'image_sha256':hashlib.sha256((DATA/s['id']/'original.png').read_bytes()).hexdigest()}}
    image=Image.open(DATA/s['id']/'original.png').convert('RGB')
    try:
        event('读取实验上下文',f'影像 {s["width"]} × {s["height"]}；'+('继承所选结果条件' if context else '新建任务'))
        if p.get('simulate_failure'):
            event('注入演示故障','用户主动触发的工具不可用测试','failed')
            raise ValueError('故障演示：工具未返回结果。已有版本保留，本次不生成蒙版或统计。')
        t=time.perf_counter(); plan=live_plan(query,context,image) if mode=='live' else parse_task(query,context)
        event('理解任务与选择路径',f'{result["provenance"]["planner"]} · {plan["action"]}',elapsed=(time.perf_counter()-t)*1000)
        result['task']=plan
        if plan['action']=='clarify':
            result.update(status='needs_clarification',message=plan['question']); event('等待条件确认','需求尚未执行','waiting')
        elif plan['action']=='scene':
            if mode=='demo' and not s['sample']: raise ValueError('演示模式无法理解新上传影像。请连接认知核心后切换模型模式。')
            result.update(status='answered',message=plan.get('answer') if mode=='live' else SAMPLES[s['sample']]['scene'])
            event('返回场景说明','此路径未调用分割工具')
        elif plan['action']=='export':
            candidate=previous if previous and previous.get('mask_url') else next((r for r in reversed(s['runs']) if r.get('mask_url')),None)
            if not candidate: raise ValueError('还没有可导出的目标结果，请先完成提取')
            result.update(status='export_ready',message='实验包已可下载，包含影像、蒙版、叠加图、统计与运行记录。',export_url=candidate['export_url'])
            event('准备实验包','引用所选或最近有效结果；不重新推理')
        else:
            target,side=plan['target'],plan['side']; t=time.perf_counter()
            if plan.get('reuse') and previous and previous.get('mask_url') and previous['mode']==mode:
                mask=Image.open(DATA/s['id']/previous['id']/'mask.png').convert('L'); event('复用当前结果','仅重新计算统计；未调用感知工具')
                result['provenance']['perception']=previous['provenance']['perception']
            elif mode=='demo':
                if not s['sample']: raise ValueError('上传影像没有预置标注。请选择模型模式，或返回内置样例体验流程。')
                if target!=SAMPLES[s['sample']]['target']: raise ValueError(f'这个样例只提供{TARGETS[SAMPLES[s["sample"]]["target"]]}的演示标注，不能据此判断其他类别。')
                mask=demo_mask(s['sample']); event('读取预置样例蒙版','近似人工多边形；未调用RemoteSAM',elapsed=(time.perf_counter()-t)*1000)
            else:
                url=os.environ.get('GEO_REMOTESAM_URL','')
                if not url: raise ValueError('RemoteSAM 服务未配置，未执行分割')
                descriptions={'building':'all buildings','aircraft':'all aircraft','road':'all roads','water':'water bodies','tree':'trees and vegetation','ship':'all ships'}
                data=post_json(url,{'task':'referring_seg','image':image_b64(image),'text':descriptions[target]})
                if data.get('status')!='success' or not data.get('mask'): raise ValueError('RemoteSAM未返回有效蒙版')
                mask=Image.open(io.BytesIO(base64.b64decode(data['mask']))).convert('L')
                if mask.size!=image.size: raise ValueError('RemoteSAM蒙版尺寸与原图不一致，请检查服务端坐标还原')
                event('RemoteSAM目标提取','真实服务调用；范围由确定性几何工具约束',elapsed=(time.perf_counter()-t)*1000)
            mask=mask.point(lambda x:255 if x>127 else 0); mask=constrain(mask,side)
            metrics=statistics(mask,side); event('范围筛选与像素统计',f'{SIDES[side]}；整图分母 {image.width*image.height:,} px')
            overlay=image.copy(); tint=Image.new('RGB',image.size,(69,213,152)); colored=Image.blend(image,tint,.48); overlay.paste(colored,(0,0),mask)
            mask.save(folder/'mask.png'); overlay.save(folder/'overlay.png')
            status='completed' if metrics['pixel_area'] else 'needs_review'
            event('结果检查','尺寸对齐、空间范围、统计与文件有效性已检查；语义精度待复核')
            base=f'/experiments/{s["id"]}/{runid}'
            result.update(status=status,metrics=metrics,mask_url=base+'/mask.png',overlay_url=base+'/overlay.png',export_url=f'/api/export/{s["id"]}/{runid}',
              message=f'已生成{SIDES[side]}{TARGETS[target]}区域：{metrics["pixel_area"]:,} px，占整幅影像 {metrics["area_ratio"]*100:.2f}%。'+('这是预置标注的流程演示，尚未进行模型推理。' if mode=='demo' else '请结合叠加图复核目标语义与边界。') if metrics['pixel_area'] else '本次未检出有效区域，需要复核，不能据此断言目标不存在。')
            s['context']={'target':target,'side':side}; save_json(folder/'statistics.json',metrics)
    except Exception as exc:
        result.update(status='failed',message=str(exc) if isinstance(exc,ValueError) else f'服务调用失败（{type(exc).__name__}），请检查模型服务。')
        if not trace or trace[-1]['state']!='failed': event('执行未完成',result['message'],'failed')
    result['duration_ms']=round((time.perf_counter()-start)*1000,2)
    save_json(folder/'result.json',{k:v for k,v in result.items() if k!='trace'}); save_json(folder/'run_log.json',trace)
    s['runs'].append(result); save_json(DATA/s['id']/'session.json',public_session(s)); return result

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw): super().__init__(*a,directory=str(WEB),**kw)
    def json(self,value,status=200):
        raw=json.dumps(value,ensure_ascii=False).encode(); self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        path=self.path.split('?')[0]
        if path=='/api/status': return self.json({'agent_configured':bool(os.environ.get('GEO_AGENT_BASE_URL') and os.environ.get('GEO_AGENT_MODEL')),'sam_configured':bool(os.environ.get('GEO_REMOTESAM_URL')),'samples':SAMPLES,'version':'0.1.0','note':'配置状态不代表服务已通过连通测试'})
        if path.startswith('/api/session/'):
            sid=path.rsplit('/',1)[-1]; s=SESSIONS.get(sid)
            return self.json(public_session(s)) if s else self.json({'error':'实验不存在或服务已重启'},404)
        if path.startswith('/api/export/'):
            parts=path.split('/'); sid,rid=parts[-2:]
            if len(parts)!=5 or not SID_RE.fullmatch(sid) or not SID_RE.fullmatch(rid): return self.send_error(404)
            folder=DATA/sid/rid
            if not (folder/'mask.png').exists(): return self.send_error(404)
            buf=io.BytesIO()
            with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
                z.write(DATA/sid/'original.png','original.png')
                for name in ('mask.png','overlay.png','result.json','statistics.json','run_log.json'): z.write(folder/name,name)
            raw=buf.getvalue(); self.send_response(200); self.send_header('Content-Type','application/zip'); self.send_header('Content-Disposition',f'attachment; filename="geoscope_{rid}.zip"'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); return self.wfile.write(raw)
        if path.startswith('/experiments/'):
            parts=path.strip('/').split('/')
            if len(parts) not in (3,4) or not SID_RE.fullmatch(parts[1]) or (len(parts)==4 and not SID_RE.fullmatch(parts[2])) or parts[-1] not in ('original.png','mask.png','overlay.png'): return self.send_error(404)
            file=DATA.joinpath(*parts[1:])
            if not file.is_file(): return self.send_error(404)
            raw=file.read_bytes(); self.send_response(200); self.send_header('Content-Type','image/png'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); return self.wfile.write(raw)
        return super().do_GET()
    def do_POST(self):
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length<=0 or length>18*1024*1024: raise ValueError('请求大小不支持')
            p=json.loads(self.rfile.read(length))
            if self.path=='/api/session': return self.json(new_session(p.get('sample','urban'),p.get('image'),p.get('name')))
            if self.path=='/api/run':
                s=SESSIONS.get(p.get('session_id'))
                if not s: raise ValueError('实验不存在，请重新选择影像')
                return self.json(run_task(s,p))
            return self.json({'error':'接口不存在'},404)
        except (ValueError,KeyError,Image.UnidentifiedImageError) as exc: return self.json({'error':str(exc)},400)
        except Exception: return self.json({'error':'无法处理影像或请求，请检查输入格式'},500)

if __name__=='__main__':
    port=int(os.environ.get('GEO_PORT','4180'))
    print(f'GeoScope: http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
