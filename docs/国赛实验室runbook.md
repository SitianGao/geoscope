# 国赛实验室 GPU 0 运行手册（GeoScope-AFR 真实推理）

> 适用：实验室机器 `10.199.139.30`（SSH 用户 gst，公钥已配置）。
> 目标：在 **GPU 0** 上启动 RemoteSAM 服务并运行 A3/A4/A5 消融。

## 0. 铁律（队长确认，违反即停）

1. **只用 GPU 0**。GPU 1 上有别人的任务（ning 用户进程），任何命令都带
   `CUDA_VISIBLE_DEVICES=0`，绝不占用 GPU 1。
2. **已有文件一律只读**。权重/tokenizer 只读引用
   `/home/gst/rf-tools-validation/models/...`；本机已有任何代码目录不写不改。
3. **所有新东西都放全新目录 `/home/gst/geoscope-national/`**（代码、venv、产物）。
4. **只用 uv**：不安装 curl、conda；环境一律 `uv` 管理。

## 1. 目录与代码传输

```bash
ssh gst@10.199.139.30 "mkdir -p /home/gst/geoscope-national"
# Windows 本机（Git Bash）上传代码（排除无关大目录）：
scp -r "c:/Users/26310/Desktop/人工智能算法精英比赛2026ver/geoscope-v07-integration/." \
    gst@10.199.139.30:/home/gst/geoscope-national/geoscope-v07-integration/
# 只传必要部分（更快）：
#   geoscope_algorithms/ evaluation/ service_transport.py runtime_config.py
#   evaluation/dataset_manifest.csv services/remotesam/upstream/remotesam.py 及其依赖
```

## 2. 环境（uv，只用托管 CPython 3.10）

```bash
cd /home/gst/geoscope-national
uv venv .venv --python ~/.local/share/uv/python/cpython-3.10.22-linux-x86_64-gnu/bin/python3
source .venv/bin/activate
```

### 2.1 依赖安装（按外网可达性分支）

先探测（不装任何新工具，用 uv 自带 python 的 pip）：

```bash
.venv/bin/python -m pip index versions torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ --timeout 10 --retries 1
.venv/bin/python -m pip index versions torch --index-url https://mirrors.aliyun.com/pypi/simple/ --timeout 10 --retries 1
.venv/bin/python -c "import urllib.request; print(urllib.request.urlopen('https://download.pytorch.org/whl/cu121/torch/', timeout=10).status)"
```

- **分支 A（镜像可达）**：
  ```bash
  uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
  # 服务只装最小推理依赖；不要改用包含训练/评测全家桶的 requirements.txt
  uv pip install -r geoscope-v07-integration/services/remotesam/upstream/requirements-service.txt \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
  uv pip check
  ```
- **分支 B（外网不通）**：在 Windows 本机下载 wheel
  （torch 2.4.1+cu121 from download.pytorch.org + 其余包用
  `pip download -d wheels/ -r requirements.txt`），`scp -r wheels/` 上传，
  再 `uv pip install wheels/*.whl`。禁止在实验室机器装 curl/conda。

### 2.2 cuDNN 初始化失败排查（2026-11-17 已验证）

`torch.cuda.is_available()==True` 但卷积报 `CUDNN_STATUS_NOT_INITIALIZED` 时，
先不要直接判断为驱动过旧。当前服务器驱动 525.89.02 满足 cuDNN 9.1 +
CUDA 12.x 的最低要求。先检查是否混装 CUDA 主版本：

```bash
uv pip list --python .venv/bin/python | grep -E 'nvidia-.*cu13|^nvidia-(cublas|cuda-|cufft|curand|cusolver|cusparse|nvjitlink|nvtx) '
grep -E 'CUDNN_(MAJOR|MINOR|PATCHLEVEL)' \
  .venv/lib/python3.10/site-packages/nvidia/cudnn/include/cudnn_version.h
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"
```

已确认的故障根因是同一 venv 同时存在 `nvidia-cudnn-cu12` 与
`nvidia-cudnn-cu13`。这些包会写入相同的 `nvidia/cudnn/` 目录，cu13 文件可
覆盖 cu12。**同一个 venv 禁止混装 cu11/cu12/cu13 NVIDIA 运行库。**修复后本机
验证组合为 `torch 2.4.1+cu121 / CUDA 12.1 / cuDNN 9.1.0`；最小卷积命令：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c \
  "import torch,torch.nn.functional as F; x=torch.randn(1,3,64,64,device='cuda:0'); w=torch.randn(8,3,3,3,device='cuda:0'); print(F.conv2d(x,w).shape)"
```

若必须修复，先保存 `uv pip freeze`，只清理反向依赖为空的错误主版本包，再按
`importlib.metadata.requires('torch')` 列出的精确版本重装；不要在未查依赖时
批量删除系统 CUDA。临时兜底仍可在进程启动前设置
`torch.backends.cudnn.enabled=False`，但只用于冒烟，不作为最终实验环境。

## 3. 启动 RemoteSAM 服务（GPU 0）

```bash
cd /home/gst/geoscope-national/geoscope-v07-integration/services/remotesam/upstream
CUDA_VISIBLE_DEVICES=0 \
GEO_REMOTESAM_DEVICE=cuda:0 \
GEO_REMOTESAM_CHECKPOINT=/home/gst/rf-tools-validation/models/RemoteSAM/RemoteSAMv1.pth \
GEO_REMOTESAM_TOKENIZER=/home/gst/rf-tools-validation/models/bert-base-uncased \
GEO_REMOTESAM_MAX_IMAGE_MB=40 \
/home/gst/geoscope-national/.venv/bin/python -u remotesam.py
```

验证（另开终端）：

```bash
nvidia-smi   # 确认只有 GPU 0 上有我们刚起的进程
.venv/bin/python -c "import json,urllib.request as u; print(u.urlopen('http://127.0.0.1:6657/ready').status)"
```

## 4. 数据集（数据根目录）

- 若手上有省赛 `selected_20260927/` 目录（WHU + iSAID 两套）→ scp 上传到
  `/home/gst/geoscope-national/data/`；
- 否则用 `prepare_whu_samples.py` / `prepare_isaid_samples.py` 重建
  （需要 WHU/iSAID/DOTA 归档或下载源，先读两个脚本确认源路径）；
- 都没有 → **A3–A5 保持 pending_lab，不伪造运行**。

校验：

```bash
cd geoscope-v07-integration
.venv/bin/python evaluation/verify_dataset_manifest.py \
  --manifest evaluation/dataset_manifest.csv --data-root /home/gst/geoscope-national/data
```

## 5. 运行 A3/A4/A5 消融

先冒烟（2 张图），再全量。A3 用 λ=1（局部替换），A4/A5 按配置 λ 扫 {0.3,0.5,0.7}
（参数选择只在开发验证集，如实记录）。

```bash
cd /home/gst/geoscope-national/geoscope-v07-integration
# 冒烟
.venv/bin/python evaluation/national_final/run_ablation.py \
  --experiment A3 --manifest evaluation/dataset_manifest.csv \
  --data-root /home/gst/geoscope-national/data \
  --base-url http://127.0.0.1:6657/predict \
  --budget-fraction 0.25 --lam 1.0 --limit 2 \
  --outdir /home/gst/geoscope-national/artifacts/a3_smoke \
  --results-csv /home/gst/geoscope-national/results/a3_smoke.csv \
  --summary-json /home/gst/geoscope-national/results/ablation_summary.json
# 全量（A3 预算 25% 与 50% 各一次；A4 λ 三档；A5 用 A4 最优 λ）
```

每张图产物：`artifacts/{image_id}/mask.png + probability_map.png + afr.json`
（A5 另加 `disagreement.png`）。

## 6. 结果回收（回 Windows 提交）

```bash
# Windows 本机：
scp -r gst@10.199.139.30:/home/gst/geoscope-national/results/ \
    "c:/Users/26310/Desktop/人工智能算法精英比赛2026ver/geoscope-v07-integration/evaluation/national_final/results/lab_runs/"
scp -r gst@10.199.139.30:/home/gst/geoscope-national/artifacts/ \
    "c:/Users/26310/Desktop/人工智能算法精英比赛2026ver/geoscope-v07-integration/evaluation/national_final/results/lab_artifacts/"
```

汇总 `ablation_summary.json` 的 status 只有 `completed/partial/failed_all` 才
允许在页面/报告中出现数字；没有 run 的实验保持 `pending_lab`。

## 7. 安全与收尾

- 全程不 touch：`/home/gst/rf-tools-validation`（只读引用权重）、
  `/home/ning/**`（别人的项目）、GPU 1。
- 跑完杀服务进程（只杀我们自己起的 remotesam.py）：
  `pkill -f "geoscope-national.*remotesam.py"` 或用进程 PID 精确 kill。
- 实验室机器上的全部产物都在 `/home/gst/geoscope-national/` 内，
  清理即删该目录，不影响任何已有环境。
