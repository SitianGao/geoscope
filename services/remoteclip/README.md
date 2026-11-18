# RemoteCLIP 真实检索服务

2026-11-18 集成后可用一条命令启动真实 CLIP 和 B 工作台（已安装隔离环境及可信权重时）：

```powershell
.venv-remoteclip/Scripts/python.exe scripts/start_national.py
```

打开 `http://127.0.0.1:4180/?workbench=national`。自动创建缺失的样例图库，重建索引；Ctrl+C 停止两个服务。已有自定义图库可通过 `--gallery` 指定。Agent/SAM 仍读取现有 `.env`，不会由 CLIP 脚本替代。`scripts/run_remoteclip_acceptance.py --browser` 可自动启停临时端口服务并验收真实页面，不拦截网络响应；需设置 `PLAYWRIGHT_PATH` 指向已安装的 playwright 包。

上游：[RemoteCLIP 官方项目](https://github.com/ChenDelong1999/RemoteCLIP)、[官方权重](https://huggingface.co/chendelong/RemoteCLIP)。网络结构、训练和权重属于上游；本项目提供服务、影像绑定、索引、检索和溯源。

从仓库根目录运行（PowerShell）：

```powershell
python -m venv .venv-remoteclip
.venv-remoteclip/Scripts/python.exe -m pip install -r services/remoteclip/requirements.txt
.venv-remoteclip/Scripts/python.exe -m services.remoteclip.download_weights
.venv-remoteclip/Scripts/python.exe -m services.remoteclip.prepare_gallery
.venv-remoteclip/Scripts/python.exe -m services.remoteclip.service --weights weights/RemoteCLIP-RN50.pt --sha256 11657d33a87fc197e19a7654a76a632415ebea42feb8a61025ef389f8990c0f9 --architecture RN50 --gallery outputs/remoteclip-gallery.json
```

在另一终端启用工作台：

```powershell
$env:GEO_REMOTECLIP_URL='http://127.0.0.1:4190'
.venv-remoteclip/Scripts/python.exe national_server.py
```

默认 CPU，可设置 `--device cuda`（需安装对应 PyTorch）。CPU 同样执行真实权重推理。服务监听本机，模型启动失败仍保留 `/ready` 返回实际故障。不得用随机模型替代。

`GET /ready`：`ready` 表示模型已加载；`index_ready` 表示图库当前可检索。`GET /model` 返回模型信息。`POST /search` 接收 `{"query":"airport with airplanes","top_k":5}`。只返回余弦相似度，不使用 softmax 或称作概率。候选 `image_id` 即已有 session ID。

索引由既有 `experiments/{image_id}/original.png` 构建，不接受查询者传路径；`--image-root` 为操作者部署配置。生成的 `manifest.json` 记录原图 SHA256、权重 SHA256、OpenCLIP/PyTorch 版本、预处理、tokenizer、embedding/index 版本。启动不带 `--gallery` 时复用索引；每次检索验证原图和索引，改变权重、预处理或原图后必须重建。`.npy` 禁用 pickle。

RN50 固定 SHA256 来自官方 LFS 指针：`https://huggingface.co/chendelong/RemoteCLIP/raw/main/RemoteCLIP-RN50.pt`。官方网络不可达时下载脚本支持 `--endpoint https://hf-mirror.com`；镜像内容必须与此官方哈希相同才被激活。其他架构须独立核对其官方哈希后使用。

负例：缺权重/依赖、哈希不符、加载失败、缺索引、空图库、原图变化、空查询、非法 top-k、OOM 均无固定排序回退。`requirements-tested.txt` 是本次本地环境快照；主程序无需安装这些模型依赖。权重、虚拟环境、图库和索引均不入 Git。

真实模型测试：设置 `REMOTECLIP_TEST_WEIGHTS`、`REMOTECLIP_TEST_SHA256`、`REMOTECLIP_TEST_ARCH=RN50` 后执行 `python -m unittest discover -s tests -p test_remoteclip_integration.py -v`。未配置时明确跳过真实测试；合成向量测试不构成模型证据。
