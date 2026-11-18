# 既有 RemoteAgent 的备用 serving 入口

`openai_service.py` 仅把用户已有的官方 `AIMGroup/RemoteAgent` 检查点通过 OpenAI 兼容接口暴露；不替代 `agent_bridge.py` 的规划校验，也不新增模型或改权重。原 vLLM 部署仍可继续使用。本次服务器旧 vLLM 环境/模型副本已不存在，因此在 A 的新目录部署此备用运行时以完成真实联调。

在**服务器独立环境**通过 uv 安装与模型 config 声明匹配的版本：Transformers 4.57.1、tokenizers 0.22.1、accelerate 1.10.1；本次只读复用已有 PyTorch 2.4.1+cu121。不要把这些包安装到队长的 SAM 环境。

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$AGENT_PYTHON" -m services.remoteagent.openai_service \
  --model-path "$AGENT_CHECKPOINT" --port 8000
```

工作台环境使用 `GEO_AGENT_BASE_URL=http://127.0.0.1:8000/v1`、`GEO_AGENT_MODEL=RemoteAgent`。只绑定服务器 loopback；不提供公网访问、任意 URL 图片下载、流式响应或凭据管理。

`GET /ready`、`GET /v1/models` 报告真实加载状态；`POST /v1/chat/completions` 接受文本及 data:image 内联图片。使用权重保存的 slow processor（use_fast=False）、最多 1024 个图像 patch token 的预处理上限、BF16/SDPA、确定性 generate。单个请求生成上限 2048 tokens，输入加生成上限 16384 tokens，超限明确报错；并发请求串行执行，防止同时占用显存。接口为本地受信工作台使用，适配行为不代表与 vLLM 性能相同。

部署前校验四个 Safetensors 分片：729 个张量、索引映射、文件完整长度和 16,584,333,312 字节张量数据。来源清单和文件 SHA256 见 `docs/evidence/server/agent_inventory.json`。权重来自用户之前保存的 `E:\models\RemoteAgent`，与 ModelScope 页面文件名及大小相符；仅复制到服务器，不在用户电脑运行模型。页面标记为申请制，未尝试绕过下载权限。

运行记录可能包含开始时的依赖/版本失败，这些记录保留用于审计。最终验收以完整真实链路 JSON 和成功日志为准；不以端口可达代替推理成功。
