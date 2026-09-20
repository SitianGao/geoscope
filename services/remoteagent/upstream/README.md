

﻿<div align="center">

# [RemoteAgent: Bridging Vague Human Intents and Earth Observation with RL-based Agentic MLLMs]()



[Liang Yao (姚亮)*](https://1e12leon.top/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Shengxiang Xu(徐圣翔)*](https://xushengxianggg.github.io/) 
<img src="assets/SEU.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Fan Liu (刘凡)](https://multimodality.group/author/%E5%88%98%E5%87%A1/) ✉ 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Chuanyi Zhang (张传一)](https://ai.hhu.edu.cn/2023/0809/c17670a264073/page.htm) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;

[Bishun Yao (姚必顺)](https://multimodality.group/author/%E7%8E%8B%E7%BF%8C%E9%AA%8F/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Rui Min (闵锐)](https://multimodality.group/author/%E9%97%B5%E9%94%90/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Yongjun Li (李勇俊)]() 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;

[Chaoqian Ouyang(欧阳超前)]() 
<img src="assets/SYU.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Shimin Di (邸世民)](https://cs.seu.edu.cn/shimindi/main.htm) 
<img src="assets/SEU.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Min-Ling Zhang (张敏灵)]() 
<img src="assets/SEU.png" alt="Logo" width="15">

\*  *Equal Contribution*    ✉ *Corresponding Author*

Model : [RemoteAgent](https://modelscope.cn/models/AIMGroup/RemoteAgent) 
</div>


## News
- **2026/8/12**: Our paper is accepted by IEEE TGRS!
- **2026/4/9**: Welcome to RemoteAgent! The preprint of our paper is available. Dataset and codes will be open-sourced at this repository.



## Introduction


Earth Observation (EO) systems are essentially designed to support domain experts who often express their requirements through vague natural language rather than precise, machine-friendly instructions. Depending on the specific application scenario, these vague queries can demand vastly different levels of visual precision. Consequently, a practical EO AI system must bridge the gap between ambiguous human queries and the appropriate multi-granularity visual analysis tasks, ranging from holistic image interpretation to fine-grained pixel-wise predictions.
While Multi-modal Large Language Models (MLLMs) demonstrate strong semantic understanding, their text-based output format is inherently ill-suited for dense, precision-critical spatial predictions. Existing agentic frameworks address this limitation by delegating tasks to external tools, but indiscriminate tool invocation is computationally inefficient and underutilizes the MLLM's native capabilities.
To this end, we propose RemoteAgent, an agentic framework that strategically respects the intrinsic capability boundaries of MLLMs. To empower this framework to understand real user intents, we construct VagueEO, a human-centric instruction dataset pairing EO tasks with simulated vague natural-language queries. By leveraging VagueEO for reinforcement fine-tuning, we align an MLLM into a robust cognitive core that directly resolves image- and sparse region-level tasks. Consequently, RemoteAgent processes suitable tasks internally while intelligently orchestrating specialized tools via the Model Context Protocol exclusively for dense predictions. Extensive experiments demonstrate that RemoteAgent achieves robust intent recognition capabilities while delivering highly competitive performance across diverse EO tasks. 


## Repository Structure

```text
remoteagent/
  config/       # defaults and tool schemas
  core/         # RemoteAgent + AgentResult
  llm/          # vLLM OpenAI-compatible client
  parsing/      # ToolCallParser for T_call parsing
  services/     # ServiceExecutor and task mappings
  utils/        # HttpUtils, ImageUtils, TextUtils
  prompts/      # default system prompt
  eval_common.py
  cli.py
servers/        # external tool service implementations
test/           # evaluation and benchmark scripts
```
## Main Classes

- `RemoteAgent`: core inference orchestrator
- `AgentResult`: output container (`text`, `history`)
- `ToolCallParser`: parse `T_call(...)` to structured arguments
- `ServiceExecutor`: route tool requests to service backends
- `VLLMChatClient`: chat/model calls to vLLM endpoint
- `RemoteAgentCLI`: CLI parser and entrypoint

## Supported Tool Names

- `object_detection`
- `referring_expression_segmentation`
- `semantic_segmentation`
- `binary_change_detection`
- `semantic_change_detection`
- `building_damage_assessment`
- `oriented_object_detection`
- `crossearth_semantic_segmentation`
- `contour_extraction`
- `region_contour_extraction`
- `subobject_contour_extraction`
- `region_subobject_contour_extraction`


## Installation

Create the `remoteagent` environment:

```bash
conda create -n remoteagent python=3.10 -y
conda activate remoteagent

python -m pip install -U pip setuptools wheel

python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

python -m pip install \
  vllm==0.15.1 \
  openai \
  nvidia-nvshmem-cu12 \
  nvidia-cusparselt-cu12

python -m pip install opencv-python
```

## Configuration

Before using any external tools, make sure each tool service is fully deployed with all required model files, checkpoints, and runtime dependencies.


### Tool service URLs

- `REMOTE_API_URL`
- `CHANGE3D_API_URL`
- `SM3DET_API_URL`
- `CROSSEARTH_API_URL`
- `SKYSENSE_DET_API_URL`
- `DIRECTSAM_API_URL`


## Usage

### 1) Start vLLM backend

```bash
conda activate remoteagent
export PYTHONNOUSERSITE=1
unset PYTHONPATH
bash ./remoteagent/run_remoteagent_vllm.sh RemoteAgent 8000
```

### 2) Start all tool services (optional)

Run each command in a separate terminal:

```bash
# Terminal A (RemoteSAM)
conda activate RemoteSAM
python servers/RemoteSAM/remotesam.py

# Terminal B (Change3D)
conda activate Change3D
python servers/Change3D/change3d.py

# Terminal C (SM3Det)
conda activate SM3Det
python servers/SM3Det/sm3det.py

# Terminal D (CrossEarth)
conda activate CrossEarth
python servers/CrossEarth/crossearth.py

# Terminal E (SkySense)
conda activate skysense
python servers/SkySense/skysense.py

# Terminal F (DirectSAM)
conda activate subobjects
python servers/DirectSAM/directsam.py
```

### 3) Run with CLI (new terminal)

```bash
conda activate remoteagent
export PYTHONNOUSERSITE=1
unset PYTHONPATH
python -m remoteagent.cli --image_path "/path/to/demo.jpg" "Describe this image."
```

### 4) Run with Python API (from environment)

```python
from remoteagent import RemoteAgent

agent = RemoteAgent.from_env(max_rounds=3, max_tokens=512)
result = agent.run(
    query="Describe objects and scene in this EO image.",
    image_path="/path/to/demo.jpg",
)

print(result.text)
print(result.history)  # round-level logs
```

### 5) Run with Python API (manual configuration)

```python
import os

from remoteagent import RemoteAgent

api_urls = {
    "remotesam": os.environ.get("REMOTE_API_URL"),
    "change3d": os.environ.get("CHANGE3D_API_URL"),
    "sm3det": os.environ.get("SM3DET_API_URL"),
    "crossearth": os.environ.get("CROSSEARTH_API_URL"),
    "skysense_det": os.environ.get("SKYSENSE_DET_API_URL"),
    "directsam": os.environ.get("DIRECTSAM_API_URL"),
}

agent = RemoteAgent(
    vllm_url=os.environ.get("VLLM_URL", "http://localhost:8000"),
    model_name=None,
    api_urls=api_urls,
    max_rounds=3,
)

result = agent.run("Find major roads.", "/path/to/demo.jpg")
print(result.text)
```

## Evaluation (Example)

Run one DIOR evaluation example (tool/vLLM URLs are optional here; script defaults are used when omitted):

```bash
python test/dior.py
```

## Extending with New Services

### Add a new external service (example: `mytool`)

When you add a brand-new backend service under `servers/`, update the agent side in this order.

1) Add default port and environment variable key in `remoteagent/config/defaults.py`:

```python
SERVICE_PORTS["mytool"] = 6660
ENV_URL_KEYS["mytool"] = "MYTOOL_API_URL"
```

2) Add a CLI argument and default URL fallback in `remoteagent/cli.py`:

```python
p.add_argument(
    "--mytool_url",
    type=str,
    default=RemoteAgentCLI._default_service_url("mytool"),
)
```

and include it in `api_urls`:

```python
api_urls["mytool"] = args.mytool_url
```

3) Register tool routing in `remoteagent/config/tools_schema.py`:

```python
MCP_TOOLS["mytool_new_task"] = ["image_path", "classes"]
TOOL_TO_SERVICE["mytool_new_task"] = "mytool"
```

4) Add task mapping (if needed) in `remoteagent/services/mappings.py`:

```python
MYTOOL_TOOL_TO_TASK = {"mytool_new_task": "new_task"}
```

5) Implement service call in `remoteagent/services/executor.py`:
- route `service == "mytool"` in `execute(...)`
- add `_call_mytool(...)` to build payload and parse response

6) Update `remoteagent/prompts/prompt.txt` so the model knows when/how to call the new tool.

7) (Optional) Set `MYTOOL_API_URL` in your shell to override the default port:

```bash
export MYTOOL_API_URL="http://127.0.0.1:6660"
```

PowerShell:

```powershell
$env:MYTOOL_API_URL="http://127.0.0.1:6660"
```

After these changes, you can usually run CLI without passing `--mytool_url` because it will auto-fallback to `SERVICE_PORTS["mytool"]`.




## Acknowledge
- Code in this repository is built on [MS-SWIFT](https://github.com/modelscope/ms-swift). We'd like to thank the authors for open sourcing their project.

## Contact
Please Contact yaoliang@hhu.edu.cn


## Cite
If you find this work useful, please cite our papers as:
```bibtex
@misc{yao2026RemoteAgent,
      title={RemoteAgent: Bridging Vague Human Intents and Earth Observation with RL-based Agentic MLLMs}, 
      author={Liang Yao and Shengxiang Xu and Fan Liu and Chuanyi Zhang and Bishun Yao and Rui Min and Yongjun Li and Chaoqian Ouyang and Shimin Di and Min-Ling Zhang},
      year={2026},
      eprint={2604.07765},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.07765}, 
}
@misc{yao2025RemoteSAM,
      title={RemoteSAM: Towards Segment Anything for Earth Observation}, 
      author={Liang Yao and Fan Liu and Delong Chen and Chuanyi Zhang and Yijun Wang and Ziyun Chen and Wei Xu and Shimin Di and Yuhui Zheng},
      year={2025},
      eprint={2505.18022},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.18022}, 
}
```
## Badge
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/1e12leon-remoteagent-badge.png)](https://mseep.ai/app/1e12leon-remoteagent)
