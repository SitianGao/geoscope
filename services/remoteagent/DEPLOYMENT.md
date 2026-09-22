# GeoScope RemoteAgent handoff

Updated: 2026-09-22

This file records verified source, deployment and integration work. It does
not treat a reachable endpoint as acceptance without exercising the official
model on a real image.

## Current outcome

The gated `AIMGroup/RemoteAgent` snapshot is downloaded at
`E:\models\RemoteAgent` and copied to the assigned laboratory host at
`/home/gst/projects/geoscope-remoteagent/RemoteAgent`. Its four BF16
Safetensors shards contain 729 indexed tensors with no missing or extra shard.
The index reports 8,292,166,656 parameters and 16,584,333,312 bytes of tensor
data.

The official model is live on laboratory GPU 0 through vLLM 0.15.1:

- laboratory endpoint: `http://127.0.0.1:8000`;
- workstation SSH-forwarded endpoint: `http://127.0.0.1:18000`;
- served model: `RemoteAgent` (`Qwen2_5_VLForConditionalGeneration`);
- context length: 4096;
- GPU: RTX 3090 index 0, 23,407 MiB used after startup;
- model loading: 15.63 GiB, with 5.18 GiB available for the KV cache;
- launch log: `experiments/remoteagent-live/remoteagent-vllm.log` locally.

The real-image acceptance probe used `web/assets/urban.png` (SHA-256
`f067e773544d44d3601fdade477f672d7bdaae456cf228fed393292070f75dc3`).
`Describe this remote sensing image.` returned an image-grounded description
of an urban area with mixed buildings, green spaces and streets in 2,732.34
ms. `Extract all buildings in this image using RemoteSAM.` returned a validated
`semantic_segmentation` call with class `building` in 3,360.87 ms. The complete
raw replies and `/v1/models` payload are retained under
`experiments/remoteagent-live/`; these runtime artifacts are intentionally
ignored by Git.

The subsequent full live Harness round trip completed on 2026-09-22 with the
same 800 x 600 source image. The official model selected
`semantic_segmentation(["building"])`; the Harness accepted one supported
class, rebound the image to its current session, called the real RemoteSAM
`semantic_seg` service, checked a 0/255 mask of the original dimensions, saved
the mask/overlay, and measured 134,003 foreground pixels out of 480,000
(ratio 0.2791729167). Its structured feedback omitted the base64 image and
included those deterministic measurements. RemoteAgent then returned
`<answer>The pixel count of buildings in the image is 134,003, and the area
ratio is approximately 0.2792.</answer>`. Total run time was 4,734.34 ms.
The source PNG is re-encoded on upload, so the session file has a different
byte hash; a pixel-by-pixel comparison confirmed the image content is identical.
The local ignored directory `experiments/remoteagent-harness-live-final/`
contains the full live JSON trace, mask, overlay, statistics, and run log.

An earlier live attempt produced a malformed tool syntax and was correctly
rejected without execution. Another attempt returned accurate measurements
without the mandatory `<answer>` tag and was marked `needs_review`; the mask
and deterministic statistics remained available. These are protocol failures,
not evidence of missing inference. The final successful run followed a
feedback prompt that explicitly requires `<answer>` and exact measurements.

`docs/model_manifest.json` records source/model metadata, size and SHA-256 for
the index and all four shards. The hashes were independently verified against
the copies on the laboratory host. ModelScope's exact repository commit was
not retained by the download client and is recorded as unknown rather than
guessed; the hashes identify the downloaded bytes.

The RemoteAgent side now supports both workflows without giving two programs
control of the same execution loop:

1. `RemoteAgent.plan(...)` sends the real image plus task/context to the model,
   parses the official `<answer>` / `T_call(...)` format, validates the tool,
   binds image arguments to the current asset, and returns without execution.
   This is the recommended GeoScope Harness interface.
2. `RemoteAgent.run(...)` remains the independent self-executing validation
   path. It now also sends the real image and retains complete structured tool
   responses in `AgentResult.tool_results`.
3. `RemoteAgent.continue_with_tool_result(...)` lets the Harness feed execution
   status and deterministic statistics back to the cognitive model. Binary
   masks/images are not copied into the language-model prompt, but remain in
   the caller-owned result object.

Only deployed tools should be passed through `allowed_tools`. For the first
GeoScope milestone use:

```python
{
    "referring_expression_segmentation",
    "semantic_segmentation",
}
```

Any call outside that set returns `status="rejected_tool"`. The Harness must
still validate task arguments and remain the sole authority for session assets,
pixel statistics, versions, export, and file access.

## Decision contract

`plan(...)` returns an `AgentResult` with these relevant fields:

| Field | Meaning |
|---|---|
| `status="completed"` | Official `<answer>` response; use `text` |
| `status="tool_call"` | Execute `tool_call.name` with `tool_call.arguments` |
| `status="rejected_tool"` | Model selected a tool not exposed at runtime |
| `status="unparsed"` | Output matched neither official answer nor tool syntax; do not execute |
| `raw_response` | Original model output for the run log |
| `history` | Parsed round-level trace without embedded image bytes |

The supplied image path is resolved and checked before inference. For all
single-image official tools, a path emitted by the model is overwritten with
that checked current path. This prevents the model from selecting another
local file.

## Recommended Harness sequence

```text
upload/current version
        |
        v
RemoteAgent.plan(real image, request, context)
        |
        +-- completed ------> show/log direct answer
        |
        +-- rejected/unparsed -> fail closed; preserve prior valid result
        |
        +-- tool_call ------> Harness validates allowlist and asset binding
                                  |
                                  v
                              RemoteSAM /predict
                                  |
                                  v
                         Harness checks mask size,
                         applies pixel scope, computes statistics,
                         stores files and version
                                  |
                                  v
                    continue_with_tool_result(summary + numbers)
                                  |
                                  v
                           final answer and run log
```

For “left/right/top/bottom”, GeoScope should continue to extract the whole
semantic target and apply the scope with its deterministic geometry tool.
Directions are image-pixel directions, not geographic east/west. Numeric areas
must come from the Harness, not from the model.

## Local source audit

| Item | Observed state |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB |
| NVIDIA driver | 561.00 |
| Active Python | 3.13.9 (`E:\\anaconda3\\python.exe`) |
| Conda | 26.1.1 |
| Existing RemoteAgent environment | Local archive only; runtime is deployed on the laboratory host |
| Official model/checkpoint files | `E:\models\RemoteAgent`, four complete Safetensors shards |
| Supplied source Git metadata | None; archive is not a Git checkout |

## Assigned remote deployment host

The assigned laboratory host is Ubuntu at `10.199.139.30` using account `gst`.
It has two RTX 3090 GPUs. The team selected GPU 0. It was rechecked before
deployment and showed only 37 MiB used and 0% utilization, with no compute
process. RemoteAgent must therefore use `CUDA_VISIBLE_DEVICES=0`; GPU 1 is
outside this deployment.

Remote storage observed: about 121 GiB free on `/` and 315 GiB free on
`/home`. The deployment working directory is
`/home/gst/projects/geoscope-remoteagent`; the official checkpoint is now in
its `RemoteAgent/` subdirectory.

The RemoteSAM contract supplied by teammate B is:

- prediction: `POST http://10.199.156.100:6657/predict`;
- health: `GET http://10.199.156.100:6657/health`;
- readiness: `GET http://10.199.156.100:6657/ready`;
- request: `{"task":"referring_seg","image":"<base64>","text":"building"}`.

The RemoteAgent host can reach this service. On 2026-09-20, `/ready` returned
HTTP 200 with service version 1.1.0, CUDA device 0, FP16 enabled and checkpoint
SHA-256 `f85dfa044a527f096b9e41eacfe298040d52e77fca642f46dfe1226745e137e7`.
No additional Windows firewall rule was needed. `.env.geoscope.example`
records the base URL without credentials.

A live `referring_seg` request was then executed from the assigned RemoteAgent
host using the repository's `web/assets/urban.png` and prompt `buildings`.
RemoteSAM returned HTTP 200 and `status=success`; input and mask were both
800 x 600, decoded mask values were exactly 0 and 255, and the mask was not
empty (52,637 foreground pixels, ratio 0.1096604167). A representative run
reported 574.53 ms service time, 843.1 ms client round trip and 1,836.53 MiB
peak VRAM. The same call through `ServiceExecutor.execute_structured(...)`
retained the complete 4,400-character base64 mask and model metadata. Local
artifacts are saved under `experiments/remotesam-live/result/`; that directory
is not committed.

The remote shell exports stale HTTP/HTTPS proxy variables pointing to
`127.0.0.1:7890`, where no proxy service is listening. A temporary authenticated
SSH reverse SOCKS tunnel is used for package and model downloads without
disabling TLS verification. The isolated Conda environment now exists at
`/home/gst/miniconda3/envs/remoteagent` with Python 3.10.21 and ModelScope
1.40.1. PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128 and torchaudio
2.9.1+cu128 are installed only in that environment. CUDA discovery was
verified with `CUDA_VISIBLE_DEVICES=0` on an RTX 3090. vLLM 0.15.1,
Transformers 4.57.6 and OpenCV 5.0.0 are installed, import successfully, and
`pip check` reports no broken requirements.

The designated private repository is `SitianGao/geoscope`, and all RemoteAgent
changes belong on `feature/remoteagent` before a pull request to `main`. Model
weights, credentials, caches and Conda environments must never be committed.

The ModelScope repository `AIMGroup/RemoteAgent` reports a 16.60 GB snapshot
updated on 2026-07-07. Access was approved and the snapshot was downloaded to
the E drive before transfer to the laboratory host. Credentials are not stored
in this repository.

The official README currently asks for a separate Python 3.10 environment,
PyTorch 2.9.1 CUDA 12.8, vLLM 0.15.1, OpenAI client dependencies, and the
`AIMGroup/RemoteAgent` model. Do not install those packages into the current
Python 3.13 base environment. The provided start script is Bash-based, so a
Linux/WSL2 or separate Linux GPU host is the lowest-risk deployment target.

The checkpoint fits GPU 0 and both image/task acceptance cases pass. A
reachable `/v1/models` endpoint alone is still not considered acceptance; the
saved real-image probe is the authoritative run record.

## Verified checks

Run from the repository root:

```powershell
$env:PYTHONPYCACHEPREFIX = Join-Path $env:TEMP 'geoscope-remoteagent-pycache'
python -m unittest discover -s tests -v
python -m compileall -q services/remoteagent/upstream/remoteagent tests
```

The temporary bytecode location avoids archive/sandbox directory metadata from
blocking creation of local `__pycache__` folders; it does not affect runtime.

The offline tests verify:

- the initial request contains an actual image data URL;
- context is included;
- model paths are rebound to the current image;
- unavailable tools fail closed;
- the complete RemoteSAM mask response is retained;
- mask bytes are omitted from the language-model feedback prompt.

The repository currently passes 34 tests, including regression tests for a
valid `T_call(...)` wrapped by the checkpoint in an `<answer>` tag. Unit tests
use controlled responses; the separate live probe described above records the
official model inference. The live RemoteSAM probe and the real
`ServiceExecutor` -> RemoteSAM call also pass.

## Operational follow-up

- Keep at least one success and one failure case; never replace missing results
  with the Demo's hand-authored polygon.
- Revoke the ModelScope token exposed during setup and create a replacement for
  future downloads.

## Modified files

- `remoteagent/core/inference.py`
- `remoteagent/run_remoteagent_vllm.sh`
- `remoteagent/core/types.py`
- `remoteagent/core/__init__.py`
- `remoteagent/services/executor.py`
- `remoteagent/utils/image.py`
- `remoteagent/cli.py`
- `remoteagent/__init__.py`
- `tests/test_geoscope_workflow.py`
- `tests/test_remoteagent_workflow.py`
- `tests/live_remoteagent_probe.py`
- `README.md`
- `GEOSCOPE_REMOTEAGENT_HANDOFF.md`
