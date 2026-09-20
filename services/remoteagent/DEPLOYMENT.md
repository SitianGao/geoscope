# GeoScope RemoteAgent handoff

Updated: 2026-09-20

This file records verified source, deployment and integration work. It does
not claim that the gated official RemoteAgent model has run until its
ModelScope download is approved and the vLLM endpoint is exercised.

## Current outcome

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
| Supplied model/checkpoint files | None found (`.safetensors/.bin/.pt/.pth/.ckpt`) |
| Supplied source Git metadata | None; archive is not a Git checkout |

## Assigned remote deployment host

The assigned laboratory host is Ubuntu at `10.199.139.30` using account `gst`.
It has two RTX 3090 GPUs. The team selected GPU 0. It was rechecked before
deployment and showed only 37 MiB used and 0% utilization, with no compute
process. RemoteAgent must therefore use `CUDA_VISIBLE_DEVICES=0`; GPU 1 is
outside this deployment.

Remote storage observed: about 121 GiB free on `/` and 315 GiB free on
`/home`. The deployment working directory is
`/home/gst/projects/geoscope-remoteagent`. No RemoteAgent checkpoint was found
under the readable `/home`, `/data`, or `/mnt` paths during the audit.

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
verified with `CUDA_VISIBLE_DEVICES=0` on an RTX 3090. The remaining vLLM
installation requires the workstation proxy to be online again; direct PyPI
TLS is intercepted by the campus gateway and is not bypassed.

The designated private repository is `SitianGao/geoscope`, and all RemoteAgent
changes belong on `feature/remoteagent` before a pull request to `main`. Model
weights, credentials, caches and Conda environments must never be committed.

The ModelScope repository `AIMGroup/RemoteAgent` is visible and reports a
16.60 GB snapshot updated on 2026-07-07, but it uses application-gated download.
Anonymous metadata access works while the file-tree API returns 404 until an
approved account token is supplied. The download target remains
`/home/gst/projects/geoscope-remoteagent/RemoteAgent`; no model bytes are routed
through or stored on the local workstation.

The official README currently asks for a separate Python 3.10 environment,
PyTorch 2.9.1 CUDA 12.8, vLLM 0.15.1, OpenAI client dependencies, and the
`AIMGroup/RemoteAgent` model. Do not install those packages into the current
Python 3.13 base environment. The provided start script is Bash-based, so a
Linux/WSL2 or separate Linux GPU host is the lowest-risk deployment target.

After download approval, record the exact model revision/checksum and verify
that the checkpoint fits GPU0. A reachable `/v1/models` endpoint does not by
itself satisfy the image/task acceptance criteria.

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

The repository currently passes 18 tests: 13 Harness tests and 5 RemoteAgent
workflow tests. The latter use controlled fake responses; they are not official
model inference. Separately, the live RemoteSAM probe and the real
`ServiceExecutor` -> RemoteSAM call described above both pass.

## Remaining acceptance work

- Complete the ModelScope gated-download approval, download the official
  checkpoint, and record its revision/source/checksums.
- Start vLLM and save `/v1/models`, launch command, GPU memory, latency, and the
  raw response for a new non-demo image.
- Verify “describe this image” returns an image-grounded direct answer.
- Verify “extract buildings” returns the official RemoteSAM `T_call` format.
- Feed the verified RemoteSAM mask/statistics metadata into the full Harness
  session after the official RemoteAgent model is online.
- Keep at least one success and one failure case; never replace missing results
  with the Demo's hand-authored polygon.

## Modified files

- `remoteagent/core/inference.py`
- `remoteagent/core/types.py`
- `remoteagent/core/__init__.py`
- `remoteagent/services/executor.py`
- `remoteagent/utils/image.py`
- `remoteagent/cli.py`
- `remoteagent/__init__.py`
- `tests/test_geoscope_workflow.py`
- `README.md`
- `GEOSCOPE_REMOTEAGENT_HANDOFF.md`
