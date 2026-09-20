# RemoteAgent to RemoteSAM integration report

Date: 2026-09-20

Source host: `10.199.139.30` (RemoteAgent, GPU 0 reserved)

Target service: `http://10.199.156.100:6657`

## Readiness

- `GET /ready`: HTTP 200
- service version: 1.1.0
- service device: CUDA 0
- FP16: enabled
- EPOC: disabled
- checkpoint SHA-256:
  `f85dfa044a527f096b9e41eacfe298040d52e77fca642f46dfe1226745e137e7`

## Real-image prediction

The live probe used `web/assets/urban.png` (800 x 600) and the English prompt
`buildings` with `task=referring_seg` and `return_probability=false`.

| Check | Result |
|---|---|
| HTTP response | 200 |
| application status | `success` |
| original size | 800 x 600 |
| output mask size | 800 x 600 |
| decoded mask values | 0 and 255 only |
| empty mask | false |
| foreground pixels | 52,637 |
| foreground ratio | 0.1096604167 |
| service total | 574.53 ms (representative run) |
| client round trip | 843.1 ms (representative run) |
| peak VRAM | 1,836.53 MiB |

The result was also obtained through RemoteAgent's
`ServiceExecutor.execute_structured(...)`. It returned `status=success`,
`service=remotesam`, preserved the complete base64 mask, and retained the
service quality and model metadata for the Harness.

The repeatable probe is `tests/live_remotesam_probe.py`. It writes the input,
decoded `mask.png`, `overlay.png`, service visualization, compact response JSON
and verification JSON to a caller-selected output directory. Generated images
and responses belong under the ignored `experiments/` directory and must not be
committed as runtime truth.
