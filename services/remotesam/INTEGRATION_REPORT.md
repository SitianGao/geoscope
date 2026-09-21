# RemoteAgent to RemoteSAM integration report

Verification date: 2026-09-21

## Result

The RemoteAgent laboratory host successfully called the independently hosted
RemoteSAM service over the laboratory network.

- RemoteAgent host: `10.199.139.30`
- RemoteSAM host: `10.199.156.100:6657`
- `GET /ready`: HTTP 200, `status=ready`, CUDA device 0, FP16 enabled
- `POST /predict`: HTTP 200, `status=success`
- Input image: 800 x 600
- Output mask: 800 x 600, PNG values limited to 0 and 255
- Foreground pixels: 52,637
- RemoteAgent preserved the complete mask and model metadata
- Remote-side contract tests: 5 passed

The corresponding RemoteAgent-side evidence and workflow notes are recorded in
`../remoteagent/INTEGRATION_REPORT.md`. This result verifies transport and data
alignment; it is not an accuracy benchmark. No credentials, weights, base64
payloads, generated masks, or machine-local absolute paths are committed.

## Captain revalidation on 2026-09-21

The service was independently probed again from the captain's workstation after
the final team merge (`97a57426e7825e312e372cda91c5c5301881db65`).

- `GET /ready`: HTTP 200, `status=ready`, service version `1.1.0`, CUDA device 0,
  FP16 enabled, EPOC disabled.
- The reported checkpoint SHA-256 remained
  `f85dfa044a527f096b9e41eacfe298040d52e77fca642f46dfe1226745e137e7`.
- `POST /predict` with `web/assets/urban.png` and prompt `buildings`: HTTP 200,
  `status=success`.
- Input and decoded mask were both 800 x 600. Mask values were exactly 0 and
  255; 52,637 pixels were foreground (ratio 0.1096604167), so the result was not
  an empty or fixed failure mask.
- Service-reported time was 437.83 ms (402.97 ms inference); client round trip
  was 1,267.24 ms and peak VRAM was 1,836.53 MiB. These are single-run
  observations, not a speed benchmark.
- Root integration tests passed 43/43 and the RemoteSAM service tests passed
  5/5 using the documented test commands.

An invalid base64 image returned HTTP 200 with `status=error`, an explicit
`ValueError: Invalid base64 image data` message, and no `mask`, `result_image`,
`probability_map`, or `masks` field. The current GeoScope caller checks the
application status and therefore fails closed. Returning HTTP 400 for malformed
input would improve API semantics, but it does not block the current integration
contract, which explicitly permits either an HTTP error or `status=error`.

Teammate B also supplied a separate 800 x 800 West Lake image run with prompt
`water`, reporting an aligned binary mask, 68,928 foreground pixels, mean
foreground confidence 0.900293, 972.56 ms inference, 1,012.75 ms service time,
and 1,837.14 MiB peak VRAM. The source image, decoded mask, overlay, and compact
JSON should remain together as local handoff artifacts if that case is used in
the final presentation; the screenshot alone is not an accuracy benchmark.

## Acceptance boundary

This evidence accepts teammate B's RemoteSAM component and the transport
contract used by the Harness. It does not prove that the official RemoteAgent
checkpoint has made a live tool decision. That final end-to-end acceptance still
requires the gated RemoteAgent weights, a running vLLM endpoint, and a saved
Agent decision followed by this verified RemoteSAM call.
