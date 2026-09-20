# RemoteAgent to RemoteSAM integration report

Verification date: 2026-09-20

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
