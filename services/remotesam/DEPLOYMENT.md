# RemoteSAM deployment record

Verification date: 2026-09-20

## Verified runtime

- Conda environment: `geoscope-remotesam` (Python 3.10)
- PyTorch: `2.4.1+cu121`
- Torchvision: `0.19.1+cu121`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
- Service bind address: `0.0.0.0:6657`
- FP16 autocast: enabled
- EPOC: disabled for the first referring-segmentation integration

## Verified model

- Source: <https://huggingface.co/1e12Leon/RemoteSAM>
- File: `RemoteSAMv1.pth`
- Size: `2,566,704,235` bytes
- Repository revision: `a6d09f44bac5b64be7f74980b15afb2a749b5e92`
- LFS SHA-256: `f85dfa044a527f096b9e41eacfe298040d52e77fca642f46dfe1226745e137e7`
- Checkpoint epoch: 58
- Checkpoint tensors: 1107
- Unexpected keys: 0
- Missing key: `text_encoder.embeddings.position_ids`, a generated Transformers buffer rather than a trained parameter

The checkpoint was inspected and loaded with PyTorch `weights_only=True`, with
only `argparse.Namespace` allowlisted. The local SHA-256 matched the official
Hugging Face LFS metadata. Model weights are not stored in this repository.

## Real smoke test

- Development image size: 800 x 800
- Prompt: `the airplane on the right`
- Result: success; the right airplane was selected and the output aligned with the input
- Foreground pixels: 24,149
- Foreground ratio: 0.0377328125
- Mean foreground confidence: 0.974341
- Inference time: 746.10 ms
- Total request time: 783.32 ms
- Peak allocated VRAM: 1,837.14 MB

The generated image, mask, overlay, probability map, and full base64 response
remain local test artifacts and are intentionally excluded from Git. These
figures describe one development-machine run, not a performance benchmark.

## Start the service on Windows

```powershell
powershell -ExecutionPolicy Bypass -File services/remotesam/start_service.ps1 `
  -CheckpointPath "D:\models\RemoteSAM\RemoteSAMv1.pth" `
  -TokenizerPath "D:\models\RemoteSAM\bert-base-uncased"
```

Do not expose port 6657 to the public Internet. The verified deployment was
used only between machines on the same trusted laboratory network.
