# 实验室 GPU 0 环境修复与真实冒烟记录

> 时间：2026-11-17（Asia/Shanghai）
> 性质：环境与链路验证；**不是有真值的 A3/A4/A5 消融实验**。

## 1. 环境修复

- 主机：`gst@10.199.139.30`，只使用 GPU 0（NVIDIA GeForce RTX 3090）。
- checkpoint：`f85dfa044a527f096b9e41eacfe298040d52e77fca642f46dfe1226745e137e7`。
- 原始故障：`CUDNN_STATUS_NOT_INITIALIZED`。
- 实际根因：专用 venv 同时存在 `nvidia-cudnn-cu12==9.1.0.70` 与
  `nvidia-cudnn-cu13==9.24.0.43`；两者共享 `nvidia/cudnn/` 安装目录，cu13
  文件覆盖 cu12，导致 torch cu121 实际加载 cuDNN 9.24。
- 修复：仅在 `/home/gst/geoscope-national/.venv` 删除无反向依赖的 cu13
  NVIDIA 残留包，并按 torch 2.4.1 声明的精确版本重装 cu12 运行库。
- 修复后：`torch 2.4.1+cu121`、CUDA runtime 12.1、cuDNN 9.1.0；启用
  cuDNN 的最小 GPU 0 卷积通过。未修改系统 Python、驱动、权重或 GPU 1。

`pre_cuda_cleanup_freeze_20261117.txt` 保存修复前包清单；
`service_cudnn_fixed.log` 保存修复后服务启动及真实请求日志。

## 2. RemoteSAM 真实冒烟

输入为实验室专用目录已有的 800×800 机场图片，提示词 `all planes`，请求
`return_probability=true`。

| 运行 | 状态 | 推理耗时 | 总耗时 | 峰值显存 | 前景像素 |
|---|---|---:|---:|---:|---:|
| 首次（含 cuDNN 热身） | success | 1206.40 ms | 1261.97 ms | 1838.00 MB | 5795 |
| 重复稳定运行 | success | 147.31 ms | 173.48 ms | 1839.94 MB | 5795 |

两次运行的二值蒙版 SHA-256 均为
`a8d07b455d47c1bf26eb2663f2be938649f08801917eac12cccf17b2b332d6d9`，
概率图 SHA-256 均为
`3793d2a03437f381d7759b9bf727f2d4f2eddd82a55abbfb11e784143aab35b9`。

## 3. GeoScope-AFR 真实链路冒烟

同一真实图片仅为覆盖切片链路而缩放到 1600×1600；配置为
`Budget(max_regions=1)`、`lambda=0.5`、`threshold=0.5`、关闭局部放大。

- 候选网格 4 个，选择 1 个；请求数 2（整图 1 + 局部 1）。
- 局部成功 1、回退 0；产生分歧图；待复核区域 0。
- 客户端墙钟 1314.25 ms；局部服务总耗时 191.22 ms。
- 配置指纹：`797e3cd199323f72c1c6dff8b2e0dd8c93ba51c2def5478b640e5187ed49ef82`。

这项冒烟只证明 AFR 在真实 RemoteSAM 输出上可执行。由于没有对应真值，
不得从该结果计算或宣称 IoU、Dice、错误发现率或算法增益。

## 4. 收尾

验证结束后已精确停止本轮 RemoteSAM 进程。停止后 `nvidia-smi` 显示 GPU 0
37 MiB、GPU 1 5 MiB，未遗留我们的模型服务进程。
