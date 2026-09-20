# 队员 B：RemoteSAM

先阅读仓库根目录 README 和 `docs/三人团队七天开发交接与AI任务说明.md` 第五节，将完整提示词交给你的 AI 编程助手。

本仓库采用原 `代码/RemoteAgent-main/servers/RemoteSAM/` 的源码作为初始快照，位于 `services/remotesam/upstream/`。不再复制一份独立 RemoteSAM 源码，避免两份运行代码混用；需要比较时使用上游 RemoteSAM 仓库。

重点文件：

- `upstream/remotesam.py`：HTTP 服务入口。
- `upstream/tasks/code/model.py`：模型初始化与分割方法。
- `upstream/requirements.txt` 和 `upstream/README.md`：上游依赖与说明。

已知待修问题：服务传入 `mask_output_hw`，本地模型函数不接受该参数；启用 EPOC 时需要额外辅助权重。请实际核实后修复，当前目录不代表服务已可运行。

目标接口维持 `POST /predict`：请求包含 `task=referring_seg`、base64 `image` 和英文 `text`；成功响应包含 `status=success`、与原图对应的 base64 PNG `mask`。变化要及时同步队长。

在 `feature/remotesam` 分支工作。第二天交真实分割服务及预测文件，第三天协助联调。权重、数据集与评测二进制资源不在本仓库；详见 `docs/上游代码来源.md`。
