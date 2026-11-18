# GeoScope 国赛工作台状态矩阵

按 PR #9 的 `1d0a138` 核对。工具失败提示来自实际响应，结果归属以影像会话 ID 和结果版本 ID 为准。 [接口合同](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/contracts/api_contract_national.md)

| 场景 | 触发或响应 | 界面与保留状态 | 复核依据 |
| --- | --- | --- | --- |
| 初次打开或切换影像 | 尚无可用结果版本 | 显示当前原图及空统计，等待发起分析 | 本地浏览器选图与上传回归 |
| 执行中 | 任务已发送，等待工具事件 | 显示正在等待状态，接到响应后更新步骤和结果 | 浏览器任务执行回归 |
| 正常完成 | 保存结果版本，状态 `completed` | 显示对应蒙版、范围、统计、模型或算法元数据及导出入口 | [服务器浏览器验收](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/docs/evidence/server/real_ui.json) |
| 切换历史版本 | 选择已保存的结果 ID | 显示所选版本的蒙版、统计、执行记录和导出入口 | 同一 P1529 的基准与 AFR 版本对照 |
| 请求失败或需澄清 | 服务失败、无效 ROI 或任务需澄清 | 显示返回的失败或澄清信息，已有蒙版仍可查看和导出 | 本地失败、无效 ROI 与澄清用例 |
| 空蒙版 | 当前范围内没有前景像素 | 显示零像素、对应的占比和空结果提示 | 本地空 ROI 结果用例 |
| 版本对照不可用 | 同一版本、自相矛盾的绑定、坐标不相容或文件缺失 | 显示可核查原因，关闭当前对照图层或显示复算失败 | 本地版本校验与接口测试 |
| VerifiedPlan 成功 | 结果含 `tool_execution.steps` | 按顺序展示真实工具步骤和产物，重开版本后可查看 | [服务器调用证据](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/docs/evidence/server/full_chain_p1529.json) |
| VerifiedPlan 中途失败 | 某一步返回失败 | 展示失败位置，后续步骤停止，已完成的前序产物保留 | 计划失败隔离测试 |
| AFR 成功 | 当前结果含有效 `result.afr` | 在同坐标显示分歧图，支持透明度与区域定位。暂无待复核区域时显示空状态 | [真实分歧图](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/docs/evidence/server/real_disagreement.png) |
| AFR 回退或分歧图缺失 | AFR 回退到整图，或引用不可读取 | 回退时显示实际整图模式。读取失败时显示分歧图不可用，当前结果仍可查看 | 本地归档回放的 404 与回退用例 |
| CLIP 未就绪或查询失败 | 状态或查询返回 `not_ready` 或请求错误 | 显示未就绪或错误，保留重试与现有影像库入口 | 本地未就绪与真实服务器检索证据 |
| CLIP 空结果或成功 | 无候选，或返回真实已标记的候选和相似度 | 显示空状态，或按排名显示候选并允许打开相应影像会话 | [实际检索响应](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/docs/evidence/server/real_query.json) |
| 明确的样例演示 | 用户选择演示模式 | 界面标明预置标注，统计与差异由保存的蒙版计算 | [本地样例对照截图](screenshots/national-demo-difference.png) |
| 实验记录缺失 | 科研分组缺少真实记录或证据链接 | 显示未完成或尚未登记，已有数字继续链接到对应 CSV | 实验页面浏览器回归 |
