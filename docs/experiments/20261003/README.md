# 2026-10-03 停止时的实验归档

阅读[实验状态与结论](../../experiment-status-20261003.md)。本目录是对已有文件的只读盘点，
没有启动游戏、策略推理或优化器。时间为 19:30:08 UTC，源代码提交 `ca8c51a`。
正文文档索引保存整理前的版本；随后新增的停止说明以本次文档提交为准。

| 类别 | 运行目录数 | 索引 |
| --- | ---: | --- |
| 在线训练 | 56 | [training-01.csv](training-01.csv) |
| 离线预训练 | 31 | [pretraining-01.csv](pretraining-01.csv) |
| 游戏评测 | 149 | [benchmark-01.csv](benchmark-01.csv) |
| 示范采集 | 11 | [demonstrations-01.csv](demonstrations-01.csv) |
| 诊断与审计 | 485 | [diagnostics-01.csv](diagnostics-01.csv) |

共 732 个目录，包含历史失败、重复、共享基线和诊断，**不是 732 个独立实验**。
不能直接相加得到独立对局数或新增训练样本数。成功标志只表示原报告执行成功，
不表示策略足够强，也不表示所有记录都经过相同级别的独立审计。

目录索引包含原始执行状态、错误、可用步数/监督更新/最佳轮次、配置/身份/结果哈希，
以及能匹配到该目录的专题文档。空字段代表该根文件没有相应字段，不补造数值。
`no_root_result` 不等于仍在运行；许多诊断只写 summary 或嵌套结果。

9045 份 JSON/YAML/CSV 证据逐文件重新计算 SHA256，记录在 `evidence-01.csv` 至
`evidence-19.csv`。每行都有所属运行、相对路径、字节数和 SHA256，每片不超过 500 条。
它们包括已有逐局计划、指标、运行身份和审计结果，可沿路径在 NAS 找到完整内容。
训练日志的 stdout、原生回放、NPZ 和 PT 分片保留原位，由运行及原 manifest 关联，
没有复制到 Git。`logs/hydra` 重复启动记录、运行验收/部署目录不计入本次实验目录数。

模型索引：[models-01.csv](models-01.csv)、[models-02.csv](models-02.csv)，共 731 个文件。
其中 218 个 initial/best/final 文件在归档时重新哈希；其余 checkpoint 仅列路径和大小，
明确标记 `listed_only`。不能将仅列出的文件解释为已重新检查权重或重测策略。

[documents-01.csv](documents-01.csv) 保存原 96 篇顶层专题的标题、路径与当时哈希，
可与 `ca8c51a` 中的文档比较。完整盘点口径和数量在 [snapshot.json](snapshot.json)。
[record_archive.py](record_archive.py) 是生成脚本，禁止覆盖已存在快照；当前归档已完成，
不需要重新运行。脚本及本轮关键 scratch 文件的哈希均已保存。

最后一轮用户中止的采集单列为
[interrupted-collection.json](interrupted-collection.json)：8 个完整局、其余局计数、
统计、分片哈希验证、退出原因及独立 Wine 服务清理；不是完整 16 局评测。
[processes.txt](processes.txt) 是归档时任务路径相关进程快照，保留基础显示/音频和旧服务；
当时列出的 `record_archive.py` 是本次归档进程，随后已退出 0。

所有内容作为证据快照保存。恢复实验须由用户明确提出，且不能覆写旧模型、结果或中止数据。
