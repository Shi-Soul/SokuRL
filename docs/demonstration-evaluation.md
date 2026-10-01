# 固定验证集上的行为保留检查

`tools/evaluate_demonstrations.py` 对已保存的共享 PPO 模型进行只读验证，支持前馈与循环模型。
它报告每个数据集的整体及双座位 NLL、最高概率动作准确率、变化帧准确率和分布熵，
同时列出复制上一实际动作的基线、有效帧数及完整局数。
指标回答教师标签拟合程度，不能替代完整神 AI 对局胜率。

数据继续经过已有严格 loader：完整采样结果、manifest/shard 哈希、契约、整局 split
和正式评估种子排除均需通过。只评分 validation，不采样训练帧。
循环模型使用整局顺序、局间归零及原序列评分实现；按座位重新评分仍保留每局的边界。
控制者为学习者时不输出价值 MSE，因为其回报不是教师策略的价值目标。
教师控制数据的价值 MSE 也只是相对采集轨迹回报的误差。

配置使用命名模型列表和命名数据集列表。每个模型明确指定检查点与训练配置。
入口读取检查点字节后，从这同一份内存字节加载并计算哈希；不会在文件可能变化时
先哈希一个版本再加载另一个版本。仍应优先选已完成训练的 best/final 或不可变的定期检查点，
不要依赖正在重写的 ZIP 文件。输出保存模型、数据 manifest/config 与验证 shard 的哈希、
验证世界种子、源码及依赖身份；验证前后核对模型参数和 PPO 步数不变。

```bash
bash scripts/linux.sh tools/evaluate_demonstrations.py linux.cuda_devices=0 \
  'models={expanded:{checkpoint:logs/pretraining/god-marisa-reimu-recurrent-expanded-20261001/best.zip,training_config:logs/pretraining/god-marisa-reimu-recurrent-expanded-20261001/config.yaml}}' \
  'datasets={original_teacher:logs/demonstrations/god-marisa-reimu-20261001,expanded_teacher:logs/demonstrations/god-marisa-reimu-expanded-20261001,recurrent_learner:logs/demonstrations/learner-recurrent-marisa-reimu-20261001}' \
  output=logs/diagnostics/expanded-on-recurrent-validation-20261001
```

新入口及相关序列/数据集回归共 26 项通过，日志
`.dev/pytest-demonstration-evaluation-20261001-v3.log`。
入口测试实际加载前馈/循环 ZIP、生成结果 JSON，并核对固定验证身份、座位加权一致性和模型只读性。
前一轮入口测试发现 NumPy 计数不支持 JSON 序列化，已显式转换整数/浮点类型；失败日志保留。

## 首次真实固定数据集比较

源码 `77a55f1` 在 GPU 0 完成原循环 BC 和扩充循环 BC 的比较，输出
`logs/diagnostics/recurrent-baselines-by-dataset-and-seat-20261001`。
两个教师验证集分别为 4 局/28800 帧、8 局/49744 帧；新增循环学习者验证集为 8 局/29303 帧。
前两集的结果与已有独立核对一致，新增学习者状态的结果如下：

| 模型 | NLL | 总准确率 | 修正帧准确率 | 分布熵 | 1P / 2P 准确率 |
| --- | --- | --- | --- | --- | --- |
| 原循环 BC | 2.63818 | 49.688% | 4.763% | 0.40659 | 51.138% / 47.473% |
| 扩充循环 BC | 3.04856 | 50.691% | 6.493% | 0.22300 | 52.075% / 48.577% |

修正帧指教师当前标签不同于学习者上一帧实际命令，共 15094 帧；
它不是教师自己连续操作时的动作切换帧。复制上一实际命令基线为 48.476%。
1P/2P 分别为 17709/11594 帧，不能将不同对局构成的差值单独归因于座位。
扩充模型在两个教师集总准确率约 93.7%，但在学习者状态仍仅约 50.7%，
且相较原循环模型分布熵更低、标签 NLL 更高。这支持继续检查学习者状态聚合，
尚不能证明聚合后完整神 AI 胜率会提高。
模型间验证身份完全配对、座位加权与整体指标一致性已核对，见
`.dev/audit-recurrent-baselines-by-dataset-and-seat-20261001.log`。
