# `.iplmodel` 模型契约 v1

模型必须通过工具打包，应用不接受裸 `.tflite`。包内只能有：

```text
model.iplmodel
├── manifest.json
└── model.tflite
```

清单完整示例见 `examples/model-manifest.example.json`。固定约束如下：

- runtime 为 `tflite`，输入和输出均为单个 `float32` 张量；
- 输入布局固定为 `[1,T,6]` (`NTC`)；
- 通道固定为 `[gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]`；
- 输入清单必须声明采样率和 6 通道 mean/std；
- mean/std 必须是有限数值，且每个 std 必须大于 0；
- 输出固定为 `[1,2]` 或 `[1,3]` 的速度，单位 `m/s`；
- 输出坐标系必须明确为 `body` 或 `world`；
- benchmark 声明线程数、热身次数、步长和最大窗口数。

打包命令：

```bash
python tools/pack_model.py \
  --model model.tflite \
  --manifest examples/model-manifest.example.json \
  --output my-model.iplmodel
```

Python 工具会拒绝未知字段、非法 benchmark 参数并校验清单和 TFLite FlatBuffer 标识；Android 导入时还会拒绝未知或重复的 ZIP 条目，并用设备上的 TFLite 解释器核对真实 tensor 数量、dtype 和 shape。任何一项不一致都会拒绝安装。

## 评测口径

- 性能：热身后逐窗口同步推理的 mean/P50/P95/P99 和吞吐；
- 精度：速度 RMSE、积分轨迹 ATE RMSE、终点漂移和相对漂移；
- 资源：进程 CPU 时间、PSS 峰值、Java heap 变化、墙钟时间和热状态；
- 能耗：设备实现 BatteryManager 能量/电荷计数器时报告 mWh/mAh 和平均功率，否则明确标记不可用。

采样率不一致时应用会拒绝评测，而不是静默重采样。精度只在位置有效掩码成立时计算，报告会记录 `world_frame`、位置来源与姿态来源。ARCore VIO 沿用原版 IMUNet 的手机端参考轨迹角色，同时明确标注它不是独立的 Vicon/RTK 测量。

应用可一次导入多个 `.iplmodel`。数据包模式按所选“模型 × 数据序列”生成确定的顺序队列；单项失败写入批量报告而不中断其他组合，各模型不并发执行，避免延迟、CPU 和内存测量相互污染。

实时联测只接受 `input.sample_rate_hz` 与当前实际采集率一致的模型。每个模型在后台维护独立滚动窗口，依照其 `stride` 调用 LiteRT，将速度积分为预测轨迹，并与同步 ARCore 局部轨迹计算端点误差。
