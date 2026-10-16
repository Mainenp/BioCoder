# BioCoder 色谱多模态 Agent 部署

研究训练与评测代码已合并到 `main`，但模型权重和受保护数据不进入 Git。
Agent 中的 `analyze_chromatogram` 工具因此默认关闭；只有冻结的外部部署物通过全量 SHA-256 校验后才会注册。

## 安全边界

- 产品运行时没有 dataset/split/answer 参数，只接受用户当前样本。
- 任何包含 `validation`、`internal_test`、answer key、training pool 或 RL selection set 的路径会被直接拒绝。
- deployment manifest 精确绑定 base model 和 processor 的完整文件集合，以及 adapter config/权重、sensor projector 与训练报告；多出、缺失或被修改的文件都会拒绝加载。
- 图像只能通过当前已鉴权聊天请求中的 `attachment_id` 读取；工具不能使用历史或其他用户附件。
- XIC 必须为有限数、已归一化到 `[0,1]`，长度必须与冻结 projector 一致（当前正式模型为 160）。
- XIC-only 后端安装视觉塔运行期 guard；任何视觉前向调用都会立即失败。

## 1. 冻结部署 manifest

在有模型权重和候选人工物的 GPU 服务器上运行：

```bash
cd backend
python -m multimodal_science.qwen3vl.freeze_deployment_cli \
  --deployment-id chrompeak-qwen3vl-production-v1 \
  --output-dir /srv/biocoder/deployments/chrompeak-qwen3vl-production-v1 \
  --model-root /srv/biocoder/models/Qwen3-VL-4B-Instruct \
  --model-revision modelscope-master \
  --adapter-dir /srv/biocoder/candidates/selected/adapter \
  --projector-path /srv/biocoder/candidates/selected/sensor_projector.safetensors \
  --processor-dir /srv/biocoder/candidates/selected/processor \
  --training-report-path /srv/biocoder/candidates/selected/fusion_training_report.json \
  --input-modality xic_only
```

命令会生成 `deployment_manifest.json`、model 文件清单和 processor 文件清单，然后立即反向校验所有绑定字节。输出目录必须不存在，避免静默覆盖已冻结部署。

## 2. 启用 Agent 工具

```bash
export CHROMATOGRAM_TOOL_ENABLED=true
export CHROMATOGRAM_DEPLOYMENT_MANIFEST_PATH=/srv/biocoder/deployments/chrompeak-qwen3vl-production-v1/deployment_manifest.json
export CHROMATOGRAM_TOOL_TIMEOUT_SECONDS=300
```

启动 API 时会先校验 manifest。校验失败时 `BioAgent` 不会带病启动，也不会降级到未绑定权重。`GET /api/health` 的 `chromatogram_tool_enabled` 可用于部署检查。

GPU 环境需要安装 `backend` 的 `training` optional dependencies（Transformers、PEFT、safetensors 与 PyTorch/CUDA 由服务器环境提供）。模型首次调用时延迟加载，之后复用同一个只读实例；CUDA 推理通过锁串行化，避免同一张卡的 KV cache 争用。

## 3. 在 BioCoder 中使用

用户在同一条聊天消息中提供：

1. 已归一化的 XIC 数组；
2. 任务：峰存在性、峰区间定位、带元数据存在性或 scientific QC；
3. `image_xic` 部署还需要一张 ROI 图像附件；`xic_only` 部署不读取图像像素；
4. metadata/QC 任务还需 `component`、`channel`、`q1`、`q3`、`expected_rt_minutes`。

Agent 会自动调用 `analyze_chromatogram`，返回结构化 JSON、deployment ID 与 deployment manifest SHA-256。峰定位输出为原图像素坐标 `[x1,y1,x2,y2]`，不是临床诊断结果。

## 不包含的内容

GitHub 仓库包含推理代码、校验器、Agent 工具、测试与文档；不包含模型权重、私有色谱数据、validation/internal-test 资产或答案。因此新机器克隆仓库后不会“自动拥有”模型；必须显式挂载已冻结的外部部署物。
