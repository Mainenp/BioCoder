# BioCoder 色谱多模态 Agent 部署

研究训练与评测代码已合并到 `main`，但模型权重和受保护数据不进入 Git。
Agent 中的 `analyze_chromatogram` 工具因此默认关闭；只有冻结的外部部署物通过全量 SHA-256 校验后才会注册。

## 安全边界

- 产品运行时没有 dataset/split/answer 参数，只接受用户当前样本。
- 任何包含 `validation`、`internal_test`、answer key、training pool 或 RL selection set 的路径会被直接拒绝。
- deployment manifest 精确绑定 base model 和 processor 的完整文件集合，以及 adapter config/权重与训练报告；XIC 路径还绑定 sensor projector。多出、缺失或被修改的文件都会拒绝加载。
- 图像只能通过当前已鉴权聊天请求中的 `attachment_id` 读取；工具不能使用历史或其他用户附件。
- XIC 必须为有限数、已归一化到 `[0,1]`，长度必须与冻结 projector 一致（当前正式模型为 160）。
- XIC-only 后端安装视觉塔运行期 guard；任何视觉前向调用都会立即失败。

## 1. 冻结三个部署 manifest

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

`image_xic` 使用相同命令并指向融合 adapter/projector；`image_only` 指向 image-LoRA
adapter，设置 `--input-modality image_only`，并且完全省略 `--projector-path`。命令会生成
`deployment_manifest.json`、model 文件清单和 processor 文件清单，然后立即反向校验所有
绑定字节。输出目录必须不存在，避免静默覆盖已冻结部署。

## 2. 启用 Agent 工具

```bash
export CHROMATOGRAM_TOOL_ENABLED=true
export CHROMATOGRAM_XIC_ONLY_MANIFEST_PATH=/srv/biocoder/deployments/chrompeak-r2/deployment_manifest.json
export CHROMATOGRAM_IMAGE_XIC_MANIFEST_PATH=/srv/biocoder/deployments/chrompeak-image-xic/deployment_manifest.json
export CHROMATOGRAM_IMAGE_ONLY_MANIFEST_PATH=/srv/biocoder/deployments/chrompeak-image-lora/deployment_manifest.json
export CHROMATOGRAM_TOOL_TIMEOUT_SECONDS=300
```

旧的 `CHROMATOGRAM_DEPLOYMENT_MANIFEST_PATH` 仍作为 XIC-only 单模型配置兼容，但新部署
应使用三个按模态命名的变量，避免隐式选择。

启动 API 时会先校验 manifest。校验失败时 `BioAgent` 不会带病启动，也不会降级到未绑定权重。`GET /api/health` 的 `chromatogram_tool_enabled` 可用于部署检查。

GPU 环境需要安装 `backend` 的 `training` optional dependencies（Transformers、PEFT、safetensors 与 PyTorch/CUDA 由服务器环境提供）。模型首次调用时延迟加载。路由器始终只保留一个 CUDA backend；模态切换会串行卸载上一模型、清空 CUDA cache，再加载目标模型，避免三份 4B 权重同时占用显存。

## 3. 在 BioCoder 中使用

用户在同一条聊天消息中提供：

1. 已归一化的 XIC 数组；
2. 任务：峰存在性、峰区间定位、带元数据存在性或 scientific QC；
3. 附件组合确定且只确定一个路由：仅 JSON/XIC → `xic_only`，图片 + JSON/XIC → `image_xic`，仅图片 → `image_only`；
4. metadata/QC 任务还需 `component`、`channel`、`q1`、`q3`、`expected_rt_minutes`。

`image_width` 和 `image_height` 描述输出边界框使用的 ROI 像素坐标画布，不是 XIC
数组形状。标准 BioCoder ROI 为 `400×300`，而 projector 输入是独立的 160 个 XIC
采样点；标准流水线的 `xic_only` 请求因此应传 `image_width=400`、
`image_height=300`，不能把 XIC 长度 160 当成图像宽度。

Agent 会自动调用 `analyze_chromatogram`，返回结构化 JSON、deployment ID 与 deployment manifest SHA-256。峰定位输出为原图像素坐标 `[x1,y1,x2,y2]`，不是临床诊断结果。

### 结构化 XIC JSON（推荐）

长数值数组不应由聊天模型重新序列化。网页检测到上述三种色谱附件组合时，会直接调用
已鉴权的 `POST /api/chromatogram/analyze`，绕过 LLM 的
function-call 参数转录，同时继续使用相同的冻结运行时、manifest 校验与 GPU 推理锁：

```json
{
  "task": "peak_grounding",
  "language": "zh-CN",
  "xic_available": true,
  "image_width": 400,
  "image_height": 300,
  "xic_signal": [0.0, 0.1, 1.0, 0.1]
}
```

上例仅展示字段，正式部署的 `xic_signal` 必须与 projector 固定长度一致（当前为
160 点）。JSON + 图片时，图片实际尺寸必须与 JSON 声明一致。仅图片时网页读取真实图片
尺寸并默认执行 `peak_grounding`。接口要求登录；数值必须有限且位于 `[0,1]`。`xic_only` 部署不会读取
任何附件作为图像，即使工具调用模型误填了文本附件 ID，也会在读取附件字节前将其
忽略。普通 JSON 文件仍按常规聊天附件处理。

## 不包含的内容

GitHub 仓库包含推理代码、校验器、Agent 工具、测试与文档；不包含模型权重、私有色谱数据、validation/internal-test 资产或答案。因此新机器克隆仓库后不会“自动拥有”模型；必须显式挂载已冻结的外部部署物。
