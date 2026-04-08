# EPNet Demo 中文说明

基于论文 **“EPNet: An Efficient Pyramid Network for Enhanced Single-Image Super-Resolution with Reduced Computational Requirements”** 的工程化单图超分辨率演示项目。

[English README](README.md)

这个仓库不只是论文复现代码，而是一个更接近解决方案演示的完整资产：

- PyTorch 版 EPNet 模型实现
- 训练、评估、推理 CLI
- FastAPI 后端
- Next.js + TypeScript 前端
- 使用分析、历史回放、可分享结果链接
- Docker 化本地完整栈
- 自动化测试与演示辅助文档

## 这个 Demo 现在可以展示什么

- 中英文界面切换
- 单图拖拽上传推理
- 批量推理模式，展示队列和聚合统计
- `EPNet` / `Bicubic` / `Baseline` 的 A/B 对比
- x2 / x3 / x4 倍率切换
- 自动发现并切换多个 checkpoint
- 输出格式切换：`PNG` / `JPEG` / `WEBP` / `BMP`
- Tile inference 开关，用于大图内存友好演示
- Session ID，用于区分不同演示会话
- 下载结果与分享链接
- 历史推理记录回放
- 模型解释页，便于讲解 PFEM / ESPM / reconstruction
- 模型版本 / checkpoint 来源 / build time / git commit / device target 面板
- 一键复制训练 / 评估 / 推理命令
- 仪表盘展示 request count、session count、平均延迟、p50、p95 等指标

## 当前状态

- EPNet 主体模型已实现。
- 论文中的主要训练默认值已经体现在配置和 CLI 中。
- 评估流程支持 PSNR / SSIM。
- 后端支持健康检查、模型信息、单图推理、批量推理、统计汇总、最近请求和历史回放。
- 结果图通过 URL 返回，不再走内联 base64。
- 前端包含动画对比、局部放大镜、系统流水线时间线、批量队列、历史回放和模型解释页。
- 已加入 Playwright 冒烟测试，覆盖单图上传、错误提示、页面刷新稳定性、批量模式等核心链路。

## 重要说明

默认的 `checkpoints/demo_x4.pt` 是一个为了端到端可运行而准备的 demo 级 checkpoint，不是按论文完整训练得到的高质量基准权重。

如果你希望做更强的视觉展示或 benchmark，请训练真实权重并放入 `checkpoints/` 目录。

论文中不明确的实现点都记录在：

- [docs/epnet_assumptions.md](/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md)

## 项目结构

- `src/epnet`：模型、模块、数据集、指标、profiling、训练/评估/推理 CLI
- `src/epnet_api`：FastAPI 后端、schema、服务层、分析数据库
- `frontend`：Next.js + TypeScript + Tailwind + Framer Motion + Recharts 前端
- `tests`：模型、后端和训练相关 smoke tests
- `docs`：假设说明、审计报告、优化报告、演示清单
- `scripts`：本地运行辅助脚本
- `outputs`：运行期生成的分析库和图片产物

## 与论文一致的实现范围

- 浅层 `3x3` 卷积做基础特征提取
- PFEM 分支包含 LFEB、改造版 Swin Transformer 和 ESAB
- ESPM 分支进行多尺度金字塔上下文建模
- 通过 `3x3` 卷积 + `PixelShuffle` 做图像重建
- 默认 `n = 4` 个 PFEM 子模块
- 训练默认值对齐论文附录：
  - patch size `48`
  - batch size `32`
  - Adam `lr=5e-4`, betas `(0.9, 0.99)`
  - L1 loss
  - EMA decay `0.999`
  - `1e6` iterations
  - no warm-up

## 安装

### 后端

```bash
python3 -m venv .venv
.venv/bin/pip install '.[dev]'
```

### 前端

```bash
cd frontend
npm install
cd ..
```

## 核心命令

```bash
PYTHONPATH=src epnet-train --output checkpoints/epnet_x4.pt --scale 4 --train-dir path/to/div2k_train_hr
PYTHONPATH=src epnet-train --output checkpoints/epnet_x4.pt --scale 4 --train-dir path/to/div2k_train_hr --resume checkpoints/epnet_x4.pt
PYTHONPATH=src epnet-eval --checkpoint checkpoints/epnet_x4.pt --hr-dir path/to/benchmark_hr
PYTHONPATH=src epnet-infer --checkpoint checkpoints/epnet_x4.pt --input data/samples/demo_input.png --output outputs/demo_output.png
PYTHONPATH=src .venv/bin/uvicorn epnet_api.app.main:app --reload
cd frontend && npm run dev
./scripts/run_local.sh
```

## 一条命令跑本地 Demo

```bash
./scripts/run_local.sh
```

这个脚本会自动：

- 创建 `.venv`
- 安装后端依赖
- 安装前端依赖
- 如果缺少 `checkpoints/demo_x4.pt`，自动生成一个 demo 用 checkpoint
- 启动 FastAPI：`http://localhost:8000`
- 启动 Next.js：`http://localhost:3000`

## API 概览

主要接口：

- `GET /api/v1/health`
- `GET /api/v1/model/info`
- `POST /api/v1/infer`
- `POST /api/v1/infer/batch`
- `GET /api/v1/usage/summary`
- `GET /api/v1/usage/recent`
- `GET /api/v1/history/{request_id}`

兼容别名：

- `POST /api/v1/super-resolve`
- `GET /api/v1/analytics/summary`

### 单图推理参数

`POST /api/v1/infer` 支持这些 multipart form 字段：

- `file`
- `session_id`
- `method`：`epnet`、`bicubic`、`baseline`
- `scale`：`2`、`3`、`4`
- `output_format`：`PNG`、`JPEG`、`WEBP`、`BMP`
- `tile_size`：`0`、`256`、`384`、`512`
- `checkpoint_name`：可选，仅对 EPNet 模式有效

### 批量推理

`POST /api/v1/infer/batch` 使用重复的 `files` 字段上传多张图片，并支持和单图模式相同的控制参数。

### 使用分析与历史回放

每条推理记录会保存：

- request ID 和 UTC 时间
- session ID
- 输入/输出分辨率
- 输入/输出字节数
- 方法与倍率
- checkpoint 名称
- 输出格式
- tile 大小
- 延迟
- 参数量
- 预估 MACs / FLOPs
- 输入/输出产物 URL

## Checkpoint 与倍率切换

后端会自动扫描 `checkpoints/*.pt`。

- 发现到的 checkpoint 会自动出现在前端切换器中
- EPNet 的可选倍率取决于当前实际加载到的 checkpoint
- 即使 EPNet 没有 x2/x3/x4 的完整权重，`Bicubic` 和 `Baseline` 仍然可以演示全部倍率

例如如果你放入：

- `checkpoints/demo_x2.pt`
- `checkpoints/demo_x3.pt`
- `checkpoints/demo_x4.pt`
- `checkpoints/epnet_stage200k_x4.pt`

前端就会自动把它们展示出来。

## 环境变量

- `EPNET_CHECKPOINT_PATH`
- `EPNET_ANALYTICS_DB_PATH`
- `EPNET_ARTIFACTS_DIR`
- `EPNET_MAX_UPLOAD_BYTES`
- `EPNET_MAX_IMAGE_PIXELS`
- `EPNET_CORS_ORIGINS`
- `EPNET_BUILD_TIME`
- `EPNET_GIT_COMMIT`
- `EPNET_PROFILE_INPUT_SIZE`

## Docker

```bash
docker compose up --build
```

- 前端：[http://localhost:3000](http://localhost:3000)
- 后端：[http://localhost:8000](http://localhost:8000)

补充说明：

- 后端 Docker 默认安装 CPU 版 PyTorch，适合本地展示
- 前端 `build` 脚本会先清理 `.next`，避免 App Router 的旧构建缓存影响产物
- Compose 已配置健康检查

## 已验证项

- `PYTHONPATH=src .venv/bin/ruff check src tests`
- `PYTHONPATH=src .venv/bin/mypy src`
- `PYTHONPATH=src .venv/bin/pytest`
- `cd frontend && npm run lint`
- `cd frontend && npm run typecheck`
- `cd frontend && npm run build`
- `cd frontend && npm run test:e2e`
- `docker compose config`
- `docker compose up -d --build`

## 更适合演示的讲解顺序

1. 先讲 EPNet 为什么是“高效超分”而不是“大模型堆料”
2. 再讲系统流：upload -> validate -> decode -> preprocess -> infer -> encode -> log
3. 然后展示版本信息、checkpoint 来源和部署元数据
4. 再切到 batch、history replay、share link、A/B compare
5. 最后用 CLI 复现命令和 Docker 收尾，强调工程完整性

## 相关文档

- [docs/epnet_assumptions.md](/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md)
- [docs/final_audit_report.md](/Users/macbook/Desktop/epnet/docs/final_audit_report.md)
- [docs/optimization_report.md](/Users/macbook/Desktop/epnet/docs/optimization_report.md)
- [docs/demo_checklist.md](/Users/macbook/Desktop/epnet/docs/demo_checklist.md)
