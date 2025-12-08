# Demucs FastAPI 微服务

基于 Demucs 的音轨分离微服务，支持异步队列处理、多并发控制和灵活的模型配置。

## 功能特性

- ✅ **预加载模型**：服务启动时一次性加载 Demucs 模型，所有请求复用同一实例
- ✅ **灵活队列控制**：支持同步/异步两种处理模式，可通过参数动态切换
- ✅ **多并发支持**：内置异步队列，可控制最大并发数
- ✅ **任务状态追踪**：实时查询任务进度和状态
- ✅ **多模型支持**：支持所有 Demucs 模型（htdemucs, mdx, mdx_extra 等）
- ✅ **多分轨模式**：支持 2-stems、4-stems、6-stems
- ✅ **GPU 加速**：自动检测并使用 GPU（如果可用）

## 快速开始

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 2. 启动服务

```bash
python3 main.py
```

服务将在 `http://localhost:8000` 启动。

### 3. 访问 API 文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 使用示例

### 单文件处理（不使用队列）

```bash
curl -X POST "http://localhost:8000/separate?use_queue=false&stems=4stems" \
  -F "file=@song.mp3"
```

### 批量处理（使用队列）

```bash
curl -X POST "http://localhost:8000/separate?use_queue=true&stems=4stems" \
  -F "file=@song.mp3"
```

### 查询任务状态

```bash
curl "http://localhost:8000/task/{task_id}"
```

### 下载分离后的文件

```bash
curl "http://localhost:8000/download/{task_id}/vocals.mp3" -o vocals.mp3
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| use_queue | boolean | true | 是否使用异步队列 |
| model | string | htdemucs | 使用的模型 |
| stems | string | 4stems | 分轨数量 |
| mp3 | boolean | true | 输出为 MP3 格式 |
| mp3_rate | integer | 320 | MP3 比特率 |

## 许可证

MIT License