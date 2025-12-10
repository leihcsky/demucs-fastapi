"""
Demucs FastAPI 微服务
支持音轨分离、异步队列处理、多并发控制
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import uvicorn
import os
import uuid
import shutil
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
import logging
from enum import Enum
import demucs.separate
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Demucs 音轨分离服务",
    description="基于 Demucs 的音轨分离微服务，支持异步队列和并发控制",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Config:
    UPLOAD_DIR = Path("./uploads")
    OUTPUT_DIR = Path("./outputs")
    TEMP_DIR = Path("./temp")
    MAX_CONCURRENT_JOBS = 3
    DEFAULT_MODEL = "htdemucs"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def init_dirs(cls):
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.TEMP_DIR.mkdir(exist_ok=True)

Config.init_dirs()

class ModelType(str, Enum):
    htdemucs = "htdemucs"
    htdemucs_ft = "htdemucs_ft"
    htdemucs_6s = "htdemucs_6s"
    hdemucs_mmi = "hdemucs_mmi"
    mdx = "mdx"
    mdx_extra = "mdx_extra"
    mdx_q = "mdx_q"
    mdx_extra_q = "mdx_extra_q"

class StemType(str, Enum):
    two_stems = "2stems"
    four_stems = "4stems"
    six_stems = "6stems"

class SoundSource(str, Enum):
    vocals = "vocals"
    bass = "bass"
    drums = "drums"
    other = "other"
    piano = "piano"
    guitar = "guitar"

class TaskStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class SeparationRequest(BaseModel):
    use_queue: bool = Field(default=True, description="是否使用异步队列")
    model: ModelType = Field(default=ModelType.htdemucs, description="使用的模型")
    stems: StemType = Field(default=StemType.four_stems, description="分轨数量")
    mp3: bool = Field(default=True, description="输出为 MP3 格式")
    mp3_rate: int = Field(default=320, description="MP3 比特率")
    float32: bool = Field(default=False, description="使用 float32 精度")
    int24: bool = Field(default=False, description="使用 int24 精度")
    sound_source: Optional[SoundSource] = Field(default=None, description="指定声音来源")

class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float
    message: str
    created_at: str
    completed_at: Optional[str] = None
    output_files: Optional[List[str]] = None
    error: Optional[str] = None

class GlobalState:
    def __init__(self):
        self.tasks: Dict[str, TaskInfo] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.processing_count = 0
        self.models_cache = {}
        
    async def load_model(self, model_name: str):
        if model_name not in self.models_cache:
            logger.info(f"加载模型: {model_name}")
            self.models_cache[model_name] = True
        return True

state = GlobalState()

def get_model_stems(model: ModelType, stems: StemType) -> str:
    # if stems == StemType.two_stems:
    #     return f"{model.value}_2stems"
    # elif stems == StemType.six_stems:
    if stems == StemType.six_stems:
        return "htdemucs_6s"
    else:
        return model.value

def convert_audio_format(
    input_path: Path,
    output_format: str,
    output_dir: Optional[Path] = None
) -> Path:
    """
    使用 ffmpeg 转换音频格式

    Args:
        input_path: 输入文件路径
        output_format: 目标格式 (wav, mp3, flac)
        output_dir: 输出目录，默认为临时目录

    Returns:
        转换后的文件路径
    """
    if output_dir is None:
        output_dir = Config.TEMP_DIR

    output_dir.mkdir(exist_ok=True, parents=True)

    # 生成输出文件名
    output_filename = input_path.stem + f".{output_format}"
    output_path = output_dir / output_filename

    # 构建 ffmpeg 命令
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-y",  # 覆盖已存在的文件
    ]

    # 根据格式添加特定参数
    if output_format == "mp3":
        cmd.extend([
            "-codec:a", "libmp3lame",
            "-b:a", "320k",
            "-ar", "44100"
        ])
    elif output_format == "flac":
        cmd.extend([
            "-codec:a", "flac",
            "-compression_level", "8"
        ])
    elif output_format == "wav":
        cmd.extend([
            "-codec:a", "pcm_s16le",
            "-ar", "44100"
        ])

    cmd.append(str(output_path))

    try:
        logger.info(f"转换音频格式: {input_path.name} -> {output_format}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )

        if result.returncode != 0:
            logger.error(f"ffmpeg 转换失败: {result.stderr}")
            raise RuntimeError(f"音频格式转换失败: {result.stderr}")

        logger.info(f"转换完成: {output_path}")
        return output_path

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg 转换超时")
        raise RuntimeError("音频格式转换超时")
    except FileNotFoundError:
        logger.error("ffmpeg 未安装或不在 PATH 中")
        raise RuntimeError("ffmpeg 未安装，无法进行格式转换")

def separate_audio(
    file_path: Path,
    output_dir: Path,
    model: str,
    mp3: bool = True,
    mp3_rate: int = 320,
    float32: bool = False,
    int24: bool = False,
    stems: StemType = StemType.four_stems,
    sound_source: Optional[SoundSource] = None
) -> List[Path]:
    try:
        args = [
            str(file_path),
            "-n", model,
            "-o", str(output_dir),
            "--device", Config.DEVICE
        ]
        # 分离人声和伴奏
        if stems and stems == StemType.two_stems:
            if sound_source:
                args.append(f"--two-stems={sound_source.value}")
            else:
                args.append("--two-stems=vocals")
        
        if stems and stems == StemType.six_stems:
            if sound_source:
                args.append(f"--two-stems={sound_source.value}")

        if mp3:
            args.extend(["--mp3", f"--mp3-bitrate={mp3_rate}"])
        if float32:
            args.append("--float32")
        if int24:
            args.append("--int24")
        
        logger.info(f"开始分离音轨: {file_path.name}")
        logger.info(f"使用参数: {' '.join(args)}")
        
        import sys
        old_argv = sys.argv
        sys.argv = ['demucs'] + args
        
        try:
            demucs.separate.main()
        finally:
            sys.argv = old_argv
        
        model_output_dir = output_dir / model / file_path.stem
        output_files = list(model_output_dir.glob("*"))
        
        logger.info(f"分离完成，生成 {len(output_files)} 个文件")
        return output_files
        
    except Exception as e:
        logger.error(f"分离失败: {str(e)}")
        raise

async def process_task(task_id: str, file_path: Path, request: SeparationRequest):
    task = state.tasks[task_id]
    
    try:
        task.status = TaskStatus.processing
        task.message = "正在处理..."
        task.progress = 0.1
        
        model_name = get_model_stems(request.model, request.stems)
        
        await state.load_model(model_name)
        task.progress = 0.3
        
        output_dir = Config.OUTPUT_DIR / task_id
        output_dir.mkdir(exist_ok=True, parents=True)
        
        task.progress = 0.4
        task.message = "正在分离音轨..."
        
        output_files = await asyncio.to_thread(
            separate_audio,
            file_path,
            output_dir,
            model_name,
            request.mp3,
            request.mp3_rate,
            request.float32,
            request.int24,
            request.stems,
            request.sound_source
        )
        
        task.progress = 0.9
        task.status = TaskStatus.completed
        task.message = "处理完成"
        task.progress = 1.0
        task.completed_at = datetime.now().isoformat()
        task.output_files = [f.name for f in output_files]
        
        logger.info(f"任务 {task_id} 完成")
        
    except Exception as e:
        task.status = TaskStatus.failed
        task.error = str(e)
        task.message = f"处理失败: {str(e)}"
        logger.error(f"任务 {task_id} 失败: {str(e)}")
    
    finally:
        state.processing_count -= 1

async def queue_worker():
    while True:
        if state.processing_count >= Config.MAX_CONCURRENT_JOBS:
            await asyncio.sleep(1)
            continue
        
        try:
            task_data = await asyncio.wait_for(state.queue.get(), timeout=1.0)
            state.processing_count += 1
            asyncio.create_task(process_task(**task_data))
        except asyncio.TimeoutError:
            continue

@app.on_event("startup")
async def startup_event():
    logger.info("启动 Demucs 服务...")
    logger.info(f"使用设备: {Config.DEVICE}")
    
    await state.load_model(Config.DEFAULT_MODEL)
    
    asyncio.create_task(queue_worker())
    
    logger.info("服务启动完成")

@app.get("/")
async def root():
    return {
        "service": "Demucs 音轨分离服务",
        "version": "1.0.0",
        "status": "running",
        "device": Config.DEVICE
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": Config.DEVICE,
        "processing_count": state.processing_count,
        "queue_size": state.queue.qsize(),
        "active_tasks": len([t for t in state.tasks.values() if t.status == TaskStatus.processing])
    }

@app.post("/separate")
async def separate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_queue: bool = Query(True),
    model: ModelType = Query(ModelType.htdemucs),
    stems: StemType = Query(StemType.four_stems),
    mp3: bool = Query(True),
    mp3_rate: int = Query(320),
    float32: bool = Query(False),
    int24: bool = Query(False),
    sound_source: Optional[SoundSource] = Query(None)
):
    try:
        task_id = str(uuid.uuid4())
        
        file_path = Config.UPLOAD_DIR / f"{task_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.pending,
            progress=0.0,
            message="任务已创建",
            created_at=datetime.now().isoformat()
        )
        state.tasks[task_id] = task_info
        
        request = SeparationRequest(
            use_queue=use_queue,
            model=model,
            stems=stems,
            mp3=mp3,
            mp3_rate=mp3_rate,
            float32=float32,
            int24=int24,
            sound_source=sound_source
        )
        
        if use_queue:
            await state.queue.put({
                "task_id": task_id,
                "file_path": file_path,
                "request": request
            })
            logger.info(f"任务 {task_id} 已加入队列")
        else:
            state.processing_count += 1
            background_tasks.add_task(process_task, task_id, file_path, request)
            logger.info(f"任务 {task_id} 直接处理")
        
        return {
            "task_id": task_id,
            "status": "accepted",
            "message": "任务已接受" + (" (队列模式)" if use_queue else " (直接模式)")
        }
        
    except Exception as e:
        logger.error(f"创建任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}", response_model=TaskInfo)
async def get_task_status(task_id: str):
    if task_id not in state.tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return state.tasks[task_id]

@app.get("/tasks")
async def list_tasks(
    status: Optional[TaskStatus] = None,
    limit: int = Query(50, le=200)
):
    tasks = list(state.tasks.values())
    
    if status:
        tasks = [t for t in tasks if t.status == status]
    
    tasks.sort(key=lambda x: x.created_at, reverse=True)
    return {
        "total": len(tasks),
        "tasks": tasks[:limit]
    }

@app.get("/download/{task_id}/{filename}")
async def download_file(
    task_id: str,
    filename: str,
    delete_after: bool = Query(False, description="下载后是否删除文件")
):
    """
    下载处理结果文件

    参数:
        - task_id: 任务ID
        - filename: 文件名
        - delete_after: 下载后是否删除文件 (默认 False)
    """
    if task_id not in state.tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = state.tasks[task_id]
    if task.status != TaskStatus.completed:
        raise HTTPException(status_code=400, detail="任务尚未完成")

    # 查找文件
    model_name = Config.DEFAULT_MODEL
    output_dir = Config.OUTPUT_DIR / task_id / model_name

    possible_paths = [
        output_dir / filename,
        Config.OUTPUT_DIR / task_id / filename,
    ]

    file_path = None
    for path in possible_paths:
        if path.exists():
            file_path = path
            break

    if not file_path:
        for root, _, files in os.walk(Config.OUTPUT_DIR / task_id):
            if filename in files:
                file_path = Path(root) / filename
                break

    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    # 确定 MIME 类型
    file_ext = file_path.suffix.lstrip('.').lower()
    mime_types = {
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
        'flac': 'audio/flac'
    }
    media_type = mime_types.get(file_ext, 'application/octet-stream')

    # 创建响应
    response = FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )

    # 如果需要删除文件
    if delete_after:
        logger.info(f"下载后将删除文件: {file_path}")

        # 使用后台任务删除文件
        async def cleanup_file():
            await asyncio.sleep(1)  # 等待响应发送完成
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"已删除文件: {file_path}")

                # 更新任务信息
                if task.output_files and filename in task.output_files:
                    task.output_files.remove(filename)
                    logger.info(f"已从任务输出列表中移除: {filename}")

            except Exception as e:
                logger.error(f"清理文件失败: {str(e)}")

        # 创建后台任务
        asyncio.create_task(cleanup_file())

    return response

@app.post("/convert")
async def convert_audio(
    file: UploadFile = File(...),
    format: str = Query(..., description="目标格式: wav, mp3, flac")
):
    """
    音频格式转换接口

    参数:
        - file: 上传的音频文件
        - format: 目标格式 (wav, mp3, flac)

    返回:
        转换后的音频文件（如果已是目标格式则直接返回原文件）
    """
    try:
        # 验证目标格式
        target_format = format.lower()
        supported_formats = ['wav', 'mp3', 'flac']
        if target_format not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的格式: {target_format}。支持的格式: {', '.join(supported_formats)}"
            )

        # 生成临时文件ID
        convert_id = str(uuid.uuid4())

        # 保存上传的文件
        original_filename = file.filename or "audio"
        file_ext = Path(original_filename).suffix.lstrip('.').lower()

        temp_input_path = Config.TEMP_DIR / f"{convert_id}_input.{file_ext}"
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"接收到转换请求: {original_filename} -> {target_format}")

        # 检查是否需要转换
        if file_ext == target_format:
            logger.info(f"文件已是目标格式 {target_format}，直接返回")

            # 确定 MIME 类型
            mime_types = {
                'mp3': 'audio/mpeg',
                'wav': 'audio/wav',
                'flac': 'audio/flac'
            }
            media_type = mime_types.get(target_format, 'application/octet-stream')

            # 生成输出文件名
            output_filename = Path(original_filename).stem + f".{target_format}"

            # 返回原文件
            response = FileResponse(
                path=temp_input_path,
                filename=output_filename,
                media_type=media_type
            )

            # 后台删除临时文件
            async def cleanup():
                await asyncio.sleep(2)
                try:
                    if temp_input_path.exists():
                        temp_input_path.unlink()
                        logger.info(f"已清理临时文件: {temp_input_path}")
                except Exception as e:
                    logger.error(f"清理临时文件失败: {str(e)}")

            asyncio.create_task(cleanup())
            return response

        # 需要转换
        logger.info(f"开始转换: {file_ext} -> {target_format}")

        # 在线程池中执行转换
        converted_path = await asyncio.to_thread(
            convert_audio_format,
            temp_input_path,
            target_format,
            Config.TEMP_DIR
        )

        # 确定 MIME 类型
        mime_types = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'flac': 'audio/flac'
        }
        media_type = mime_types.get(target_format, 'application/octet-stream')

        # 生成输出文件名
        output_filename = Path(original_filename).stem + f".{target_format}"

        logger.info(f"转换完成: {output_filename}")

        # 返回转换后的文件
        response = FileResponse(
            path=converted_path,
            filename=output_filename,
            media_type=media_type
        )

        # 后台删除临时文件
        async def cleanup():
            await asyncio.sleep(2)
            try:
                if temp_input_path.exists():
                    temp_input_path.unlink()
                    logger.info(f"已清理输入文件: {temp_input_path}")
                if converted_path.exists():
                    converted_path.unlink()
                    logger.info(f"已清理转换文件: {converted_path}")
            except Exception as e:
                logger.error(f"清理临时文件失败: {str(e)}")

        asyncio.create_task(cleanup())
        return response

    except RuntimeError as e:
        # 格式转换失败
        logger.error(f"格式转换失败: {str(e)}")

        # 清理临时文件
        if temp_input_path.exists():
            temp_input_path.unlink()

        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"转换请求处理失败: {str(e)}")

        # 清理临时文件
        if 'temp_input_path' in locals() and temp_input_path.exists():
            temp_input_path.unlink()

        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    if task_id not in state.tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    upload_files = list(Config.UPLOAD_DIR.glob(f"{task_id}_*"))
    for f in upload_files:
        f.unlink(missing_ok=True)

    output_dir = Config.OUTPUT_DIR / task_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    del state.tasks[task_id]

    return {"message": "任务已删除"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )