"""
logger.py 定义了日志记录模块

    @Time    : 2025/05/11
    @Author  : JackWang
    @File    : logger.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
import os
import sys
import datetime
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager

# Third-Party Library
import rich
from loguru import logger
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.style import Style
from rich.console import Console

# Torch Library

# My Library


class RichProgressLogger:
    def __init__(
        self,
        log_dir: str = "logs",
        exp_name: Optional[str] = None,
        log_level: str = "INFO",
        enable_file_logging: bool = True,
        rich_console: Optional[Console] = None,
    ):
        """
        增强版训练日志记录器，结合loguru和rich实现丰富的终端输出

        Args:
            log_dir: 日志文件保存目录
            exp_name: 实验名称，用于命名日志文件
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
            enable_file_logging: 是否启用文件日志
            rich_console: 可传入自定义的rich Console实例
        """
        self.log_dir = log_dir
        self.exp_name = exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_level = log_level
        self.enable_file_logging = enable_file_logging

        # 初始化rich console
        self.console = rich_console or Console()

        # 配置loguru
        self._configure_loguru()

        # 进度条相关状态
        self._progress = None
        self._epoch_progress = None
        self._batch_progress = None
        self._live = None
        self._info_panel = None
        self._current_info = {}

        # 训练状态标志
        self._is_training = False

    def _configure_loguru(self):
        """配置loguru日志记录器"""
        # 移除默认handler
        logger.remove()

        # 控制台输出配置 (使用rich处理颜色)
        logger.add(
            sys.stdout,
            level=self.log_level,
            format="<level>{message}</level>",
            enqueue=True,  # 线程安全
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        # 文件日志配置
        if self.enable_file_logging:
            os.makedirs(self.log_dir, exist_ok=True)
            log_file = os.path.join(self.log_dir, f"{self.exp_name}.log")
            logger.add(
                log_file,
                level=self.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
                enqueue=True,
                backtrace=True,
                diagnose=True,
                rotation="10 MB",  # 日志轮转
                retention="7 days",  # 保留7天
                compression="zip",  # 压缩归档
            )

    def setup_progress(self, total_epochs: int, total_batches: int):
        """初始化训练进度条"""
        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=self.console,
            refresh_per_second=10,
        )

        # Batch进度条 (上方)
        self._batch_progress = self._progress.add_task(
            "[magenta]Batch Progress", total=total_batches
        )

        # Epoch进度条 (下方)
        self._epoch_progress = self._progress.add_task(
            "[cyan]Epoch Progress", total=total_epochs
        )

        # 信息面板
        self._info_panel = Panel("", title="Training Info", border_style="blue")

        # Live显示
        self._live = Live(
            self._get_layout(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )

    def _get_layout(self):
        """获取rich布局"""
        # 创建布局表格
        layout = Table.grid(padding=(0, 1))
        layout.add_row(self._info_panel)
        layout.add_row(self._progress)
        return layout

    def update_info(self, info_dict: dict[str, Any]):
        """更新训练信息显示"""
        self._current_info.update(info_dict)

        # 创建信息表格
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        for key, value in self._current_info.items():
            info_table.add_row(
                Text(f"{key}:", style="bold"),
                str(value),
            )

        self._info_panel.renderable = info_table

        # 如果正在训练，则更新显示
        if self._is_training and self._live:
            self._live.update(self._get_layout())

    def log(self, level: str, message: str):
        """记录日志"""
        logger.log(level, message)

        # 如果不在训练中，直接打印到控制台
        if not self._is_training:
            self.console.print(message)

    def debug(self, message: str):
        self.log("DEBUG", message)

    def info(self, message: str):
        self.log("INFO", message)

    def warning(self, message: str):
        self.log("WARNING", message)

    def error(self, message: str):
        self.log("ERROR", message)

    def critical(self, message: str):
        self.log("CRITICAL", message)

    def log_config(self, config: dict[str, Any]):
        """记录配置信息"""
        self.info("Experiment Configuration:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")

    def start_training(self):
        """开始训练，显示进度条"""
        if self._progress is None:
            raise ValueError("Progress not initialized. Call setup_progress() first.")

        self._is_training = True
        self._live.start()

    def end_training(self):
        """结束训练，关闭进度条"""
        self._is_training = False
        if self._live:
            self._live.stop()

    def update_batch(self, completed: int):
        """更新batch进度"""
        if self._batch_progress is not None:
            self._progress.update(self._batch_progress, completed=completed)

    def update_epoch(self, completed: int):
        """更新epoch进度"""
        if self._epoch_progress is not None:
            self._progress.update(self._epoch_progress, completed=completed)

    def reset_batch_progress(self, total_batches: Optional[int] = None):
        """重置batch进度条"""
        if self._batch_progress is not None:
            if total_batches is not None:
                self._progress.reset(
                    self._batch_progress,
                    total=total_batches,
                    description="[magenta]Batch Progress",
                )
            else:
                self._progress.reset(self._batch_progress)

    @contextmanager
    def training_context(self, total_epochs: int, total_batches: int):
        """
        训练上下文管理器，自动处理进度条的启动和关闭

        用法:
        with logger.training_context(total_epochs=10, total_batches=100):
            # 训练代码
            logger.update_batch(1)
            logger.update_epoch(1)
        """
        self.setup_progress(total_epochs, total_batches)
        self.start_training()
        try:
            yield
        finally:
            self.end_training()

    def print_metrics(self, metrics: dict[str, float], prefix: str = ""):
        """美观地打印指标字典"""
        table = Table(title=f"{prefix} Metrics", box=None)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right")

        for name, value in metrics.items():
            table.add_row(name, f"{value:.4f}")

        self.console.print(table)

    def exception(self, message: str):
        """记录异常信息"""
        logger.exception(message)
