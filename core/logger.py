"""
logger.py 定义了日志记录模块

    @Time    : 2025/05/11
    @Author  : JackWang
    @File    : logger.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from __future__ import annotations

import os
import math
from io import StringIO
from pathlib import Path
from typing import Optional
from functools import partial
from contextlib import contextmanager

# Third-Party Library
import loguru
from rich import box
from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.layout import Layout
from rich.abc import RichRenderable
from rich.console import Console, ConsoleOptions, RenderResult

# Torch Library

# My Library


class Header:
    """控制台标题"""

    def __init__(self, title: Optional[str] = None):
        self.title = (
            "Weakly Supervised Semantic Segmentation (WSSS) Training"
            if title is None
            else title
        )

    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_row(self.title)
        return Panel(grid, border_style="blue")


class RenderableConsole(Console):
    """
    rich原始的console可以记录打印的内容, 而后以多种格式导出, 所以非常适合
    用于logger.
    但是关键问题在于rich原始的console不是renderable的, 所以没有办法
    把它放到一个Layout/Panel中, 所以这里通过 Rich Protocol对其进行扩展
    """

    def __init__(self, *args, **kwargs):
        console_file = open(os.devnull, "w")
        super().__init__(record=True, file=console_file, *args, **kwargs)

        self._segments: list[Text] = []

    def begin_capture(self) -> None:
        self._enter_buffer()

    def end_capture(self) -> str:
        render_result = self._render_buffer(self._buffer)
        # del self._buffer[:]
        self._exit_buffer()
        return render_result

    def print(self, *args, **kwargs):
        with self.capture() as capture:
            super().print(*args, **kwargs)
        self._segments.extend([Text(i) for i in capture.get().split("\n") if i != ""])
        max_segments = 1000
        if len(self._segments) > max_segments:
            self._segments = self._segments[-max_segments:]

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Text("\n").join(self._segments[-options.height :])


class RichuruLogger:

    def __init__(
        self,
        epochs: int,
        batches: int,
        log_dir: Path | str = Path(__file__).resolve().parents[1] / "logs",
    ):

        self.epochs = epochs
        self.batches = batches

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.file_console = Console(file=(log_dir / "running.log").open(mode="a"))
        self.terminal_console = Console(stderr=True)

        self._setup_layout()
        self.setup_header()
        self._setup_main()
        self._setup_footer()
        self._setup_logger()

        self.live = Live(
            self.layout,
            console=self.terminal_console,
            refresh_per_second=10,
            transient=False,
        )

    def _get_level_color(self, level: str) -> str:
        return {
            "INFO": "white",
            "WARNING": "bold yellow",
            "ERROR": "red",
            "CRITICAL": "bold red",
            "SUCCESS": "bold green",
            "DEBUG": "bold blue",
        }.get(level, "white")

    def sink(
        self,
        message: loguru.Message,
        console: Optional[Console],
    ) -> None:
        """Sink function to handle log messages."""
        if "rich" in message.record["extra"]:
            console.print(message.record["extra"]["rich"])
        else:
            msg = f"[cyan]{message.record['time']:%Y-%d-%b@%H:%M:%S}[/cyan]│ "
            level = message.record["level"].name
            color = self._get_level_color(level)
            msg += f"[{color}]{message.record['message']}[/]"
            console.print(Text.from_markup(str(msg)), end="\n")

    def _setup_logger(self) -> None:
        from loguru import logger

        self.logger = logger
        self.logger.remove()

        # terminal handler
        self.logger.add(
            level="DEBUG",
            format="{time:YYYY-D-MMMM@HH:mm:ss}│ {message}",
            sink=partial(self.sink, console=self.main_console),
        )
        # file handler
        self.logger.add(
            level="DEBUG",
            format="{time:YYYY-D-MMMM@HH:mm:ss}│ {message}",
            sink=partial(self.sink, console=self.file_console),
        )

    def _setup_layout(self) -> Layout:
        if hasattr(self, "layout"):
            return self.layout

        self.layout = Layout(name="root")
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=6),
        )
        return self.layout

    def setup_header(self, title: Optional[str] = None) -> Header:
        if hasattr(self, "header"):
            return self.header
        self.header = Header(title)
        self.layout["header"].update(self.header)
        return self.header

    def _setup_main(self) -> RenderableConsole:
        self.main_console = RenderableConsole(color_system="truecolor")

        self.main_panel = Panel(
            self.main_console,
            title="Log Messages",
            box=box.SIMPLE_HEAD,
        )
        self.layout["main"].update(self.main_panel)
        return self.main_console

    def _setup_footer(self) -> Table:
        # sourcery skip: class-extract-method
        self.footer_table = Table.grid(expand=True)
        self.footer_table.add_column(justify="left", ratio=1)
        self.footer_table.add_column(justify="center", ratio=1)

        self.info_panel = Panel(
            Text("Information will be displayed here"),
            title="Information",
            border_style="green",
        )
        self.progress_panel = Panel(
            self._setup_progress(), title="[b]Training Progress", border_style="red"
        )
        self.footer_table.add_row(self.info_panel, self.progress_panel)
        self.layout["footer"].update(self.footer_table)
        return self.footer_table

    def _setup_info_panel(
        self, rich_renderable: Optional[RichRenderable] = None
    ) -> RichRenderable:
        previous_renderable = self.info_panel.renderable
        if rich_renderable is not None:
            self.info_panel.renderable = rich_renderable
        return previous_renderable

    def _setup_progress(self) -> Progress:
        self.progress = Progress(
            "{task.description}",
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[green]{task.completed}/{task.total}[/]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        self.batch_task = self.progress.add_task("[magenta]Batch", total=self.batches)
        self.epoch_task = self.progress.add_task("[cyan]Epoch", total=self.epochs)
        self.total_task = self.progress.add_task(
            "[green]Total", total=self.batches * self.epochs
        )
        return self.progress

    def make_info_table(self, info_dict: dict[str, float]) -> Table:
        table = Table.grid(expand=True)
        info_per_row = math.ceil(len(info_dict) / 3)

        for col_idx in range(info_per_row * 2):
            table.add_column(justify="left" if col_idx % 2 == 0 else "center", ratio=1)

        row = []
        for info_idx, (key, value) in enumerate(info_dict.items()):
            row.extend([Text(f"{key}", style="bold"), Text(f"{value:.2f}")])
            if (info_idx + 1) % info_per_row == 0:
                table.add_row(*row)
                row = []
        table.add_row(*row)
        return table

    def export_html(self):
        self.main_console.save_html(
            self.log_dir / "report.html",
        )

    @contextmanager
    def training_context(self):
        """用于训练的上下文管理器"""

        with self.live:
            yield self

    def update_batch(
        self,
        info_dict: Optional[dict[str, float]] = None,
        rich_renderable: Optional[RichRenderable] = None,
    ) -> None:
        assert info_dict is not None or rich_renderable is not None

        if info_dict is not None and isinstance(info_dict, dict):
            self._setup_info_panel(self.make_info_table(info_dict))
        elif rich_renderable is not None and isinstance(
            rich_renderable, RichRenderable
        ):
            self._setup_info_panel(rich_renderable)

        self.progress.update(self.batch_task, advance=1)
        self.progress.update(self.total_task, advance=1)

    def update_epoch(
        self,
        info_dict: Optional[dict[str, float]] = None,
        rich_renderable: Optional[RichRenderable] = None,
    ) -> None:
        assert info_dict is not None or rich_renderable is not None

        if info_dict is not None and isinstance(info_dict, dict):
            self._setup_info_panel(self.make_info_table(info_dict))
        elif rich_renderable is not None:
            self._setup_info_panel(rich_renderable)

        if self.progress._tasks[self.epoch_task].completed < self.epochs - 1:
            self.progress.reset(self.batch_task, total=self.batches)
            self.progress.update(self.batch_task, completed=0)
        self.progress.update(self.epoch_task, advance=1)

    def info(self, message: str | RichRenderable):
        if isinstance(message, RichRenderable):
            self.logger.bind(rich=message).info(message)
        else:
            self.logger.opt(colors=True).info(message)

    def warning(self, message: str | RichRenderable):
        if isinstance(message, RichRenderable):
            self.logger.bind(rich=message).warning(message)
        else:
            self.logger.opt(colors=True).warning(message)

    def error(self, message: str | RichRenderable):
        if isinstance(message, RichRenderable):
            self.logger.bind(rich=message).error(message)
        else:
            self.logger.opt(colors=True).error(message)

    def success(self, message: str | RichRenderable):
        if isinstance(message, RichRenderable):
            self.logger.bind(rich=message).success(message)
        else:
            self.logger.opt(colors=True).success(message)

    def debug(self, message: str | RichRenderable):
        if isinstance(message, RichRenderable):
            self.logger.bind(rich=message).debug(message)
        else:
            self.logger.opt(colors=True).debug(message)

    def critical(self, message: str | RichRenderable):
        if isinstance(message, RichRenderable):
            self.logger.bind(rich=message).critical(message)
        else:
            self.logger.opt(colors=True).critical(message)


if __name__ == "__main__":
    import time
    import random

    logger = RichuruLogger(
        epochs=10,
        batches=100,
        log_dir=Path(__file__).resolve().parents[1] / "test/logs-example",
    )

    with logger.training_context():
        for epoch in range(logger.epochs):
            for batch in range(logger.batches):
                time.sleep(0.01)
                logger.success(
                    f"Epoch {epoch + 1}/{logger.epochs}, Batch {batch + 1}/{logger.batches}"
                )

                logger.update_batch(
                    info_dict={
                        "loss": random.random(),
                        "accuracy": random.random(),
                    },
                )
                logger.info("[dark_orange3]This is a test message[/]")

            logger.update_epoch(
                info_dict={
                    "loss": random.random(),
                    "accuracy": random.random(),
                },
            )

    logger.export_html()
