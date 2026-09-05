# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import torch
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard import SummaryWriter

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import Color, device_module, device_type, NoColor


# named tuple for passing device memory stats for logging
DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


class DeviceMemoryMonitor:
    def __init__(self, device: str = f"{device_type}:0"):
        # pyrefly: ignore [read-only]
        self.device = torch.device(device)  # device object
        self.device_name = device_module.get_device_name(self.device)
        self.device_index = device_module.current_device()
        self.device_capacity = device_module.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        device_module.reset_peak_memory_stats()
        device_module.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        device_info = device_module.memory_stats(self.device)

        max_active = device_info.get("active_bytes.all.peak", -1)
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = device_info.get("reserved_bytes.all.peak", -1)
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = device_info.get("num_alloc_retries", -1)
        num_ooms = device_info.get("num_ooms", -1)

        if num_retries > 0:
            logger.warning(
                f"{num_retries} {device_type.upper()} memory allocation retries."
            )
        if num_ooms > 0:
            logger.warning(f"{num_ooms} {device_type.upper()} OOM errors thrown.")

        return DeviceMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self):
        device_module.reset_peak_memory_stats()


def build_device_memory_monitor():
    device_memory_monitor = DeviceMemoryMonitor(device_type)
    logger.info(
        f"{device_type.upper()} capacity: {device_memory_monitor.device_name} "
        f"with {device_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )
    return device_memory_monitor


class BaseLogger:
    """Logger that does nothing, used when logging is disabled."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def log_histogram(self, name: str, values: Sequence[float], step: int) -> None:
        pass

    def log_histogram_bins(
        self, name: str, bin_edges: Sequence[float], counts: Sequence[float], step: int
    ) -> None:
        """Publish an already-binned histogram.

        `bin_edges` has one more entry than `counts`. Unlike `log_histogram`,
        the caller controls what a bin means, so a bin can be an expert index
        rather than a range of values.
        """

    def log_image(self, name: str, image_chw: torch.Tensor, step: int) -> None:
        """Publish an RGB image given as a float `[3, H, W]` tensor in [0, 1]."""

    def log_text(self, name: str, text: str, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    """Logger implementation for TensorBoard."""

    def __init__(self, log_dir: str, tag: str | None = None):
        self.tag = tag
        self.writer = SummaryWriter(log_dir, max_queue=1000)
        logger.info(f"TensorBoard logging enabled. Logs will be saved at {log_dir}")

    def _tag(self, name: str) -> str:
        return name if self.tag is None else f"{self.tag}/{name}"

    def log(self, metrics: dict[str, Any], step: int) -> None:
        summary = Summary()
        for key, value in metrics.items():
            summary.value.add(
                tag=key if self.tag is None else f"{self.tag}/{key}",
                simple_value=float(value),
            )

        # Submit one event instead of one queue operation per scalar.
        file_writer = cast(Any, self.writer._get_file_writer())
        file_writer.add_summary(summary, step)

    def log_histogram(self, name: str, values: Sequence[float], step: int) -> None:
        self.writer.add_histogram(
            self._tag(name),
            torch.tensor(values, dtype=torch.float32),
            step,
        )

    def log_histogram_bins(
        self, name: str, bin_edges: Sequence[float], counts: Sequence[float], step: int
    ) -> None:
        # add_histogram would re-bin a sample population by magnitude. The raw
        # form takes the buckets as given, which is what lets a bucket stand for
        # an expert index. The moments describe the bin-index distribution
        # weighted by count, so TensorBoard's summary stats stay meaningful.
        total = float(sum(counts))
        centers = [
            (bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(len(counts))
        ]
        self.writer.add_histogram_raw(
            tag=self._tag(name),
            min=bin_edges[0],
            max=bin_edges[-1],
            num=total,
            sum=sum(c * w for c, w in zip(centers, counts)),
            sum_squares=sum(c * c * w for c, w in zip(centers, counts)),
            bucket_limits=list(bin_edges[1:]),
            bucket_counts=list(counts),
            global_step=step,
        )

    def log_image(self, name: str, image_chw: torch.Tensor, step: int) -> None:
        self.writer.add_image(self._tag(name), image_chw, step, dataformats="CHW")

    def log_text(self, name: str, text: str, step: int) -> None:
        self.writer.add_text(self._tag(name), text, step)

    def close(self) -> None:
        self.writer.close()


class WandBLogger(BaseLogger):
    """Logger implementation for Weights & Biases."""

    def __init__(
        self,
        log_dir: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        # Import wandb here to avoid startup import
        import wandb

        self.wandb = wandb
        self.tag = tag

        # Create logging directory
        os.makedirs(log_dir, exist_ok=True)

        self.wandb.init(
            entity=os.getenv("WANDB_TEAM", None),
            project=os.getenv("WANDB_PROJECT", "torchtitan"),
            name=os.getenv("WANDB_RUN_NAME", None),
            id=os.getenv("WANDB_RUN_ID", None),
            notes=os.getenv("WANDB_RUN_NOTES", None),
            tags=os.getenv("WANDB_RUN_TAGS", None),
            group=os.getenv("WANDB_RUN_GROUP", None),
            job_type=os.getenv("WANDB_RUN_JOB_TYPE", None),
            resume_from=os.getenv("WANDB_RESUME_FROM", None),
            fork_from=os.getenv("WANDB_FORK_FROM", None),
            dir=log_dir,
            config=config_dict,
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        wandb_metrics = {
            (k if self.tag is None else f"{self.tag}/{k}"): v
            for k, v in metrics.items()
        }
        self.wandb.log(wandb_metrics, step=step)

    def _tag(self, name: str) -> str:
        return name if self.tag is None else f"{self.tag}/{name}"

    def log_histogram(self, name: str, values: Sequence[float], step: int) -> None:
        self.wandb.log(
            {self._tag(name): self.wandb.Histogram(list(values))}, step=step
        )

    def log_histogram_bins(
        self, name: str, bin_edges: Sequence[float], counts: Sequence[float], step: int
    ) -> None:
        self.wandb.log(
            {
                self._tag(name): self.wandb.Histogram(
                    np_histogram=(list(counts), list(bin_edges))
                )
            },
            step=step,
        )

    def log_image(self, name: str, image_chw: torch.Tensor, step: int) -> None:
        self.wandb.log(
            {self._tag(name): self.wandb.Image(image_chw.permute(1, 2, 0).numpy())},
            step=step,
        )

    def log_text(self, name: str, text: str, step: int) -> None:
        self.wandb.log({self._tag(name): text}, step=step)

    def close(self) -> None:
        if self.wandb.run is not None:
            self.wandb.finish()


class LoggerContainer(BaseLogger):
    """Container to call all loggers enabled in the job config."""

    def __init__(self) -> None:
        self._loggers: list[BaseLogger] = []

    def add_logger(self, logger_instance: BaseLogger) -> None:
        self._loggers.append(logger_instance)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log(metrics, step)

    def log_histogram(self, name: str, values: Sequence[float], step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log_histogram(name, values, step)

    def log_histogram_bins(
        self, name: str, bin_edges: Sequence[float], counts: Sequence[float], step: int
    ) -> None:
        for logger_instance in self._loggers:
            logger_instance.log_histogram_bins(name, bin_edges, counts, step)

    def log_image(self, name: str, image_chw: torch.Tensor, step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log_image(name, image_chw, step)

    def log_text(self, name: str, text: str, step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log_text(name, text, step)

    @property
    def number_of_loggers(self) -> int:
        return len(self._loggers)

    def close(self) -> None:
        for logger_instance in self._loggers:
            logger_instance.close()


def ensure_pp_loss_visible(
    *, parallel_dims: ParallelDims, pp_schedule: str, color: Color | NoColor
) -> None:
    """
    Ensures that the loss is visible on the console for pipeline-parallel training.

    For pipeline-parallel training, the loss is only visible on the last pipeline stage.
    This function checks if the appropriate rank is included in the LOG_RANK environment
    variable and warns if it's not.
    """

    # V Block Schedules return loss on rank 0
    if pp_schedule == "ZBVZeroBubble":
        return

    # Calculate the rank where loss is visible (first rank of the last pipeline stage)
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    loss_visible_rank = (world_size // pp_size) * (pp_size - 1)

    # Check if the loss-visible rank is included in LOG_RANK environment variable
    env_logged_ranks = os.environ.get("LOG_RANK", "").split(",")
    if env_logged_ranks == [""]:
        env_logged_ranks = []

    if str(loss_visible_rank) not in env_logged_ranks:
        logger.warning(
            f"{color.red}Pipeline Parallel loss is not visible. "
            f"Please add {color.yellow}rank {loss_visible_rank}{color.red} "
            f"to LOG_RANK environment variable in run_train.sh.{color.reset}"
        )


def _get_metrics_rank(
    *,
    parallel_dims: ParallelDims,
    pp_schedule: str,
) -> int:
    """
    Determines which rank should log metrics.

    Returns:
       int: The rank responsible for logging metrics:
            - Rank 0 for non-pipeline-parallel configs
            - Rank 0 for pipeline-parallel 'ZBVZeroBubble' schedule
            - The first rank of the last pipeline stage for other pipeline-parallel schedules
    """
    # Early return for non-pipeline-parallel configurations
    if not parallel_dims.pp_enabled:
        return 0

    # V Block Schedules return loss on rank 0
    if pp_schedule == "ZBVZeroBubble":
        return 0

    # Calculate first rank of the last pipeline stage
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    return (world_size // pp_size) * (pp_size - 1)


@dataclass(kw_only=True, slots=True)
class TensorLoggingConfig(Configurable.Config):
    """Controls model-tensor statistics recorded with ordinary metrics."""

    enabled: bool = False
    """Whether to record registered model-tensor statistics."""

    freq: int = 5
    """Requested step interval; regular publication follows its least common multiple
    with `log_freq`.

    Eager non-logging steps skip tensor-statistic calls. Compiled and captured
    graphs retain the calls, but a device gate skips reading the tensor and updating
    statistic buffers. Code that prepares an optional logged value can still run.
    Cross-rank reduction runs only on logging steps, outside the compiled or captured
    graph.
    """

    publish_filter_regex: str = (
        r"\.(?:numel|nonfinite_count|abs_mean|square_mean|abs_max)$"
    )
    """Regex allowlist for final metric names.

    Possible suffixes are `numel`, `nonfinite_count`, `observation_count`,
    `zero_count`, `zero_frac`, `abs_sum`, `abs_mean`, `square_mean`, `rms`,
    `kurtosis`, and `abs_max`.

    The regex runs after collection and derivation. It changes only what is published.
    """

    vector_metrics: bool = False
    """Whether to record width-preserving vector metrics such as MoE expert load.

    Vector metrics are reduced element-wise, so a per-expert imbalance ratio
    stays correct under EP and CP. Collapsing the same counts into the scalar
    path would lose the per-element identity the ratio depends on.
    """

    publish_vector_elements: bool = False
    """Whether to publish one scalar per vector element in addition to summaries.

    Off by default: a 64-expert model with 32 MoE layers would emit 2048 extra
    series per logging step. Derived `total`, `max`, `min`, `mean` and
    `imbalance` values are published either way.
    """

    publish_vector_distributions: bool = False
    """Whether to publish histogram and heatmap views of each vector metric.

    Scalar series show drift over time; these show the shape of the
    distribution at a step. One histogram per vector plus one image stacking
    sibling layers carries what `publish_vector_elements` would need thousands
    of scalar series to convey.
    """

    moe_shape_manifest: str = ""
    """Path to write the static MoE grouped-GEMM shape manifest, or empty to skip.

    Written once at setup. Grouped-GEMM `K` and `N` come from the model config,
    so recording them needs no per-step instrumentation; the per-expert token
    counts supply the matching `M`.
    """

    def __post_init__(self) -> None:
        if self.freq <= 0:
            raise ValueError("metrics.tensor_logging.freq must be positive")


class MetricsProcessor(Configurable):
    """Metrics processor to processes the metrics and log metrics.

    The current MetricsProcessor log some metrics to STDOUT and some metrics to
    TensorBoard or WandB.

    Args:
        config (Config): Metrics configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        dump_folder (str): Base folder for log output.
        pp_schedule (str): Pipeline parallel schedule name.
        ft_enable (bool): Whether fault tolerance is enabled.
        ft_replica_id (int): Fault tolerance replica ID.
        config_dict (dict | None): Full job config dict for WandB logging.
        tag (str | None): Tag to use for TensorBoard or WandB. Defaults to None.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        log_freq: int = 10
        """How often to log metrics to TensorBoard, in iterations"""

        enable_tensorboard: bool = False
        """Whether to log metrics to TensorBoard"""

        disable_color_printing: bool = False
        """Whether to disable color printing in logs"""

        save_tb_folder: str = "tb"
        """Folder to dump TensorBoard states"""

        save_for_all_ranks: bool = False
        """
        Whether to save TensorBoard/Wandb metrics only for rank 0 or for all ranks.
        When this option is False and pipeline_parallel_degree is > 1, the metrics
        component uses the 0th rank of the last stage pipeline group, which is the
        only stage that computes loss metrics.
        """

        enable_wandb: bool = False
        """Whether to log metrics to Weights & Biases"""

        tensor_logging: TensorLoggingConfig = field(default_factory=TensorLoggingConfig)

    config: Config
    logger: BaseLogger
    parallel_dims: ParallelDims
    device_memory_monitor: DeviceMemoryMonitor
    color: utils.NoColor | utils.Color

    gpu_peak_flops: float
    ntokens_since_last_log: int
    data_loading_times: list[float]
    time_last_log: float

    num_flops_per_token: int
    has_quantization: bool
    optimizers: OptimizersContainer | None
    model_parts: list[torch.nn.Module] | None

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        dump_folder: str = "./outputs",
        pp_schedule: str = "1F1B",
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
        has_quantization: bool = False,
    ):
        self.logger = self._build_metric_logger(
            config=config,
            parallel_dims=parallel_dims,
            dump_folder=dump_folder,
            pp_schedule=pp_schedule,
            ft_enable=ft_enable,
            ft_replica_id=ft_replica_id,
            config_dict=config_dict,
            tag=tag,
        )
        self.parallel_dims = parallel_dims
        self.config = config
        self.device_memory_monitor = build_device_memory_monitor()
        # used for colorful printing
        self.color = utils.NoColor() if config.disable_color_printing else utils.Color()

        self.gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )
        self.ntokens_since_last_log = 0
        self.data_loading_times = []
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

        self.has_quantization = has_quantization

        # These variables have to be set later as they depend on other components or model.
        self.num_flops_per_token = -1
        self.optimizers = None
        self.model_parts = None

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.config.log_freq == 0

    def _build_metric_logger(
        self,
        *,
        config: Config,
        parallel_dims: ParallelDims,
        dump_folder: str,
        pp_schedule: str,
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> BaseLogger:
        """
        Build an appropriate metric logger based on configuration.
        """
        # Log initial config state
        logger.debug(
            f"Building logger with config: wandb={config.enable_wandb}, "
            f"tensorboard={config.enable_tensorboard}"
        )

        # Check if any logging backend is enabled
        has_logging_enabled = config.enable_tensorboard or config.enable_wandb

        # Determine if this rank should log
        should_log = has_logging_enabled
        if (not config.save_for_all_ranks) and should_log:
            metrics_rank = _get_metrics_rank(
                parallel_dims=parallel_dims, pp_schedule=pp_schedule
            )
            should_log = torch.distributed.get_rank() == metrics_rank

        logger.debug(
            f"Logging decision: has_logging_enabled={has_logging_enabled}, should_log={should_log}"
        )

        if not should_log:
            logger.debug("Returning BaseLogger due to should_log=False")
            return BaseLogger()

        # Setup logging directory
        base_log_dir = os.path.join(
            dump_folder,
            config.save_tb_folder,
            datetime.now().strftime("%Y%m%d-%H%M"),
        )

        if ft_enable:
            base_log_dir = os.path.join(
                base_log_dir,
                f"replica_{ft_replica_id}",
            )

        if config.save_for_all_ranks:
            base_log_dir = os.path.join(
                base_log_dir, f"rank_{torch.distributed.get_rank()}"
            )

        # Create logger container
        logger_container = LoggerContainer()

        # Create loggers in priority order
        if config.enable_wandb:
            logger.debug("Attempting to create WandB logger")
            try:
                wandb_logger = WandBLogger(
                    base_log_dir, config_dict=config_dict, tag=tag
                )
                logger_container.add_logger(wandb_logger)
            except Exception as e:
                if "No module named 'wandb'" in str(e):
                    logger.error(
                        "Failed to create WandB logger: No module named 'wandb'. Please install it using 'pip install wandb'."
                    )
                else:
                    logger.error(f"Failed to create WandB logger: {e}")

        if config.enable_tensorboard:
            logger.debug("Creating TensorBoard logger")
            tensorboard_logger = TensorBoardLogger(base_log_dir, tag)
            logger_container.add_logger(tensorboard_logger)

        if logger_container.number_of_loggers == 0:
            logger.debug("No loggers enabled, returning an empty LoggerContainer")
        return logger_container

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ):
        """
        Log training metrics including loss, throughput, and memory statistics.

        Args:
            step: Current training step
            global_avg_loss: Global average loss across all valid tokens on all ranks
                Defined as global_loss_sum / global_valid_tokens
            global_max_loss: Maximum local loss across all ranks
                Defined as max(local_loss_sum / local_valid_tokens)
            grad_norm: Gradient norm after clipping
            extra_metrics: Optional additional metrics to log

        """
        assert self.num_flops_per_token > 0, "num_flops_per_token must be set"

        time_delta = time.perf_counter() - self.time_last_log

        # tokens per second per device, abbreviated as tps
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )
        # model FLOPS utilization
        # For its definition and calculation, please refer to the PaLM paper:
        # https://arxiv.org/abs/2204.02311
        # MFU is based on BF16 peak FLOPS which is misleading when quantization
        # (FP8/MX) is active, so we skip it in that case.
        tflops = self.num_flops_per_token * tps / 1e12
        if self.has_quantization:
            mfu = None
        else:
            mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops

        time_end_to_end = time_delta / self.config.log_freq
        time_data_loading = sum(self.data_loading_times) / len(self.data_loading_times)
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        metrics = {
            "loss_metrics/global_avg_loss": global_avg_loss,
            "loss_metrics/global_max_loss": global_max_loss,
            "grad_norm": grad_norm,
            "throughput(tps)": tps,
            "tflops": tflops,
            "time_metrics/end_to_end(s)": time_end_to_end,
            "time_metrics/data_loading(s)": time_data_loading,
            "time_metrics/data_loading(%)": time_data_loading_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }
        if mfu is not None:
            metrics["mfu(%)"] = mfu

        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        color = self.color
        mfu_str = f"{mfu:.2f}%" if mfu is not None else "N/A"
        logger.info(
            f"{color.red}step: {step:2}  "
            f"{color.green}loss: {global_avg_loss:8.5f}  "
            f"{color.orange}grad_norm: {grad_norm:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}  "
            f"{color.cyan}tflops: {tflops:,.2f}  "
            f"{color.magenta}mfu: {mfu_str}{color.reset}"
        )

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def log_validation(
        self, loss: float, step: int, extra_metrics: dict[str, Any] | None = None
    ):
        time_delta = time.perf_counter() - self.time_last_log

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        # tokens per second per device, abbreviated as tps
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )

        metrics = {
            "validation_metrics/loss": loss,
            "validation_metrics/throughput(tps)": tps,
            "validation_metrics/memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "validation_metrics/memory/max_active(%)": device_mem_stats.max_active_pct,
            "validation_metrics/memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "validation_metrics/memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
        }

        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.yellow}validate step: {step:2}  "
            f"{color.green}loss: {loss:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}{color.reset}"
        )

        self.ntokens_since_last_log = 0
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def close(self):
        self.logger.close()
