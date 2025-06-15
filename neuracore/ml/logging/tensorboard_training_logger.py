"""TensorBoard-based training logger with cloud sync capabilities.

This module provides a unified logging interface similar to Weights & Biases
but using TensorBoard for visualization. Supports both local and cloud modes
with automatic syncing to GCS buckets.
"""

import json
import logging
import tempfile
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import requests
import torch
from torch.utils.tensorboard import SummaryWriter

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.ml.logging.training_logger import TrainingLogger

logger = logging.getLogger(__name__)


class LoggingMode(Enum):
    """Enum for logging modes."""

    LOCAL = "local"
    CLOUD = "cloud"


class TensorboardTrainingLogger(TrainingLogger):
    """TensorBoard-based logger with optional cloud synchronization.

    Provides a W&B-like interface for logging training metrics, images, and other
    data to TensorBoard. Supports both local-only logging and cloud sync to GCS.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: LoggingMode = LoggingMode.LOCAL,
        sync_interval: int = 60,
        comment: str = "",
        purge_step: Optional[int] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
    ):
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save logs. If None, creates temp directory.
            experiment_name: Name of the experiment (used for organization).
            run_name: Name of this specific run.
            config: Hyperparameters and configuration to log.
            mode: "local" for local-only, "cloud" for cloud sync.
            sync_interval: Seconds between cloud syncs (if mode="cloud").
            comment: Comment to append to the log directory name.
            purge_step: Step to purge events before (TensorBoard parameter).
            max_queue: Maximum queue size for TensorBoard writer.
            flush_secs: Seconds between flushes to disk.
            filename_suffix: Suffix for the log filename.
        """
        self.mode = mode
        self.sync_interval = sync_interval
        self.experiment_name = experiment_name or "default_experiment"
        self.run_name = run_name or f"run_{int(time.time())}"
        self.config = config or {}

        # Set up log directory
        if log_dir is None:
            if mode == LoggingMode.LOCAL:
                log_dir = f"./tensorboard_logs/{self.experiment_name}/{self.run_name}"
            else:
                log_dir = tempfile.mkdtemp(prefix=f"tensorboard_{self.run_name}_")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )

        # Step tracking
        self.global_step = 0
        self._step_lock = threading.Lock()

        # Cloud sync setup
        self._sync_thread = None
        self._stop_sync = threading.Event()
        self._last_sync_time = 0

        self._logging_upload_dir: str = ""
        if mode == LoggingMode.CLOUD:
            training_id = None
            upload_url_response = requests.get(
                f"{API_URL}/training/{training_id}/resumable_upload_url",
                headers=get_auth().get_headers(),
            )
            upload_url_response.raise_for_status()
            self._logging_upload_dir = upload_url_response.json()["url"]
            # self._start_cloud_sync()

        # Log initial configuration
        if self.config:
            self.log_hyperparameters(self.config)

        logger.info(f"TensorBoard logger initialized: {self.log_dir}")
        if mode == "local":
            logger.info(f"View logs with: tensorboard --logdir {self.log_dir}")

    def _get_step(self, step: Optional[int]) -> int:
        """Get the step number, auto-incrementing if not provided."""
        if step is not None:
            return step

        with self._step_lock:
            current_step = self.global_step
            self.global_step += 1
            return current_step

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar metric.

        Args:
            name: Name of the metric (e.g., "train/loss", "val/accuracy").
            value: Scalar value to log.
            step: Training step. If None, uses auto-incrementing global step.
        """
        step = self._get_step(step)
        self.writer.add_scalar(name, value, step)

    def log_scalars(
        self, scalars: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple scalar metrics at once.

        Args:
            scalars: Dictionary of metric name -> value.
            step: Training step. If None, uses auto-incrementing global step.
        """
        step = self._get_step(step)
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)

    def log_image(
        self,
        name: str,
        image: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
        dataformats: str = "CHW",
    ) -> None:
        """Log an image.

        Args:
            name: Name for the image.
            image: Image data as numpy array or torch tensor.
            step: Training step. If None, uses auto-incrementing global step.
            dataformats: Format of the image data (e.g., "CHW", "HWC").
        """
        step = self._get_step(step)
        self.writer.add_image(name, image, step, dataformats=dataformats)

    def log_images(
        self,
        name: str,
        images: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
        dataformats: str = "NCHW",
    ) -> None:
        """Log multiple images as a grid.

        Args:
            name: Name for the image grid.
            images: Batch of images as numpy array or torch tensor.
            step: Training step. If None, uses auto-incrementing global step.
            dataformats: Format of the image data (e.g., "NCHW").
        """
        step = self._get_step(step)
        self.writer.add_images(name, images, step, dataformats=dataformats)

    def log_histogram(
        self,
        name: str,
        values: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
    ) -> None:
        """Log a histogram of values.

        Args:
            name: Name for the histogram.
            values: Values to create histogram from.
            step: Training step. If None, uses auto-incrementing global step.
            bins: Binning method ("tensorflow", "auto", "fd", etc.).
        """
        step = self._get_step(step)
        self.writer.add_histogram(name, values, step)

    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        """Log text data.

        Args:
            name: Name for the text data.
            text: Text string to log.
            step: Training step. If None, uses auto-incrementing global step.
        """
        step = self._get_step(step)
        self.writer.add_text(name, text, step)

    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log hyperparameters and optionally metrics.

        Args:
            hparams: Dictionary of hyperparameters.
            metrics: Optional dictionary of metrics to associate with hparams.
        """
        # Convert any non-primitive types to strings for TensorBoard compatibility
        clean_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                clean_hparams[key] = value
            else:
                clean_hparams[key] = str(value)

        self.writer.add_hparams(clean_hparams, metrics or {})

        # Also save as JSON for easier access
        config_file = self.log_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(hparams, f, indent=2, default=str)

    def log_model_graph(
        self, model: torch.nn.Module, input_to_model: Optional[torch.Tensor] = None
    ) -> None:
        """Log the model computational graph.

        Args:
            model: PyTorch model to log.
            input_to_model: Example input tensor for the model.
        """
        self.writer.add_graph(model, input_to_model)

    # def _start_cloud_sync(self) -> None:
    #     """Start background thread for cloud synchronization."""
    #     if self._sync_thread is not None:
    #         return

    #     self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
    #     self._sync_thread.start()
    #     logger.info("Started cloud sync thread")

    # def _sync_loop(self) -> None:
    #     """Background sync loop for cloud mode."""
    #     while not self._stop_sync.is_set():
    #         try:
    #             current_time = time.time()
    #             if current_time - self._last_sync_time >= self.sync_interval:
    #                 self._sync_to_cloud()
    #                 self._last_sync_time = current_time
    #         except Exception as e:
    #             logger.error(f"Error during cloud sync: {e}")

    #         # Wait with ability to be interrupted
    #         self._stop_sync.wait(min(10, self.sync_interval))

    # def _sync_to_cloud(self) -> None:
    #     """Sync TensorBoard logs to cloud storage."""
    #     try:
    #         # Flush any pending writes
    #         self.writer.flush()

    #         # TODO: We need to upload any changes; how do we do this?

    #         logger.debug(
    #             f"Synced logs to cloud: {self.experiment_name}/{self.run_name}"
    #         )

    #     except Exception as e:
    #         logger.error(f"Failed to sync logs to cloud: {e}")

    def flush(self) -> None:
        """Force flush any pending writes to disk."""
        self.writer.flush()

    def close(self) -> None:
        """Close the logger and clean up resources."""
        if self._sync_thread is not None:
            self._stop_sync.set()
            self._sync_thread.join(timeout=30)

        # Final sync if in cloud mode
        if self.mode == "cloud" and self.storage_handler is not None:
            self._sync_to_cloud()

        self.writer.close()
        logger.info("TensorBoard logger closed")

    def get_log_dir(self) -> str:
        """Get the log directory path."""
        return str(self.log_dir)
