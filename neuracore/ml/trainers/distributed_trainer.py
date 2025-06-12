import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from neuracore_training.memory_monitor import MemoryMonitor, OutOfMemoryError
from neuracore_training.metrics import MetricsLogger
from neuracore_training.storage_handler import StorageHandler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from neuracore.ml import BatchedTrainingOutputs, NeuracoreModel

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Trainer for distributed multi-GPU training on a single node."""

    def __init__(
        self,
        model: NeuracoreModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        metrics_logger: MetricsLogger,
        storage_handler: StorageHandler,
        output_dir: Path,
        algorithm_config: dict[str, Any],
        num_epochs: int,
        save_freq: int = 1,
        validate_freq: int = 1,
        save_checkpoints: bool = True,
        clip_grad_norm: Optional[float] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Initialize the distributed trainer.

        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            metrics_logger: Logger for metrics
            storage_handler: Handler for model storage
            output_dir: Directory for output files
            algorithm_config: Configuration for the algorithm
            num_epochs: Number of epochs to train
            save_freq: Frequency to save checkpoints (in epochs)
            validate_freq: Frequency to run validation (in epochs)
            save_checkpoints: Whether to save checkpoints
            clip_grad_norm: Maximum norm for gradient clipping
            rank: Rank of this process
            world_size: Total number of processes/GPUs
        """
        self.device = torch.device(
            f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Process {rank} using device: {self.device}")

        # Set up the model for distributed training
        self.model = model.to(self.device)
        if torch.cuda.is_available() and world_size > 1:
            self.model = DDP(
                self.model, device_ids=[rank], find_unused_parameters=False
            )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics_logger = metrics_logger
        self.storage_handler = storage_handler
        self.output_dir = output_dir
        self.algorithm_config = algorithm_config
        self.num_epochs = num_epochs
        self.save_freq = save_freq
        self.validate_freq = validate_freq
        self.save_checkpoints = save_checkpoints
        self.clip_grad_norm = clip_grad_norm
        self.rank = rank
        self.world_size = world_size

        # Configure optimizer
        if hasattr(model, "configure_optimizers"):
            self.optimizers = model.configure_optimizers()
        else:
            self.optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3)]

        # Initialize best metrics
        self.best_val_loss = float("inf")

        # Create checkpoint directory
        if rank == 0:
            self.checkpoint_dir = output_dir / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> dict[str, float]:
        """Run one epoch of training."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = []

        memory_monitor = MemoryMonitor(
            max_ram_utilization=0.8, max_gpu_utilization=0.95
        )

        for batch in tqdm(self.train_loader, desc="Training", disable=self.rank != 0):

            memory_monitor.check_memory()

            # Move tensors to device and format batch
            batch = batch.to(self.device)

            # Forward pass
            batch_output: BatchedTrainingOutputs = self.model.training_step(batch)
            loss = sum(batch_output.losses.values()).mean()

            # Backward pass
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            loss.backward()

            # Clip gradients if configured
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            for optimizer in self.optimizers:
                optimizer.step()

            epoch_losses.append({
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in batch_output.losses.items()
            })
            epoch_metrics.append({
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in batch_output.metrics.items()
            })

        # Average metrics
        avg_metrics = {}
        for key in epoch_losses[0].keys():
            avg_metrics[key] = sum(x[key] for x in epoch_losses) / len(epoch_losses)

        # If distributed, synchronize metrics across processes
        if self.world_size > 1:
            avg_metrics = self._gather_metrics(avg_metrics)

        return avg_metrics

    def validate(self) -> dict[str, float]:
        """Run validation."""
        self.model.train()  # So we can get losses
        val_losses = []
        val_metrics = []

        with torch.no_grad():
            for batch in tqdm(
                self.val_loader, desc="Validating", disable=self.rank != 0
            ):
                batch = batch.to(self.device)

                # Forward pass
                batch_output: BatchedTrainingOutputs = self.model.training_step(batch)

                val_losses.append({
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in batch_output.losses.items()
                })
                val_metrics.append({
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in batch_output.metrics.items()
                })

        # Average metrics
        avg_metrics = {}
        for key in val_losses[0].keys():
            avg_metrics[key] = sum(x[key] for x in val_losses) / len(val_losses)

        # If distributed, synchronize metrics across processes
        if self.world_size > 1:
            avg_metrics = self._gather_metrics(avg_metrics)

        return avg_metrics

    def train(self, start_epoch: int = 0) -> None:
        """Run the training loop.

        Args:
            start_epoch: Epoch to start from (for resuming training)
        """
        if self.rank == 0:
            self.storage_handler.save_metadata(0)

        try:
            start_epoch = max(start_epoch, 1)
            for epoch in range(start_epoch, self.num_epochs + 1):
                # Set epoch for distributed sampler to
                # ensure different shuffling each epoch
                if isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)

                # Training phase
                if self.rank == 0:
                    logger.info(f"Starting Epoch {epoch}")
                    epoch_start_time = time.time()

                train_metrics = self.train_epoch()

                if self.rank == 0:
                    logger.info(
                        f"Epoch {epoch} took {time.time() - epoch_start_time:.2f}s"
                    )

                # Log training metrics (only from rank 0)
                if self.rank == 0:
                    self.metrics_logger.log_training_metrics(train_metrics, epoch=epoch)
                    log_str = "Train - "
                    log_str += ", ".join(
                        f"{k}: {v:.4f}" for k, v in train_metrics.items()
                    )
                    logger.info(log_str)

                # Save checkpoint and artifacts periodically (only from rank 0)
                if self.rank == 0 and epoch % self.save_freq == 0:
                    # Only save checkpoints with rank 0
                    reduced_loss = sum(train_metrics.values()) / len(train_metrics)
                    is_best = reduced_loss < self.best_val_loss
                    self.save_checkpoint(epoch, train_metrics, is_best=is_best)

                    # Save model artifacts
                    self.storage_handler.save_model_artifacts(
                        model=self.get_model_without_ddp(),
                        output_dir=self.output_dir,
                    )

                # Validation phase
                if epoch % self.validate_freq == 0:

                    if self.rank == 0:
                        logger.info(f"Starting Validation for Epoch {epoch}")
                        val_start_t = time.time()

                    val_metrics = self.validate()

                    if self.rank == 0:
                        logger.info(
                            f"Val, epoch {epoch} took {time.time() - val_start_t:.2f}s"
                        )

                    # Log validation metrics (only from rank 0)
                    if self.rank == 0:
                        self.metrics_logger.log_validation_metrics(
                            val_metrics, epoch=epoch
                        )
                        log_str = "Val - "
                        log_str += ", ".join(
                            f"{k}: {v:.4f}" for k, v in val_metrics.items()
                        )
                        logger.info(log_str)

                        # Update learning rate based on validation loss
                        reduced_loss = sum(val_metrics.values()) / len(val_metrics)

                        # Update best validation loss
                        if reduced_loss < self.best_val_loss:
                            self.best_val_loss = reduced_loss

                # Save metadata
                if self.rank == 0:
                    self.storage_handler.save_metadata(epoch)

        except OutOfMemoryError:
            error_msg = (
                f"Batch size {self.train_loader.batch_size} is too large. "
                "Try reducing batch size or using a more powerful machine."
            )
            logger.error(error_msg)
            if self.rank == 0:
                self.storage_handler.save_error(error_msg)
            raise  # Re-raise to ensure proper exit code
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            error_msg = f"Error during training. \n{traceback.format_exc()}"
            if self.rank == 0:
                self.storage_handler.save_error(error_msg)
            raise  # Re-raise to ensure proper exit code

    def _gather_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Gather and average metrics from all processes.

        Args:
            metrics: Local metrics from this process

        Returns:
            Averaged metrics across all processes
        """
        if self.world_size <= 1:
            return metrics

        averaged_metrics = {}
        for key, value in metrics.items():
            tensor = torch.tensor([value], device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            averaged_metrics[key] = (tensor / self.world_size).item()

        return averaged_metrics

    def get_model_without_ddp(self):
        """Get the model without DDP wrapper."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False) -> None:
        """Save checkpoint with metadata."""
        if not self.save_checkpoints or self.rank != 0:
            return
        logger.info("Saving checkpoint...")

        # Get the model state dict (different for DDP vs non-DDP models)
        model_state = self.get_model_without_ddp().state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_states": [opt.state_dict() for opt in self.optimizers],
            "metrics": metrics,
            "best_val_loss": self.best_val_loss,
            "algorithm_config": self.algorithm_config,
        }

        # Save regular checkpoint
        self.storage_handler.save_checkpoint(checkpoint, "checkpoint_latest.pt")

        # Save best model if needed
        if is_best:
            self.storage_handler.save_checkpoint(checkpoint, "checkpoint_best.pt")
        logger.info("... checkpoint saved!")

    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint and restore training state."""
        checkpoint = self.storage_handler.load_checkpoint(path)

        # Handle model loading (different for DDP vs non-DDP models)
        self.get_model_without_ddp().load_state_dict(checkpoint["model_state"])
        for optimizer, opt_state in zip(
            self.optimizers, checkpoint["optimizer_states"]
        ):
            optimizer.load_state_dict(opt_state)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if "algorithm_config" in checkpoint:
            self.algorithm_config = checkpoint["algorithm_config"]

        return checkpoint


def setup_distributed(rank: int, world_size: int) -> None:
    """Initialize the distributed process group.

    Args:
        rank: Rank of this process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Set device for this process
    torch.cuda.set_device(rank)

    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    logger.info(f"Initialized process group for rank {rank}/{world_size}")


def cleanup_distributed() -> None:
    """Clean up the distributed process group."""
    dist.destroy_process_group()
    logger.info("Destroyed process group")
