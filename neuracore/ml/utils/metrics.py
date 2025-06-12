import json
import logging
import os
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

from google.cloud import firestore

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Log and store training metrics."""

    @abstractmethod
    def log_training_metrics(self, metrics: dict[str, float], epoch: int):
        pass

    @abstractmethod
    def log_validation_metrics(self, metrics: dict[str, float], epoch: int):
        pass


class LocalMetricsLogger:
    """Log and store training metrics."""

    def __init__(
        self,
        local_dir: str = "./output",
        log_every: int = 1,
    ):
        self.local_dir = Path(local_dir)
        self.log_every = log_every
        self.train_metrics: list[dict[str, Any]] = []
        self.val_metrics: list[dict[str, Any]] = []
        # Load existing metrics if resuming
        self._load_existing_metrics()

    def log_training_metrics(self, metrics: dict[str, float], epoch: int):
        """Log training metrics for current epoch."""
        metric_entry = {"epoch": epoch, "timestamp": time.time(), **metrics}
        self.train_metrics.append(metric_entry)
        if epoch % self.log_every == 0:
            self._save_metrics("train")

    def log_validation_metrics(self, metrics: dict[str, float], epoch: int):
        """Log validation metrics for current epoch."""
        metric_entry = {"epoch": epoch, "timestamp": time.time(), **metrics}
        self.val_metrics.append(metric_entry)
        if epoch % self.log_every == 0:
            self._save_metrics("validation")

    def _load_existing_metrics(self):
        """Load existing metrics when resuming training."""
        try:
            train_path = self.local_dir / "train_metrics.json"
            val_path = self.local_dir / "validation_metrics.json"
            if train_path.exists():
                with open(train_path) as f:
                    self.train_metrics = json.load(f)
            if val_path.exists():
                with open(val_path) as f:
                    self.val_metrics = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing metrics: {str(e)}")

    def _save_metrics(self, metric_type: str):
        """Save metrics to storage."""
        try:
            metrics_list = (
                self.train_metrics if metric_type == "train" else self.val_metrics
            )
            self.local_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.local_dir / f"{metric_type}_metrics.json"
            with open(save_path, "w") as f:
                json.dump(metrics_list, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save {metric_type} metrics: {str(e)}")


class GCPMetricsLogger:
    """Log and store training metrics."""

    def __init__(
        self,
        database_name: str,
        org_id: str,
        training_id: str,
        log_every: int = 1,
    ):
        self.database_name = database_name
        self.org_id = org_id
        self.training_id = training_id
        self.log_every = log_every
        self.db = firestore.Client(
            project=os.environ["GCP_PROJECT"], database=database_name
        )
        self.train_metrics: list[dict[str, Any]] = []
        self.val_metrics: list[dict[str, Any]] = []

    def log_training_metrics(self, metrics: dict[str, float], epoch: int):
        """Log training metrics for current epoch."""
        metric_entry = {"epoch": epoch, "timestamp": time.time(), **metrics}
        self.train_metrics.append(metric_entry)
        if epoch % self.log_every == 0:
            self._save_metrics("train")
            self.train_metrics.clear()

    def log_validation_metrics(self, metrics: dict[str, float], epoch: int):
        """Log validation metrics for current epoch."""
        metric_entry = {"epoch": epoch, "timestamp": time.time(), **metrics}
        self.val_metrics.append(metric_entry)
        if epoch % self.log_every == 0:
            self._save_metrics("validation")
            self.val_metrics.clear()

    def _get_metrics_collection(
        self, metric_type: str
    ) -> firestore.CollectionReference:
        return (
            self.db.collection("organizations")
            .document(self.org_id)
            .collection("training")
            .document(self.training_id)
            .collection(metric_type)
        )

    def _save_metrics(self, metric_type: str):
        """Save metrics to storage."""
        try:
            metrics_list = (
                self.train_metrics if metric_type == "train" else self.val_metrics
            )
            for metric in metrics_list:
                self._get_metrics_collection(metric_type).document(
                    str(metric["timestamp"])
                ).set(metric)
        except Exception as e:
            logger.error(f"Failed to save {metric_type} metrics: {str(e)}")
