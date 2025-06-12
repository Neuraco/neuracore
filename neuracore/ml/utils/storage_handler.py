import json
import logging
import os
import tempfile
import zipfile
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from google.cloud import firestore, storage
from torch import nn

from neuracore.ml import NeuracoreModel
from neuracore.ml.utils.mar import create_mar
from neuracore.ml.utils.validate import AlgorthmCheck

FILE_PATH = Path(__file__).parent / "handlers.py"

logger = logging.getLogger(__name__)


class StorageHandler:
    """Handles storage operations for both local and GCS."""

    @abstractmethod
    def save_checkpoint(self, checkpoint: dict, checkpoint_name: str) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_name: str) -> dict:
        pass

    @abstractmethod
    def save_model_artifacts(self, model: nn.Module, output_dir: Path) -> None:
        pass

    @abstractmethod
    def save_metadata(self, epoch: int) -> None:
        pass

    @abstractmethod
    def save_error(self, error_message: str) -> None:
        """Save error message from failed training."""
        pass

    @abstractmethod
    def save_algorithm_validation_error(self, error_message: str) -> None:
        """Save error message from failed algorithm validation."""
        pass

    @abstractmethod
    def download_algorithm(
        self, algorithm_dir: Path, extract_dir: Optional[Path] = None
    ) -> Path:
        """
        Download and extract algorithm code from storage.

        Args:
            algorithm_dir: Directory to save algorithm code to
            extract_dir: Directory to extract algorithm code to (optional)

        Returns:
            Tuple of (extract_path, algorithm_config)
        """
        pass


class LocalStorageHandler(StorageHandler):
    """Handles storage operations for local."""

    def __init__(
        self,
        local_dir: str = "./output",
    ):
        self.local_dir = Path(local_dir)

    def save_checkpoint(self, checkpoint: dict, checkpoint_name: str) -> None:
        """Save checkpoint to storage."""
        save_path = self.local_dir / checkpoint_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_name: str) -> dict:
        """Load checkpoint from storage."""
        load_path = self.local_dir / checkpoint_name
        return torch.load(load_path)

    def save_model_artifacts(self, model: NeuracoreModel, output_dir: Path) -> None:
        """Save model artifacts for either local or GCS storage."""
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        create_mar(model=model, output_dir=artifacts_dir)

    def save_metadata(self, epoch: int) -> None:
        """Save metrics to storage."""
        pass

    def save_error(self, error_message: str) -> None:
        """Save error message to storage."""
        error_path = self.local_dir / "error.txt"
        with open(error_path, "w") as f:
            f.write(error_message)

            # Also include stack trace if available
            import traceback

            f.write("\n\nStack trace:\n")
            f.write(traceback.format_exc())

    def save_algorithm_validation_check(
        self, checklist: AlgorthmCheck, error_message: str
    ) -> None:
        success = all(list(checklist.model_dump().values()))
        dict_to_save = checklist.model_dump()
        if success:
            dict_to_save["status"] = "success"
        else:
            dict_to_save["status"] = "failed"
            dict_to_save["error"] = error_message
        self.save_error(json.dumps(dict_to_save, indent=2))

    def download_algorithm(
        self, algorithm_dir: Path, extract_dir: Optional[Path] = None
    ) -> Path:
        """
        Download algorithm code for local development scenario.

        For local testing, we assume the algorithm is already available in the local
        algorithms directory.

        Args:
            algorithm_dir: Directory to save algorithm code to
            extract_dir: Directory to extract algorithm code to (optional)

        Returns:
            Tuple of (extract_path, algorithm_config)
        """
        if not algorithm_dir.exists():
            raise FileNotFoundError(
                f"Algorithm not found at {algorithm_dir}. "
                f"For local testing, please place the algorithm "
                f"zip file in {algorithm_dir}/algorithm.zip"
            )

        zip_path = algorithm_dir / "algorithm.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Algorithm zip not found at {zip_path}")

        # Extract to a temporary directory if not specified
        if extract_dir is None:
            extract_dir = Path(tempfile.mkdtemp()) / "algorithm"
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Extract the algorithm zip
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        return extract_dir


class GCPStorageHandler(StorageHandler):
    """Handles storage operations for local."""

    def __init__(
        self,
        bucket_name: str,
        database_name: str,
        org_id: str,
        training_id: str,
        algorithm_id: str = None,
    ):
        self.bucket_name = bucket_name
        self.database_name = database_name
        self.org_id = org_id
        self.training_id = training_id
        self.algorithm_id = algorithm_id
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.db = firestore.Client(
            project=os.environ["GCP_PROJECT"], database=database_name
        )

    def save_checkpoint(self, checkpoint: dict, checkpoint_name: str) -> None:
        """Save checkpoint to storage."""
        tmp_path = Path("tmp_checkpoint.pt")
        torch.save(checkpoint, tmp_path)
        path = (
            f"organizations/{self.org_id}/training/"
            f"{self.training_id}/checkpoints/{checkpoint_name}"
        )
        blob = self.bucket.blob(path)
        blob.upload_from_filename(tmp_path)
        tmp_path.unlink()

    def load_checkpoint(self, checkpoint_name: str) -> dict:
        """Load checkpoint from storage."""
        tmp_path = Path("tmp_checkpoint.pt")
        path = (
            f"organizations/{self.org_id}/training/"
            f"{self.training_id}/checkpoints/{checkpoint_name}"
        )
        blob = self.bucket.blob(path)
        blob.download_to_filename(tmp_path)
        checkpoint = torch.load(tmp_path)
        tmp_path.unlink()
        return checkpoint

    def save_model_artifacts(self, model: NeuracoreModel, output_dir: Path) -> None:
        """Save model artifacts for either local or GCS storage."""
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        create_mar(model=model, output_dir=artifacts_dir)

        for file_path in artifacts_dir.glob("*"):
            blob_path = f"organizations/{self.org_id}/training/{self.training_id}"
            blob = self.bucket.blob(f"{blob_path}/{file_path.name}")
            blob.upload_from_filename(str(file_path))

    def save_metadata(self, epoch: int) -> None:
        """Save metrics to storage."""
        if epoch == 0:
            doc = (
                self.db.collection("organizations")
                .document(self.org_id)
                .collection("training")
                .document(self.training_id)
            )
            update_dict = {
                "epoch": epoch,
                "start_time": datetime.now().timestamp(),
            }
            if not doc.get().exists:
                doc.set(update_dict)
            else:
                doc.update(update_dict)
        else:
            self.db.collection("organizations").document(self.org_id).collection(
                "training"
            ).document(self.training_id).update({
                "epoch": epoch,
            })

    def save_error(self, error_message: str) -> None:
        """Save error message to storage."""
        # Save to Firestore
        self.db.collection("organizations").document(self.org_id).collection(
            "training"
        ).document(self.training_id).update({
            "error": error_message,
            "status": "failed",
        })

    def save_algorithm_validation_check(
        self, checklist: AlgorthmCheck, error_message: str
    ) -> None:
        assert self.algorithm_id is not None, "Algorithm ID not provided"
        # check if all values in model (checklist) are true
        success = all(list(checklist.model_dump().values()))
        dict_to_save = {
            "validation_checklist": checklist.model_dump(),
        }
        if success:
            dict_to_save["validation_status"] = "available"
        else:
            dict_to_save["validation_status"] = "validation_failed"
            dict_to_save["validation_message"] = error_message

        self.db.collection("organizations").document(self.org_id).collection(
            "algorithms"
        ).document(self.algorithm_id).update(dict_to_save)

    def download_algorithm(
        self, algorithm_dir: Path, extract_dir: Optional[Path] = None
    ) -> Path:
        """
        Download and extract algorithm code from GCS.

        Args:
            algorithm_dir: Directory to save algorithm code to
            extract_dir: Directory to extract algorithm code to (optional)

        Returns:
            Tuple of (extract_path, algorithm_config)
        """
        logger = logging.getLogger(__name__)

        # GCS path to algorithm zip
        gcs_path = f"{algorithm_dir}/algorithm.zip"

        # Create a temporary directory for the zip file if extract_dir not provided
        if extract_dir is None:
            extract_dir = Path(tempfile.mkdtemp()) / "algorithm"
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Path for the local zip file
        local_zip_path = extract_dir / "algorithm.zip"

        # Download the zip file
        logger.info(f"Downloading algorithm from {gcs_path} to {local_zip_path}")
        blob = self.bucket.blob(gcs_path)

        if not blob.exists():
            raise FileNotFoundError(f"Algorithm not found at GCS path: {gcs_path}")

        blob.download_to_filename(local_zip_path)

        # Extract the zip file
        logger.info(f"Extracting algorithm to {extract_dir}")
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Clean up the zip file
        local_zip_path.unlink()

        return extract_dir
