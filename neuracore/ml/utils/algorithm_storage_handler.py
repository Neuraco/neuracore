"""Handles storage operations for algorithms."""

import logging
import zipfile
from pathlib import Path
from typing import Optional

import requests

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL

FILE_PATH = Path(__file__).parent / "handlers.py"

logger = logging.getLogger(__name__)


class AlgorithmStorageHandler:
    """Handles storage operations for algorithms."""

    def __init__(self, algorithm_id: Optional[str] = None):
        """Initialize the AlgorithmStorageHandler.

        Args:
            algorithm_id: Optional ID of the algorithm to manage.
                If provided, will enable cloud logging and validation.
        """
        self.algorithm_id = algorithm_id
        self.log_to_cloud = self.algorithm_id is not None
        if self.log_to_cloud:
            response = requests.get(
                f"{API_URL}/algorithm/{self.algorithm_id}",
                headers=get_auth().get_headers(),
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Algorithm {self.algorithm_id} not found or access denied."
                )

    def save_algorithm_validation_error(self, error_message: str) -> None:
        """Save error message from failed algorithm validation.

        Args:
            error_message: Error message to save.
        """
        if self.log_to_cloud:
            response = requests.post(
                f"{API_URL}/algorithm/{self.algorithm_id}/validation-error",
                headers=get_auth().get_headers(),
                json={"error_message": error_message},
            )
            if response.status_code != 200:
                logger.error(
                    f"Failed to save algorithm validation error: {response.text}"
                )

    def download_algorithm(self, extract_dir: Path) -> None:
        """Download and extract algorithm code from storage.

        Args:
            extract_dir: Directory to extract algorithm code to.
        """
        if self.log_to_cloud:
            raise NotImplementedError(
                "Local storage mode is not implemented for algorithm download."
            )

        # Path for the local zip file
        local_zip_path = extract_dir / "algorithm.zip"

        # Extract the zip file
        logger.info(f"Extracting algorithm to {extract_dir}")
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Clean up the zip file
        local_zip_path.unlink()
