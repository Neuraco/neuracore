"""Dataset management and streaming for Neuracore robot recordings.

This module provides classes for managing datasets, streaming episodes,
and iterating over synchronized robot data including video frames and
sensor information. It supports both organizational and shared datasets
with efficient streaming capabilities.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

import requests
from tqdm import tqdm

from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_dataset import SynchronizedDataset

from ..auth import Auth, get_auth
from ..const import API_URL
from ..exceptions import DatasetError
from ..nc_types import DataType, SyncedDataset

DEFAULT_CACHE_DIR = Path(tempfile.gettempdir() + "/neuracore_cache")


logger = logging.getLogger(__name__)


class Dataset:
    """Represents a dataset containing robot demonstration recordings.

    This class provides access to collections of robot recordings that can be
    streamed for analysis or used for training machine learning models. It
    supports both organizational and shared datasets with efficient iteration
    over episodes and synchronized data access.
    """

    def __init__(
        self,
        id: str,
        name: str,
        size_bytes: int,
        tags: list[str],
        is_shared: bool,
        recordings: Optional[list[dict]] = None,
    ):
        """Initialize a dataset from server response data.

        Args:
        """
        self.id = id
        self.name = name
        self.size_bytes = size_bytes
        self.tags = tags
        self.is_shared = is_shared
        self.recordings = recordings or self._get_recordings()
        self.num_recordings = len(self.recordings)
        self._recording_idx = 0
        self.cache_dir = DEFAULT_CACHE_DIR

    def _get_recordings(self) -> list[dict]:
        """Get the list of recordings in the dataset."""
        auth = get_auth()
        response = requests.get(
            f"{API_URL}/datasets/{self.id}/recordings", headers=auth.get_headers()
        )
        response.raise_for_status()
        data = response.json()
        return data["recordings"]

    @staticmethod
    def get(name: str, non_exist_ok: bool = False) -> Optional["Dataset"]:
        """Retrieve an existing dataset by name.

        Searches through both organizational and shared datasets to find
        a dataset with the specified name.

        Args:
            name: Name of the dataset to retrieve.
            non_exist_ok: If True, returns None when dataset is not found
                instead of raising an exception.

        Returns:
            The Dataset instance if found, or None if non_exist_ok is True
            and the dataset doesn't exist.

        Raises:
            DatasetError: If the dataset is not found and non_exist_ok is False.
        """
        auth: Auth = get_auth()
        req = requests.get(
            f"{API_URL}/datasets/by-name/{name}",
            headers=auth.get_headers(),
        )
        if req.status_code != 200:
            if non_exist_ok:
                return None
            raise DatasetError(f"Dataset '{name}' not found.")
        dataset_json = req.json()
        return Dataset(
            id=dataset_json["id"],
            name=dataset_json["name"],
            size_bytes=dataset_json["size_bytes"],
            tags=dataset_json["tags"],
            is_shared=dataset_json["is_shared"],
        )

    @staticmethod
    def create(
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        shared: bool = False,
    ) -> "Dataset":
        """Create a new dataset or return existing one with the same name.

        Creates a new dataset with the specified parameters. If a dataset
        with the same name already exists, returns the existing dataset
        instead of creating a duplicate.

        Args:
            name: Unique name for the dataset.
            description: Optional description of the dataset contents and purpose.
            tags: Optional list of tags for organizing and searching datasets.
            shared: Whether the dataset should be shared/open-source.
                Note that setting shared=True is only available to specific
                members allocated by the Neuracore team.

        Returns:
            The newly created Dataset instance, or existing dataset if
            name already exists.
        """
        ds = Dataset.get(name, non_exist_ok=True)
        if ds is None:
            ds = Dataset._create_dataset(name, description, tags, shared=shared)
        else:
            logger.info(f"Dataset '{name}' already exist.")
        return ds

    @staticmethod
    def _create_dataset(
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        shared: bool = False,
    ) -> "Dataset":
        """Create a new dataset via API call.

        Args:
            name: Unique name for the dataset.
            description: Optional description of the dataset.
            tags: Optional list of tags for the dataset.
            shared: Whether the dataset should be shared.
                Note that setting shared=True is only available to specific
                members allocated by the Neuracore team.

        Returns:
            The newly created Dataset instance.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        auth: Auth = get_auth()
        response = requests.post(
            f"{API_URL}/datasets",
            headers=auth.get_headers(),
            json={
                "name": name,
                "description": description,
                "tags": tags,
                "is_shared": shared,
            },
        )
        response.raise_for_status()
        dataset_json = response.json()
        return Dataset(
            id=dataset_json["id"],
            name=dataset_json["name"],
            size_bytes=dataset_json["size_bytes"],
            tags=dataset_json["tags"],
            is_shared=dataset_json["is_shared"],
        )

    def _synchronize(
        self, frequency: int = 0, data_types: Optional[list[DataType]] = None
    ) -> SyncedDataset:
        response = requests.post(
            f"{API_URL}/synchronize/synchronize-dataset",
            headers=get_auth().get_headers(),
            json={
                "dataset_id": self.id,
                "frequency": frequency,
                "data_types": data_types,
            },
        )
        response.raise_for_status()
        dataset_json = response.json()
        return SyncedDataset.model_validate(dataset_json)

    def synchronize(
        self, frequency: int = 0, data_types: Optional[list[DataType]] = None
    ) -> SynchronizedDataset:
        synced_dataset = self._synchronize(frequency=frequency, data_types=data_types)
        total = synced_dataset.num_demonstrations
        processed = synced_dataset.num_processed_demonstrations
        if total != processed:
            pbar = tqdm(total=total, desc="Synchronizing dataset", unit="episode")
            pbar.n = processed
            pbar.refresh()
            while processed < total:
                time.sleep(5.0)
                synced_dataset = self._synchronize(
                    frequency=frequency, data_types=data_types
                )
                new_processed = synced_dataset.num_processed_demonstrations
                if new_processed > processed:
                    pbar.update(new_processed - processed)
                    processed = new_processed
            pbar.close()
        else:
            logger.info("Dataset is already synchronized.")
        return SynchronizedDataset(
            dataset=self,
            frequency=frequency,
            data_types=data_types,
            dataset_description=synced_dataset.dataset_description,
        )

    def __iter__(self) -> "Dataset":
        """Initialize iterator over episodes in the dataset.

        Returns:
            Self for iteration over episodes.
        """
        return self

    def __len__(self) -> int:
        """Get the number of episodes in the dataset.

        Returns:
            Number of demonstration episodes in the dataset.
        """
        return self.num_recordings

    def __getitem__(self, idx) -> Union[Recording, "Dataset"]:
        """Support for indexing and slicing dataset episodes.

        Args:
            idx: Integer index or slice object for accessing episodes.

        Returns:
            For integer indices: EpisodeIterator for the specified episode.
            For slices: New Dataset containing the selected episodes.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the index is not an integer or slice.
        """
        if isinstance(idx, slice):
            # Handle slice
            recordings = self.recordings[idx.start : idx.stop : idx.step]
            ds = Dataset(
                id=self.id,
                name=self.name,
                size_bytes=self.size_bytes,
                tags=self.tags,
                is_shared=self.is_shared,
                recordings=recordings,
            )
            return ds
        else:
            # Handle single index
            if isinstance(idx, int):
                if idx < 0:  # Handle negative indices
                    idx += len(self.recordings)
                if not 0 <= idx < len(self.recordings):
                    raise IndexError("Dataset index out of range")
                return Recording(
                    self,
                    self.recordings[idx]["id"],
                    self.recordings[idx]["total_bytes"],
                )
            raise TypeError(
                f"Dataset indices must be integers or slices, not {type(idx)}"
            )

    def __next__(self):
        """Get the next episode in the dataset iteration.

        Returns:
            EpisodeIterator for the next episode.

        Raises:
            StopIteration: When all episodes have been processed.
        """
        if self._recording_idx >= len(self.recordings):
            raise StopIteration

        recording = self.recordings[self._recording_idx]
        self._recording_idx += 1  # Increment counter
        return Recording(self, recording["id"], recording["total_bytes"])
