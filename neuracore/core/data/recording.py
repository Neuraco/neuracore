"""TODO"""

from typing import TYPE_CHECKING, Optional

from neuracore.core.data.synced_recording import SynchronizedRecording

from ..exceptions import DatasetError
from ..nc_types import DataType

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset


class Recording:
    """Iterator for streaming synchronized data from a single recording episode.

    This class provides efficient streaming access to robot demonstration data
    including video frames from multiple cameras, depth data, and sensor
    information. It manages concurrent video streams and synchronizes data
    according to the episode's timestamp information.
    """

    def __init__(self, dataset: "Dataset", recording_id: str, size_bytes: int):
        """Initialize episode iterator for a specific recording.

        Args:
            dataset: Parent Dataset instance.
            recording: Recording dictionary containing episode metadata.
        """
        self.dataset = dataset
        self.id = recording_id
        self.size_bytes = size_bytes

    def synchronize(
        self,
        frequency: int = 0,
        data_types: Optional[list[DataType]] = None,
    ) -> SynchronizedRecording:
        """Synchronize the episode with specified frequency and data types.

        Args:
            frequency: Frequency at which to synchronize the episode.
            data_types: List of DataType to include in synchronization.
                If None, uses the default data types from the recording.

        Raises:
            DatasetError: If synchronization fails.
        """
        if frequency <= 0:
            raise DatasetError("Frequency must be greater than 0")
        return SynchronizedRecording(
            dataset=self.dataset,
            recording_id=self.id,
            frequency=frequency,
            data_types=data_types or [],
        )
