import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import av
import av.datasets
import numpy as np
import torch
from neuracore_training.memory_monitor import MemoryMonitor
from PIL import Image

from neuracore.core.data.cache_manager import CacheManager
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.core.nc_types import CameraData, DataType, JointData
from neuracore.core.utils.depth_utils import rgb_to_depth
from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.datasets.pytorch_neuracore_dataset import (
    PytorchNeuracoreDataset as BaseNeuracoreDataset,
)

logger = logging.getLogger(__name__)


# Single training sample is identical type, but with no batch dimension
TrainingSample = BatchedTrainingSamples


class PytorchSynchronizedDataset(BaseNeuracoreDataset):
    """Dataset for loading episodic robot data from GCS with filesystem caching."""

    def __init__(
        self,
        synchronized_dataset: SynchronizedDataset,
        input_data_types: list[DataType],
        output_data_types: list[DataType],
        output_prediction_horizon: int,
        cache_dir: Optional[str] = None,
        max_cache_percent: float = 80.0,
        cache_check_interval: int = 100,
        tokenize_text: Optional[
            Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ):
        if not isinstance(synchronized_dataset, SynchronizedDataset):
            raise TypeError(
                "synchronized_dataset must be an instance of SynchronizedDataset"
            )
        super().__init__(
            input_data_types=input_data_types,
            output_data_types=output_data_types,
            output_prediction_horizon=output_prediction_horizon,
            tokenize_text=tokenize_text,
        )
        self.synchronized_dataset = synchronized_dataset
        self.dataset_description = self.synchronized_dataset.dataset_description

        # Setup cache
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "episodic_dataset_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_manager = CacheManager(
            self.cache_dir,
            max_usage_percent=max_cache_percent,
            check_interval=cache_check_interval,  # Configurable check interval
        )

        self._max_error_count = 100
        self._error_count = 0
        self._memory_monitor = MemoryMonitor(
            max_ram_utilization=0.8, max_gpu_utilization=1.0, gpu_id=None
        )
        self._mem_check_counter = 0

    @staticmethod
    def _get_timestep(episode_length: int) -> int:
        max_start = max(0, episode_length)
        return np.random.randint(0, max_start - 1)

    def load_sample(
        self, episode_idx: int, timestep: int | None = None
    ) -> BatchedTrainingSamples:
        """Load image from cache or GCS."""
        if self._mem_check_counter % CHECK_MEMORY_INTERVAL == 0:
            self._memory_monitor.check_memory()
            self._mem_check_counter = 0
        self._mem_check_counter += 1
        try:
            synced_recording: SynchronizedRecording = self.synchronized_dataset[
                episode_idx
            ]
            synced_frames = synced_recording._recording_synced.frames
            episode_length = len(synced_frames)
            timestep = timestep or self._get_timestep(episode_length)
            tensor_cache_path = self.cache_dir / f"ep_{episode_idx}_frame_{timestep}.pt"
            if tensor_cache_path.exists():
                return torch.load(tensor_cache_path, weights_only=False)
            else:
                # Check disk space periodically (based on check_interval)
                if not self.cache_manager.ensure_space_available():
                    logger.warning("Low disk space. Some cache files were removed.")

                sample = TrainingSample(
                    output_predicition_mask=torch.ones(
                        (self.output_prediction_horizon,), dtype=torch.float32
                    ),
                )
                initial_sync_point = synced_frames[0]
                sync_point = synced_frames[timestep]
                future_sync_points = synced_frames[
                    timestep + 1 : timestep + 1 + self.output_prediction_horizon
                ]
                # Padding for future sync points
                for _ in range(
                    self.output_prediction_horizon - len(future_sync_points)
                ):
                    future_sync_points.append(future_sync_points[-1])

                if sync_point.rgb_images:
                    number_of_frames_to_get = 1
                    if DataType.RGB_IMAGE in self.output_data_types:
                        number_of_frames_to_get = self.output_prediction_horizon + 1
                    rgbs = self._get_video_frames(
                        episode_idx,
                        "rgbs",
                        sync_point.rgb_images,
                        initial_sync_point.rgb_images,
                        number_of_frames_to_get=number_of_frames_to_get,
                    )
                    initial_frames = [rgb[0] for rgb in rgbs]
                    rgb_images = self._create_camera_maskable_input_data(initial_frames)
                    if DataType.RGB_IMAGE in self.input_data_types:
                        sample.inputs.rgb_images = rgb_images
                    if DataType.RGB_IMAGE in self.output_data_types:
                        future_frames = [rgb[1:] for rgb in rgbs]
                        rgb_images = self._create_camera_maskable_outout_data(
                            future_frames
                        )
                        sample.outputs.rgb_images = rgb_images

                if sync_point.depth_images:
                    number_of_frames_to_get = 1
                    if DataType.DEPTH_IMAGE in self.output_data_types:
                        number_of_frames_to_get = self.output_prediction_horizon + 1
                    depths = self._get_video_frames(
                        episode_idx,
                        "depths",
                        sync_point.rgb_images,
                        initial_sync_point.rgb_images,
                        number_of_frames_to_get=number_of_frames_to_get,
                    )
                    initial_frames = [depth[0] for depth in depths]
                    batched_depth_data = self._create_camera_maskable_input_data(depths)
                    if DataType.DEPTH_IMAGE in self.input_data_types:
                        depth_in_m = rgb_to_depth(
                            batched_depth_data.data.numpy() * 255.0
                        )
                        batched_depth_data.data = torch.tensor(
                            depth_in_m, dtype=torch.float32
                        )
                        sample.inputs.depth_images = batched_depth_data
                    if DataType.DEPTH_IMAGE in self.output_data_types:
                        future_frames = [rgb[1:] for rgb in rgbs]
                        depth_data = self._create_camera_maskable_outout_data(
                            future_frames
                        )
                        depth_in_m = rgb_to_depth(depth_data.data.numpy() * 255.0)
                        batched_depth_data.data = torch.tensor(
                            depth_in_m, dtype=torch.float32
                        )
                        sample.outputs.depth_images = batched_depth_data

                if sync_point.joint_positions:
                    if DataType.JOINT_POSITIONS in self.input_data_types:
                        sample.inputs.joint_positions = (
                            self._create_joint_maskable_input_data(
                                sync_point.joint_positions,
                                self.dataset_description.joint_positions.max_len,
                            )
                        )

                    if DataType.JOINT_POSITIONS in self.output_data_types:
                        sample.outputs.joint_positions = (
                            self._create_joint_maskable_output_data(
                                [sp.joint_positions for sp in future_sync_points],
                                self.dataset_description.joint_positions.max_len,
                            )
                        )

                if sync_point.joint_velocities:
                    if DataType.JOINT_VELOCITIES in self.input_data_types:
                        sample.inputs.joint_velocities = (
                            self._create_joint_maskable_input_data(
                                sync_point.joint_velocities,
                                self.dataset_description.joint_velocities.max_len,
                            )
                        )
                    if DataType.JOINT_VELOCITIES in self.output_data_types:
                        sample.outputs.joint_velocities = (
                            self._create_joint_maskable_output_data(
                                [sp.joint_velocities for sp in future_sync_points],
                                self.dataset_description.joint_velocities.max_len,
                            )
                        )

                if sync_point.joint_torques:
                    if DataType.JOINT_TORQUES in self.input_data_types:
                        sample.inputs.joint_torques = (
                            self._create_joint_maskable_input_data(
                                sync_point.joint_torques,
                                self.dataset_description.joint_torques.max_len,
                            )
                        )
                    if DataType.JOINT_TORQUES in self.output_data_types:
                        sample.outputs.joint_torques = (
                            self._create_joint_maskable_output_data(
                                [sp.joint_torques for sp in future_sync_points],
                                self.dataset_description.joint_torques.max_len,
                            )
                        )

                if sync_point.joint_target_positions:
                    if DataType.JOINT_TARGET_POSITIONS in self.input_data_types:
                        sample.inputs.joint_target_positions = (
                            self._create_joint_maskable_input_data(
                                sync_point.joint_target_positions,
                                self.dataset_description.joint_target_positions.max_len,
                            )
                        )
                    if DataType.JOINT_TARGET_POSITIONS in self.output_data_types:
                        # We dont need to shift the sync_point by 1, since we are
                        # using the target joint positions as the action
                        jtp_points = synced_frames[
                            timestep : timestep + self.output_prediction_horizon
                        ]
                        for _ in range(
                            self.output_prediction_horizon - len(jtp_points)
                        ):
                            jtp_points.append(jtp_points[-1])
                        sample.outputs.joint_target_positions = (
                            self._create_joint_maskable_output_data(
                                [sp.joint_target_positions for sp in jtp_points],
                                self.dataset_description.joint_target_positions.max_len,
                            )
                        )

                if sync_point.language_data:
                    input_ids, attention_mask = self.tokenize_text(
                        [sync_point.language_data.text]
                    )

                    language_tokens = MaskableData(input_ids, attention_mask)
                    if DataType.LANGUAGE in self.input_data_types:
                        sample.inputs.language_tokens = language_tokens
                    if DataType.LANGUAGE in self.output_data_types:
                        sample.outputs.language_tokens = language_tokens

                sample.output_predicition_mask = self._create_output_prediction_mask(
                    episode_length,
                    timestep,
                    self.output_prediction_horizon,
                )

                torch.save(sample, tensor_cache_path)

            return sample

        except Exception as e:
            logger.error(
                f"Error loading frame {timestep} from episode {episode_idx}: {str(e)}"
            )
            raise e

    def _create_joint_maskable_input_data(
        self, joint_data: JointData, max_len: int
    ) -> MaskableData:
        jdata = torch.tensor(list(joint_data.values.values()), dtype=torch.float32)
        num_existing_states = jdata.shape[0]
        extra_states = max_len - num_existing_states
        if extra_states > 0:
            jdata = torch.cat(
                [jdata, torch.zeros(extra_states, dtype=torch.float32)], dim=0
            )
        jdata_mask = torch.tensor(
            [1.0] * num_existing_states + [0.0] * extra_states, dtype=torch.float32
        )
        return MaskableData(jdata, jdata_mask)

    def _create_joint_maskable_output_data(
        self, joint_data: list[JointData], max_len: int
    ) -> MaskableData:
        maskable_data_for_each_t = [
            self._create_joint_maskable_input_data(jd, max_len) for jd in joint_data
        ]
        stacked_maskable_data = torch.stack(
            [maskable_data.data for maskable_data in maskable_data_for_each_t]
        )
        stacked_maskable_mask = torch.stack(
            [maskable_data.mask for maskable_data in maskable_data_for_each_t]
        )
        return MaskableData(stacked_maskable_data, stacked_maskable_mask)

    def _create_output_prediction_mask(
        self, episode_length: int, timestep: int, output_prediction_horizon: int
    ) -> torch.FloatTensor:
        output_prediction_mask = torch.zeros(
            output_prediction_horizon, dtype=torch.float32
        )
        for i in range(output_prediction_horizon):
            if timestep + i >= episode_length:
                break
            else:
                output_prediction_mask[i] = 1.0
        return output_prediction_mask

    def _create_camera_maskable_input_data(
        self, camera_data: list[Image.Image]
    ) -> MaskableData:
        # Want to create tensors of shape [CAMS, C, H, W]
        cam_image_tensors = torch.stack(
            [self.camera_transform(cam_data) for cam_data in camera_data]
        )
        num_cameras = cam_image_tensors.shape[0]
        extra_cameras = self.dataset_description.max_num_rgb_images - num_cameras
        if extra_cameras > 0:
            empty_image = torch.zeros_like(cam_image_tensors[0])
            cam_image_tensors = torch.cat(
                [cam_image_tensors, empty_image.repeat(extra_cameras, 1, 1, 1)],
                dim=0,
            )
        camera_images_mask = torch.tensor(
            [1.0] * num_cameras + [0.0] * extra_cameras,
            dtype=torch.float32,
        )
        return MaskableData(cam_image_tensors, camera_images_mask)

    def _create_camera_maskable_outout_data(
        self, camera_data: list[list[Image.Image]]
    ) -> MaskableData:
        # Inputs is list [CAMS, T, ...]
        num_cams = len(camera_data)
        num_frames = len(camera_data[0])
        maskable_data_for_each_t = [
            self._create_camera_maskable_input_data(
                [camera_data[c][i] for c in range(num_cams)]
            )
            for i in range(num_frames)
        ]
        # Gives [T, CAMS, ...]
        stacked_maskable_data = torch.stack(
            [maskable_data.data for maskable_data in maskable_data_for_each_t]
        )
        stacked_maskable_mask = torch.stack(
            [maskable_data.mask for maskable_data in maskable_data_for_each_t]
        )
        return MaskableData(stacked_maskable_data, stacked_maskable_mask)

    def _get_video_frames(
        self,
        episode_idx: int,
        camera_type: str,
        cam_metadata: dict[str, CameraData],
        t0_cam_metadata: dict[str, CameraData],
        number_of_frames_to_get: int = 1,
    ) -> list[list[Image.Image]]:
        camera_ids = list(cam_metadata.keys())
        video_cache_path = self.cache_dir / f"{episode_idx}_videos"

        if not video_cache_path.exists():
            if not self.cache_manager.ensure_space_available():
                logger.warning("Low disk space. Some cache files were removed.")
            video_cache_path.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Downloading videos for camera ids "
                f"{camera_ids} for episode {episode_idx}..."
            )
            cameras_video_bytes = self.data_streamer.load_videos(
                self.recording_ids[episode_idx],
                [f"{camera_type}/{cid}" for cid in camera_ids],
            )
            for cam_id, video_bytes in zip(camera_ids, cameras_video_bytes):
                video_path = video_cache_path / f"{cam_id}.mp4"
                with open(video_path, "wb") as f:
                    f.write(video_bytes.read())

        image_frames = []
        for cam_id, cam_data in cam_metadata.items():
            content = av.datasets.curated(str(video_cache_path / f"{cam_id}.mp4"))
            container = av.open(content)
            video_stream = container.streams.video[0]
            # TODO and NOTE:
            # Seeking to the exact frame does not seems possible.
            # I dont fully understand whats happening, but it seems to be
            # seeking in chunks of about ~5s. So we need to seek to the closest
            # frame and then read until we reach the target frame.
            start_time = t0_cam_metadata[cam_id].timestamp
            ts = cam_data.timestamp - start_time
            target_pts = int(ts / float(video_stream.time_base))
            container.seek(target_pts, stream=video_stream)
            # Find the closest frame to our target
            frames = []
            for frame in container.decode(video=0):
                frame_pts = frame.pts
                diff = frame_pts - target_pts

                if diff >= 0:
                    frames.append(Image.fromarray(frame.to_rgb().to_ndarray()))
                    if len(frames) >= number_of_frames_to_get:
                        break

            for _ in range(number_of_frames_to_get - len(frames)):
                # If we are missing frames, just use the last frame
                frames.append(frames[-1])
            image_frames.append(frames)

        return image_frames

    def __len__(self) -> int:
        return self.num_samples
