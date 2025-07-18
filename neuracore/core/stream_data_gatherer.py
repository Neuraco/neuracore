"""Synchronized data retrieval from robot sensors and remote sources.

This module provides functions to collect and synchronize the latest sensor
data from a robot, including camera feeds, joint states, and language inputs.
It supports merging data from local robot streams with data from remote
sources via the Neuracore platform's live data streaming capabilities.
"""

import time
from typing import Optional

import numpy as np

from neuracore.core.exceptions import RobotError
from neuracore.core.nc_types import CameraData, JointData, LanguageData, SyncPoint
from neuracore.core.robot import Robot
from neuracore.core.streaming.p2p.consumer.org_nodes_manager import (
    get_org_nodes_manager,
)
from neuracore.core.streaming.p2p.consumer.sync_point_parser import merge_sync_points
from neuracore.core.streaming.p2p.provider.global_live_data_enabled import (
    global_consume_live_data_manager,
)
from neuracore.core.utils.depth_utils import depth_to_rgb


class StreamDataGatherer:
    """Gathers and synchronizes data from local robot sensors and remote sources.

    This class provides a unified interface to collect the latest sensor data
    from a robot, including camera feeds, joint states, and language inputs.
    It supports merging data from local robot streams with data received from
    remote sources via WebRTC-based live data streaming.
    """

    def __init__(self, robot: Robot, include_remote: bool = True) -> None:
        """Initialise the stream data gatherer.

        Args:
            robot: The robot to get the remote data from.
            include_remote: wether to connect to remote nodes to gather their
                data. This is ignored if NEURACORE_CONSUME_LIVE_DATA is disabled.

        Raises:
            RobotError: If the robot is not initialized.
        """
        if robot.id is None:
            raise RobotError("Robot not initialized. Call init() first.")
        self.robot = robot

        self.consumer_manager = None

        if include_remote:
            org_node_manager = get_org_nodes_manager(robot.org_id)

            self.consumer_manager = org_node_manager.get_robot_consumer(
                robot_id=robot.id, robot_instance=robot.instance
            )

    @staticmethod
    def _maybe_add_existing_data(
        existing: Optional[JointData], to_add: JointData
    ) -> JointData:
        """Merge joint data from multiple streams into a single data structure.

        Combines joint data while preserving existing values and updating
        timestamps. Used to aggregate data from multiple joint streams.

        Args:
            existing: Existing joint data or None.
            to_add: New joint data to merge.

        Returns:
            Combined JointData with merged values.
        """
        # Check if the joint data already exists
        if existing is None:
            return to_add
        existing.timestamp = to_add.timestamp
        existing.values.update(to_add.values)
        if existing.additional_values and to_add.additional_values:
            existing.additional_values.update(to_add.additional_values)
        return existing

    def num_remote_nodes(self) -> int:
        """Get the number of remote nodes that should be connected.

        Based on the current information of other nodes get the number of
        remote nodes that should be connected. for this robot.

        Returns:
            The number of remote nodes that should be connected.
        """
        if not self.consumer_manager:
            return 0
        return self.consumer_manager.num_remote_nodes()

    def all_remote_nodes_connected(self) -> bool:
        """Get whether all the remote nodes are connected.

        Returns:
            True if all remote nodes are connected, False otherwise.
        """
        if not self.consumer_manager:
            return True
        return self.consumer_manager.all_remote_nodes_connected()

    def get_latest_data(self) -> SyncPoint:
        """Create a synchronized data point from current robot sensor streams.

        Collects the latest data from all active robot streams including
        cameras, joint sensors, and language inputs. Organizes the data
        into a synchronized structure with consistent timestamps.

        Returns:
            SyncPoint containing all current sensor data.

        Raises:
            NotImplementedError: If an unsupported stream type is encountered.
        """
        sync_point = SyncPoint(timestamp=time.time())
        for stream_name, stream in self.robot.list_all_streams().items():
            if "rgb" in stream_name:
                stream_data = stream.get_latest_data()
                assert isinstance(stream_data, np.ndarray)
                if sync_point.rgb_images is None:
                    sync_point.rgb_images = {}
                sync_point.rgb_images[stream_name] = CameraData(
                    timestamp=time.time(), frame=stream_data
                )
            elif "depth" in stream_name:
                stream_data = stream.get_latest_data()
                assert isinstance(stream_data, np.ndarray)
                if sync_point.depth_images is None:
                    sync_point.depth_images = {}
                sync_point.depth_images[stream_name] = CameraData(
                    timestamp=time.time(),
                    frame=depth_to_rgb(stream_data),
                )
            elif "joint_positions" in stream_name:
                stream_data = stream.get_latest_data()
                assert isinstance(stream_data, JointData)
                sync_point.joint_positions = self._maybe_add_existing_data(
                    sync_point.joint_positions, stream_data
                )
            elif "joint_velocities" in stream_name:
                stream_data = stream.get_latest_data()
                assert isinstance(stream_data, JointData)
                sync_point.joint_velocities = self._maybe_add_existing_data(
                    sync_point.joint_velocities, stream_data
                )
            elif "language" in stream_name:
                stream_data = stream.get_latest_data()
                assert isinstance(stream_data, LanguageData)
                sync_point.language_data = stream_data
            else:
                raise NotImplementedError(
                    f"Support for stream {stream_name} is not implemented yet"
                )

        if not self.consumer_manager or global_consume_live_data_manager.is_disabled():
            return sync_point

        return merge_sync_points(sync_point, self.consumer_manager.get_latest_data())
