import os
import sys
import time

import numpy as np

import neuracore as nc
from examples.common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO_ROOT)


def main():
    # Login and connect to robot
    nc.login()
    nc.connect_robot(
        robot_name="Mujoco VX300s",
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )

    # Create dataset
    nc.create_dataset(
        name="Point Cloud Logging Test",
        description="Testing nc.log_point_cloud() performance",
    )
    nc.start_recording()

    CAM_NAME = "test_cam"
    NUM_POINTS = 121_461
    NUM_LOGS = 2

    # Generate dummy point cloud
    points = np.random.rand(NUM_POINTS, 3).astype(np.float32)
    rgb_points = np.random.randint(0, 256, size=(NUM_POINTS, 3), dtype=np.uint8)
    extrinsics = np.eye(4, dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)

    for i in range(NUM_LOGS):
        t0 = time.perf_counter()
        nc.log_point_cloud(
            camera_id=CAM_NAME,
            points=points,
            rgb_points=rgb_points,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            timestamp=time.time(),
        )
        t1 = time.perf_counter()
        print(f"[LOG {i+1}] Time: {(t1 - t0)*1000:.2f} ms")

    nc.stop_recording()
    print("Test complete.")


if __name__ == "__main__":
    main()
