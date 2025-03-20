import numpy as np

MAX_DEPTH = 10.0


def depth_to_rgb(depth_img: np.ndarray) -> np.ndarray:
    """
    Convert a depth image (in meters) to an RGB image (uint8).
    This encodes depth values across all three channels to maximize precision.

    Args:
        depth_img: Depth image in meters as float32

    Returns:
        rgb_img: uint8 RGB image with depth encoded across all channels
    """
    # Clip depths to the maximum range
    clipped_depth = np.clip(depth_img, 0, MAX_DEPTH)

    # Normalize to 0-1 range
    normalized_depth = clipped_depth / MAX_DEPTH

    # Scale to 24-bit precision (8 bits per channel × 3 channels)
    depth_scaled = normalized_depth * (2**24 - 1)

    # Extract the contribution for each channel
    r = np.floor(depth_scaled / (256 * 256)).astype(np.uint8)
    g = np.floor((depth_scaled / 256) % 256).astype(np.uint8)
    b = np.floor(depth_scaled % 256).astype(np.uint8)

    # Stack channels to create RGB image
    rgb_img = np.stack([r, g, b], axis=-1)

    return rgb_img


def rgb_to_depth(rgb_img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB-encoded depth image back to a depth image in meters.

    Args:
        rgb_img: uint8 RGB image with depth encoded across channels
        MAX_DEPTH: Maximum depth value in meters used during encoding (default: 10.0)

    Returns:
        depth_img: Depth image in meters as float32
    """
    # Convert back to original depth
    r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]

    depth_value = (
        r.astype(np.float32) * 256 * 256
        + g.astype(np.float32) * 256
        + b.astype(np.float32)
    )

    # Convert normalized values back to meters
    depth_img = (depth_value / (2**24 - 1)) * MAX_DEPTH

    return depth_img
