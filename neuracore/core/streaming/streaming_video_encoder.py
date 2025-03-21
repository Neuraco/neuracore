import io
import json
import logging
import queue
import threading
from fractions import Fraction

import av
import numpy as np
import requests

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.streaming.resumable_upload import ResumableUpload

logger = logging.getLogger(__name__)

PTS_FRACT = 1000000  # Timebase for pts in microseconds
CHUNK_MULTIPLE = 256 * 1024  # Chunk size multiple of 256 KiB


class StreamingVideoEncoder:
    """A video encoder that handles variable framerate and streams."""

    def __init__(
        self,
        resumable_upload: ResumableUpload,
        width: int,
        height: int,
        codec: str = "libx264",
        pixel_format: str = "yuv444p10le",
        chunk_size: int = CHUNK_MULTIPLE,
    ):
        """
        Initialize a streaming video encoder.

        Args:
            resumable_upload: Resumable upload handler
            width: Frame width
            height: Frame height
            codec: Video codec
            pixel_format: Pixel format
            chunk_size: Size of chunks to upload
        """
        self.uploader = resumable_upload
        self.width = width
        self.height = height
        self.pixel_format = pixel_format

        # Ensure chunk_size is a multiple of 256 KiB
        if chunk_size % CHUNK_MULTIPLE != 0:
            self.chunk_size = ((chunk_size // CHUNK_MULTIPLE) + 1) * CHUNK_MULTIPLE
            logger.info(
                f"Adjusted chunk size to {self.chunk_size/1024:.0f} "
                "KiB to ensure it's a multiple of {CHUNK_MULTIPLE} MiB"
            )
        else:
            self.chunk_size = chunk_size

        # Create in-memory buffer
        self.buffer = io.BytesIO()

        # Open output container to write to memory buffer
        self.container = av.open(
            self.buffer,
            mode="w",
            format="mp4",
            options={"movflags": "frag_keyframe+empty_moov"},
        )

        # Create video stream
        self.stream = self.container.add_stream(codec)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = pixel_format
        self.stream.codec_context.options = {"qp": "0", "preset": "ultrafast"}

        self.stream.time_base = Fraction(1, PTS_FRACT)

        # Keep track of timestamps
        self.first_timestamp = None
        self.last_pts = None

        # Track bytes and buffer positions
        self.total_bytes_written = 0
        self.last_upload_position = 0

        # Create a dedicated buffer for upload chunks
        self.upload_buffer = bytearray()
        self.last_write_position = 0
        self.timestamps = []

        self._streaming_done = False
        self._upload_queue = queue.Queue()
        self._upload_thread = threading.Thread(target=self._upload_loop)
        self._upload_thread.start()

    def _upload_loop(self) -> None:
        """
        Upload chunks in a separate thread.
        """
        # If final has not been called, or we still have items in the queue
        while not self._streaming_done or self._upload_queue.qsize() > 0:
            try:
                frame_data, timestamp = self._upload_queue.get(timeout=0.1)
                if frame_data is None:
                    break
                self._add_frame(frame_data, timestamp)
            except queue.Empty:
                continue

        # Flush encoder
        for packet in self.stream.encode(None):
            self.container.mux(packet)

        # Close the container to finalize the MP4
        self.container.close()

        current_pos = self.buffer.tell()
        current_chunk_size = current_pos - self.last_write_position
        self.buffer.seek(self.last_write_position)
        chunk_data = self.buffer.read(current_chunk_size)
        self.upload_buffer.extend(chunk_data)
        self.last_write_position = current_pos

        final_chunk = bytes(self.upload_buffer)
        success = self.uploader.upload_chunk(final_chunk, is_final=True)

        if not success:
            raise RuntimeError("Failed to upload final chunk")

        logger.info(
            "Video encoding and upload complete: "
            f"{self.uploader.total_bytes_uploaded} bytes"
        )
        self._upload_timestamps()

    def add_frame(self, frame_data: np.ndarray, timestamp: float) -> None:
        """
        Add frame to the video with timestamp and stream if buffer large enough.

        Args:
            frame_data: RGB frame data as numpy array with shape (height, width, 3)
            timestamp: Frame timestamp in seconds (can be irregular)
        """
        self._upload_queue.put((frame_data, timestamp))
        logger.info(f"Added frame to upload queue at {timestamp}")

    def _add_frame(self, frame_data: np.ndarray, timestamp: float) -> None:
        """
        Add frame to the video with timestamp and stream if buffer large enough.

        Args:
            frame_data: RGB frame data as numpy array with shape (height, width, 3)
            timestamp: Frame timestamp in seconds (can be irregular)
        """

        # Handle first frame timestamp
        if self.first_timestamp is None:
            self.first_timestamp = timestamp

        # Calculate pts in timebase units (microseconds)
        relative_time = timestamp - self.first_timestamp
        pts = int(relative_time * PTS_FRACT)  # Convert to microseconds

        # Ensure pts is monotonically increasing (required by most codecs)
        if self.last_pts is not None and pts <= self.last_pts:
            pts = self.last_pts + 1

        self.last_pts = pts

        # Create video frame from numpy array
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        frame = frame.reformat(format=self.pixel_format)
        frame.pts = pts

        # Encode and mux
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

        # Get current buffer position after encoding
        current_pos = self.buffer.tell()
        current_chunk_size = current_pos - self.last_write_position
        if current_chunk_size >= self.chunk_size:
            self.buffer.seek(self.last_write_position)
            chunk_data = self.buffer.read(current_chunk_size)
            self.upload_buffer.extend(chunk_data)
            self.last_write_position = current_pos
            self.buffer.seek(current_pos)
            self._upload_chunks()

        # Total bytes written
        self.total_bytes_written = current_pos
        self.timestamps.append(timestamp)

    def _upload_chunks(self) -> None:
        """
        Upload chunks of exactly chunk_size bytes if enough data is available.
        """
        # Upload complete chunks while we have enough data
        while len(self.upload_buffer) >= self.chunk_size:
            # Extract a chunk of exactly chunk_size bytes
            chunk = bytes(self.upload_buffer[: self.chunk_size])

            # Remove this chunk from our upload buffer
            self.upload_buffer = self.upload_buffer[self.chunk_size :]

            # Upload the chunk
            success = self.uploader.upload_chunk(chunk, is_final=False)
            logger.info(f"Uploaded {len(chunk)} bytes")

            if not success:
                raise RuntimeError("Failed to upload chunk")

    def finish(self) -> None:
        """
        Finish encoding and upload any remaining data.
        """
        self._upload_queue.put((None, None))
        self._streaming_done = True
        self._upload_thread.join()

    def _upload_timestamps(self) -> None:
        """
        Upload timestamps to the server.
        """
        stream_name = f"cameras/{self.uploader.camera_id}/timestamps.json"
        upload_url_response = requests.get(
            f"{API_URL}/recording/{self.uploader.recording_id}/json_upload_url?filename={stream_name}",
            headers=get_auth().get_headers(),
        )
        upload_url_response.raise_for_status()
        upload_url = upload_url_response.json()["url"]
        data = json.dumps(self.timestamps)
        logger.info(f"Uploading {len(data)} bytes to {upload_url}")
        response = requests.put(
            upload_url, headers={"Content-Length": str(len(data))}, data=data
        )
        response.raise_for_status()
        self.timestamps = []
