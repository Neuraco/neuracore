"""TensorBoard server management utilities for local development.

This module provides utilities for launching and managing local TensorBoard
servers for viewing training logs.
"""

import logging
import os
import signal
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TensorboardServer:
    """Manager for local TensorBoard server instances."""

    def __init__(
        self,
        log_dir: str,
        port: int = 6006,
        host: str = "localhost",
        reload_interval: int = 30,
        purge_orphaned_data: bool = True,
        max_reload_threads: int = 1,
    ):
        """Initialize TensorBoard server manager.

        Args:
            log_dir: Directory containing TensorBoard logs.
            port: Port to serve TensorBoard on.
            host: Host to bind to.
            reload_interval: How often to reload logs (seconds).
            purge_orphaned_data: Whether to purge orphaned data.
            max_reload_threads: Maximum number of reload threads.
        """
        self.log_dir = Path(log_dir)
        self.port = port
        self.host = host
        self.reload_interval = reload_interval
        self.purge_orphaned_data = purge_orphaned_data
        self.max_reload_threads = max_reload_threads
        self.process: Optional[subprocess.Popen] = None
        self.pid_file = Path.home() / ".neuracore" / f"tensorboard_{port}.pid"

    def start(self, open_browser: bool = True, wait_for_startup: bool = True) -> str:
        """Start the TensorBoard server.

        Args:
            open_browser: Whether to automatically open browser.
            wait_for_startup: Whether to wait for server to be ready.

        Returns:
            URL of the TensorBoard server.

        Raises:
            RuntimeError: If server fails to start.
        """
        if self.is_running():
            logger.info(f"TensorBoard already running on port {self.port}")
            return self.get_url()

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Build TensorBoard command
        cmd = [
            "tensorboard",
            "--logdir",
            str(self.log_dir),
            "--port",
            str(self.port),
            "--host",
            self.host,
            "--reload_interval",
            str(self.reload_interval),
            "--max_reload_threads",
            str(self.max_reload_threads),
        ]

        if self.purge_orphaned_data:
            cmd.append("--purge_orphaned_data")

        logger.info(f"Starting TensorBoard: {' '.join(cmd)}")

        try:
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            # Save PID for cleanup
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.pid_file, "w") as f:
                f.write(str(self.process.pid))

            # Wait for startup if requested
            if wait_for_startup:
                self._wait_for_startup()

            url = self.get_url()
            logger.info(f"TensorBoard started at {url}")

            # Open browser if requested
            if open_browser:
                webbrowser.open(url)

            return url

        except FileNotFoundError:
            raise RuntimeError(
                "TensorBoard not found. Install with: pip install tensorboard"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start TensorBoard: {e}")

    def stop(self) -> None:
        """Stop the TensorBoard server."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            finally:
                self.process = None

        # Clean up PID file
        if self.pid_file.exists():
            try:
                with open(self.pid_file, "r") as f:
                    pid = int(f.read().strip())
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # Process already dead
            except (ValueError, FileNotFoundError):
                pass
            finally:
                self.pid_file.unlink(missing_ok=True)

        logger.info("TensorBoard server stopped")

    def restart(self, open_browser: bool = False) -> str:
        """Restart the TensorBoard server.

        Args:
            open_browser: Whether to open browser after restart.

        Returns:
            URL of the restarted server.
        """
        self.stop()
        time.sleep(1)  # Give it a moment to fully stop
        return self.start(open_browser=open_browser)

    def is_running(self) -> bool:
        """Check if TensorBoard server is running.

        Returns:
            True if server is running, False otherwise.
        """
        if self.process and self.process.poll() is None:
            return True

        # Check if another instance is running on this port
        try:
            import requests

            response = requests.get(self.get_url(), timeout=1)
            return response.status_code == 200
        except (requests.RequestException, ImportError):
            return False

    def get_url(self) -> str:
        """Get the TensorBoard server URL.

        Returns:
            Full URL to the TensorBoard server.
        """
        return f"http://{self.host}:{self.port}"

    def get_logs(self) -> tuple[str, str]:
        """Get stdout and stderr logs from the TensorBoard process.

        Returns:
            Tuple of (stdout, stderr) strings.
        """
        if not self.process:
            return "", ""

        try:
            stdout, stderr = self.process.communicate(timeout=0.1)
            return stdout, stderr
        except subprocess.TimeoutExpired:
            return "", ""

    def _wait_for_startup(self, timeout: int = 30) -> None:
        """Wait for TensorBoard server to be ready.

        Args:
            timeout: Maximum time to wait in seconds.

        Raises:
            RuntimeError: If server doesn't start within timeout.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.is_running():
                return

            # Check if process died
            if self.process and self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(
                    f"TensorBoard process died. " f"stdout: {stdout}, stderr: {stderr}"
                )

            time.sleep(0.5)

        raise RuntimeError(f"TensorBoard failed to start within {timeout} seconds")

    def __enter__(self) -> "TensorboardServer":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()


class TensorboardManager:
    """Manager for multiple TensorBoard instances."""

    def __init__(self) -> None:
        """Initialize the manager."""
        self.servers: dict[int, TensorboardServer] = {}

    def start_server(
        self,
        log_dir: str,
        port: int = 6006,
        open_browser: bool = True,
        **kwargs: Any,
    ) -> TensorboardServer:
        """Start a TensorBoard server.

        Args:
            log_dir: Directory containing logs.
            port: Port to serve on.
            open_browser: Whether to open browser.
            **kwargs: Additional arguments for TensorBoardServer.

        Returns:
            TensorBoardServer instance.
        """
        if port in self.servers:
            server = self.servers[port]
            if server.is_running():
                logger.info(f"TensorBoard already running on port {port}")
                return server
            else:
                # Server exists but not running, remove it
                del self.servers[port]

        server = TensorboardServer(log_dir, port=port, **kwargs)
        server.start(open_browser=open_browser)
        self.servers[port] = server
        return server

    def stop_server(self, port: int) -> None:
        """Stop a TensorBoard server.

        Args:
            port: Port of the server to stop.
        """
        if port in self.servers:
            self.servers[port].stop()
            del self.servers[port]

    def stop_all(self) -> None:
        """Stop all TensorBoard servers."""
        for server in list(self.servers.values()):
            server.stop()
        self.servers.clear()

    def list_servers(self) -> dict[int, dict]:
        """List all running servers.

        Returns:
            Dictionary mapping port to server info.
        """
        info = {}
        for port, server in self.servers.items():
            info[port] = {
                "url": server.get_url(),
                "log_dir": str(server.log_dir),
                "running": server.is_running(),
            }
        return info

    def get_server(self, port: int) -> Optional[TensorboardServer]:
        """Get a server by port.

        Args:
            port: Port of the server.

        Returns:
            TensorBoardServer instance or None if not found.
        """
        return self.servers.get(port)


# Global manager instance
_tb_manager = TensorboardManager()


def start_tensorboard(
    log_dir: str,
    port: int = 6006,
    open_browser: bool = True,
    **kwargs: Any,
) -> TensorboardServer:
    """Start a TensorBoard server (convenience function).

    Args:
        log_dir: Directory containing TensorBoard logs.
        port: Port to serve on.
        open_browser: Whether to open browser.
        **kwargs: Additional arguments for TensorBoardServer.

    Returns:
        TensorBoardServer instance.
    """
    return _tb_manager.start_server(log_dir, port, open_browser, **kwargs)


def stop_tensorboard(port: int = 6006) -> None:
    """Stop a TensorBoard server (convenience function).

    Args:
        port: Port of the server to stop.
    """
    _tb_manager.stop_server(port)


def list_tensorboard_servers() -> dict[int, dict]:
    """List all running TensorBoard servers.

    Returns:
        Dictionary mapping port to server info.
    """
    return _tb_manager.list_servers()


def cleanup_tensorboard_servers() -> None:
    """Stop all TensorBoard servers."""
    _tb_manager.stop_all()
