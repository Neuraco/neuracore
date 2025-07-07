"""This module provides a global enable manager for live data.

This streamlines disabling all live data producers useful during testing and
validation.
"""

from neuracore.core.const import LIVE_DATA_ENABLED
from neuracore.core.streaming.event_loop_utils import get_running_loop
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager

global_live_data_enabled_manager = EnabledManager(
    LIVE_DATA_ENABLED, loop=get_running_loop()
)
