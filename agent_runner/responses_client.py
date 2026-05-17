from __future__ import annotations

from .json_action_client import JsonActionClient

# Compatibility alias for older imports. The runner uses JSON-action local execution.
ResponsesClient = JsonActionClient
