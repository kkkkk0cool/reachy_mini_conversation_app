import logging
from typing import Any, Dict, Tuple, Literal

import numpy as np
from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

Direction = Literal["left", "right", "up", "down", "front"]


class MoveHead(Tool):
    """Move head in a given direction."""

    name = "move_head"
    description = "Move your head in a given direction: left, right, up, down or front."
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "up", "down", "front"],
            },
        },
        "required": ["direction"],
    }

    # mapping: direction -> args for create_head_pose
    DELTAS: Dict[str, Tuple[int, int, int, int, int, int]] = {
        "left": (0, 0, 0, 0, 0, 40),
        "right": (0, 0, 0, 0, 0, -40),
        "up": (0, 0, 0, 0, -30, 0),
        "down": (0, 0, 0, 0, 30, 0),
        "front": (0, 0, 0, 0, 0, 0),
    }

    def _get_start_pose(self, deps: ToolDependencies) -> tuple[Any, tuple[float, float], float]:
        """Use cached command state first to avoid extra motor bus reads."""
        movement_manager = deps.movement_manager
        try:
            status = movement_manager.get_status()
            last_pose = status.get("last_commanded_pose", {})
            head = last_pose.get("head")
            antennas = last_pose.get("antennas")
            body_yaw = last_pose.get("body_yaw", 0.0)
            if head is not None and antennas is not None:
                return (
                    np.array(head, dtype=np.float32),
                    (float(antennas[0]), float(antennas[1])),
                    float(body_yaw or 0.0),
                )
        except Exception as exc:
            logger.debug("Could not use cached movement pose: %s", exc)

        current_head_pose = deps.reachy_mini.get_current_head_pose()
        _, current_antennas = deps.reachy_mini.get_current_joint_positions()
        return current_head_pose, (float(current_antennas[0]), float(current_antennas[1])), float(current_antennas[0])

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Move head in a given direction."""
        direction_raw = kwargs.get("direction")
        if not isinstance(direction_raw, str):
            return {"error": "direction must be a string"}
        direction: Direction = direction_raw  # type: ignore[assignment]
        logger.info("Tool call: move_head direction=%s", direction)

        deltas = self.DELTAS.get(direction, self.DELTAS["front"])
        target = create_head_pose(*deltas, degrees=True)

        # Use new movement manager
        try:
            movement_manager = deps.movement_manager

            current_head_pose, current_antennas, current_body_yaw = self._get_start_pose(deps)

            # Create goto move
            goto_move = GotoQueueMove(
                target_head_pose=target,
                start_head_pose=current_head_pose,
                target_antennas=(0, 0),  # Reset antennas to default
                start_antennas=(
                    current_antennas[0],
                    current_antennas[1],
                ),  # Skip body_yaw
                target_body_yaw=0,  # Reset body yaw
                start_body_yaw=current_body_yaw,
                duration=deps.motion_duration_s,
            )

            movement_manager.queue_move(goto_move)
            movement_manager.set_moving_state(deps.motion_duration_s)

            return {"status": f"looking {direction}"}

        except Exception as e:
            logger.error("move_head failed")
            return {"error": f"move_head failed: {type(e).__name__}: {e}"}
